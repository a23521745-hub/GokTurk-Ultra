"""Evaluation harness.

Three separate things, deliberately not mixed:

``format``  Schema/quality validation of a JSONL dataset. This says nothing
            about model quality.
``score``   Deterministic, rule-based scoring of *already produced* model
            outputs against a fixed evaluation set. Reproducible and cheap;
            it measures task success on our own small set, not a public
            benchmark.
``judge``   Optional teacher-model judging (LLM-as-judge). Off by default and
            clearly reported as unvalidated, because an unvalidated judge can
            inflate scores.

Nothing here reports accuracy, "Trojan catch rate", or a benchmark number
unless the corresponding run actually happened. Unmeasured categories are
reported as ``null`` / ``ölçülmedi``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ConfigError, QualityConfig, load_quality_config
from .data_quality import assess_record, read_jsonl
from .teachers import TeacherClient, TeacherError

LOGGER = logging.getLogger("gokturk.evaluate")

CATEGORIES = ("turkish_language", "instruction_following", "coding", "tool_calling")

CHECK_TYPES = (
    "contains_all",
    "contains_any",
    "not_contains",
    "regex",
    "json_valid",
    "json_keys",
    "min_words",
    "max_words",
    "code_fence",
    "starts_with",
    "ends_with",
    "min_lines",
)


# ---------------------------------------------------------------------------
# eval set
# ---------------------------------------------------------------------------
@dataclass
class EvalItem:
    id: str
    category: str
    prompt: str
    checks: list[dict[str, Any]]
    weight: float = 1.0
    notes: str = ""

    def validate(self, index: int = 0) -> None:
        if not self.id.strip():
            raise ConfigError(f"eval[{index}]: id empty")
        if self.category not in CATEGORIES:
            raise ConfigError(
                f"eval[{index}] ({self.id}): category {self.category!r} not in {list(CATEGORIES)}"
            )
        if not self.prompt.strip():
            raise ConfigError(f"eval[{index}] ({self.id}): prompt empty")
        if not self.checks:
            raise ConfigError(f"eval[{index}] ({self.id}): at least one check required")
        for check in self.checks:
            kind = check.get("type")
            if kind not in CHECK_TYPES:
                raise ConfigError(
                    f"eval[{index}] ({self.id}): unknown check type {kind!r}; "
                    f"known: {list(CHECK_TYPES)}"
                )
            if kind != "code_fence" and "value" not in check and kind not in (
                "json_valid",
                "code_fence",
            ):
                raise ConfigError(f"eval[{index}] ({self.id}): check {kind} needs 'value'")
        if self.weight <= 0:
            raise ConfigError(f"eval[{index}] ({self.id}): weight must be > 0")


def load_eval_set(path: str | Path) -> list[EvalItem]:
    records, issues = read_jsonl(path)
    if issues:
        raise ConfigError(
            f"{path}: {len(issues)} invalid line(s); first: {issues[0].reason} "
            f"(line {issues[0].line})"
        )
    items: list[EvalItem] = []
    for i, rec in enumerate(records):
        item = EvalItem(
            id=str(rec.get("id", "")),
            category=str(rec.get("category", "")),
            prompt=str(rec.get("prompt", "")),
            checks=list(rec.get("checks", [])),
            weight=float(rec.get("weight", 1.0)),
            notes=str(rec.get("notes", "")),
        )
        item.validate(i)
        items.append(item)
    if not items:
        raise ConfigError(f"{path}: evaluation set is empty")
    dupes = {i.id for i in items if [j.id for j in items].count(i.id) > 1}
    if dupes:
        raise ConfigError(f"{path}: duplicate ids {sorted(dupes)}")
    return items


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
_TOOL_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return "\n".join(blocks) if blocks else text


def _extract_json(text: str) -> Any | None:
    for candidate in (text.strip(), _strip_code_fences(text).strip()):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def run_check(check: Mapping[str, Any], response: str) -> tuple[bool, str]:
    kind = check["type"]
    value = check.get("value")
    text = response.strip()
    lowered = text.casefold()

    if kind == "contains_all":
        missing = [v for v in value if str(v).casefold() not in lowered]
        return (not missing, f"eksik: {missing}" if missing else "tamam")
    if kind == "contains_any":
        hit = [v for v in value if str(v).casefold() in lowered]
        return (bool(hit), f"bulunan: {hit}")
    if kind == "not_contains":
        hit = [v for v in value if str(v).casefold() in lowered]
        return (not hit, f"istenmeyen: {hit}" if hit else "tamam")
    if kind == "regex":
        pattern = re.compile(str(value), flags=re.MULTILINE)
        return (bool(pattern.search(text)), f"desen: {value!r}")
    if kind == "json_valid":
        return (_extract_json(text) is not None, "geçerli JSON bulunamadı")
    if kind == "json_keys":
        obj = _extract_json(text)
        if not isinstance(obj, Mapping):
            return (False, "JSON nesnesi yok")
        missing = [k for k in value if k not in obj]
        return (not missing, f"eksik anahtar: {missing}" if missing else "tamam")
    if kind == "min_words":
        n = len(text.split())
        return (n >= int(value), f"{n} kelime (>= {value})")
    if kind == "max_words":
        n = len(text.split())
        return (n <= int(value), f"{n} kelime (<= {value})")
    if kind == "min_lines":
        n = len([ln for ln in text.splitlines() if ln.strip()])
        return (n >= int(value), f"{n} satır (>= {value})")
    if kind == "code_fence":
        has = "```" in text
        lang = check.get("language")
        if has and lang:
            has = bool(re.search(rf"```{re.escape(str(lang))}", text))
        return (has, "kod bloğu " + ("var" if has else "yok"))
    if kind == "starts_with":
        return (lowered.startswith(str(value).casefold()), f"başlangıç: {value!r}")
    if kind == "ends_with":
        return (lowered.rstrip().endswith(str(value).casefold()), f"bitiş: {value!r}")
    return (False, f"bilinmeyen kontrol: {kind}")  # pragma: no cover


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@dataclass
class ItemResult:
    id: str
    category: str
    passed: bool
    check_details: list[dict[str, Any]] = field(default_factory=list)
    response_preview: str = ""
    missing_prediction: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CategoryScore:
    category: str
    n_items: int
    n_passed: int
    pass_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    eval_set: str
    predictions: str
    scored_at: str
    n_items: int
    n_scored: int
    overall_pass_rate: float | None
    weighted_pass_rate: float | None
    categories: list[CategoryScore]
    items: list[ItemResult]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def render_markdown(self) -> str:
        lines = [
            "# Değerlendirme raporu",
            "",
            f"- Değerlendirme seti: `{self.eval_set}`",
            f"- Tahmin dosyası: `{self.predictions}`",
            f"- Zaman: {self.scored_at}",
            f"- Öğe sayısı: {self.n_items}, skorlanan: {self.n_scored}",
            "",
            "| Kategori | Öğe | Geçen | Geçme oranı |",
            "| --- | --- | --- | --- |",
        ]
        for c in self.categories:
            rate = "ölçülmedi" if c.pass_rate is None else f"{c.pass_rate * 100:.1f}%"
            lines.append(f"| {c.category} | {c.n_items} | {c.n_passed} | {rate} |")
        overall = (
            "ölçülmedi"
            if self.overall_pass_rate is None
            else f"{self.overall_pass_rate * 100:.1f}%"
        )
        weighted = (
            "ölçülmedi"
            if self.weighted_pass_rate is None
            else f"{self.weighted_pass_rate * 100:.1f}%"
        )
        lines += [
            "",
            f"**Genel geçme oranı:** {overall}  ",
            f"**Ağırlıklı geçme oranı:** {weighted}",
            "",
            "## Sınırlamalar",
            "",
        ]
        lines += [f"- {c}" for c in self.caveats]
        failed = [i for i in self.items if not i.passed]
        if failed:
            lines += ["", "## Başarısız örnekler", ""]
            for item in failed[:20]:
                reason = "; ".join(
                    f"{d['type']}: {d['detail']}" for d in item.check_details if not d["passed"]
                )
                lines.append(f"- `{item.id}` ({item.category}): {reason}")
        return "\n".join(lines) + "\n"


DEFAULT_CAVEATS = [
    "Bu skor yalnızca bu repodaki sabit, küçük değerlendirme setinde ölçülen "
    "kural tabanlı geçme oranıdır; genel model kalitesi veya kamuya açık benchmark "
    "sonucu değildir.",
    "Kontroller deterministik ve yüzeyseldir (anahtar kelime, JSON şeması, uzunluk). "
    "Anlam doğruluğunu ölçmez.",
    "Ölçülmemiş kategoriler 'ölçülmedi' olarak raporlanır; eksik veri sıfır sayılmaz.",
]


def score_predictions(
    items: Sequence[EvalItem],
    predictions: Mapping[str, str],
    *,
    eval_set_path: str = "",
    predictions_path: str = "",
) -> EvalReport:
    results: list[ItemResult] = []
    for item in items:
        if item.id not in predictions:
            results.append(
                ItemResult(
                    id=item.id,
                    category=item.category,
                    passed=False,
                    check_details=[{"type": "prediction", "passed": False, "detail": "tahmin yok"}],
                    missing_prediction=True,
                )
            )
            continue
        response = predictions[item.id]
        details: list[dict[str, Any]] = []
        all_passed = True
        for check in item.checks:
            ok, detail = run_check(check, response)
            details.append({"type": check["type"], "passed": ok, "detail": detail})
            all_passed = all_passed and ok
        results.append(
            ItemResult(
                id=item.id,
                category=item.category,
                passed=all_passed,
                check_details=details,
                response_preview=response[:300],
            )
        )

    categories: list[CategoryScore] = []
    for cat in CATEGORIES:
        rows = [r for r in results if r.category == cat]
        if not rows:
            categories.append(CategoryScore(cat, 0, 0, None))
            continue
        passed = sum(1 for r in rows if r.passed)
        categories.append(CategoryScore(cat, len(rows), passed, passed / len(rows)))

    n_scored = sum(1 for r in results if not r.missing_prediction)
    scored_rows = [r for r in results if not r.missing_prediction]
    overall = (
        sum(1 for r in scored_rows if r.passed) / n_scored if n_scored else None
    )
    weight_by_id = {i.id: i.weight for i in items}
    total_w = sum(weight_by_id[r.id] for r in scored_rows)
    weighted = (
        sum(weight_by_id[r.id] for r in scored_rows if r.passed) / total_w
        if total_w and n_scored
        else None
    )

    return EvalReport(
        eval_set=eval_set_path,
        predictions=predictions_path,
        scored_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_items=len(items),
        n_scored=n_scored,
        overall_pass_rate=overall,
        weighted_pass_rate=weighted,
        categories=categories,
        items=results,
        caveats=list(DEFAULT_CAVEATS),
    )


# ---------------------------------------------------------------------------
# format validation mode
# ---------------------------------------------------------------------------
def format_check(path: str | Path, quality: QualityConfig | None = None) -> dict[str, Any]:
    """Validate file format + quality heuristics. Not a model metric."""
    cfg = quality or QualityConfig()
    records, issues = read_jsonl(path)
    problems: dict[str, list[str]] = {}
    for rec in records:
        rec_problems = assess_record(rec, cfg)
        if rec_problems:
            problems[str(rec.get("id", "<no-id>"))] = rec_problems
    acceptable = len(records) - len(problems)
    return {
        "mode": "format",
        "path": str(path),
        "note": "Bu çıktı yalnızca veri biçimi/kalitesi hakkındadır; model kalitesi ölçmez.",
        "records": len(records),
        "parse_errors": [asdict(i) for i in issues],
        "records_with_problems": len(problems),
        "acceptable": acceptable,
        "acceptable_ratio": (acceptable / len(records)) if records else 0.0,
        "problems": dict(list(problems.items())[:50]),
    }


# ---------------------------------------------------------------------------
# optional teacher judge (unvalidated)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "Bir yanıtı değerlendiren titiz bir Türkçe değerlendiricisin. Yalnızca JSON "
    "döndür: {\"puan\": 1-5, \"gerekce\": \"<kısa Türkçe gerekçe>\"}. "
    "Puanla: 5 = tam doğru ve eksiksiz, 1 = yanlış veya konu dışı."
)

JUDGE_USER = """Soru/istek:
{prompt}

Yanıt:
{response}

Yukarıdaki yanıtı 1-5 arasında puanla ve JSON döndür."""


def judge_predictions(
    items: Sequence[EvalItem],
    predictions: Mapping[str, str],
    client: TeacherClient,
    model: str,
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    """LLM-as-judge scoring. Explicitly unvalidated: treat as a weak signal."""
    scores: dict[str, Any] = {}
    rows = list(items)[: max_items] if max_items else list(items)
    for item in rows:
        response = predictions.get(item.id)
        if response is None:
            continue
        try:
            resp = client.chat(
                model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": JUDGE_USER.format(prompt=item.prompt, response=response),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except TeacherError as exc:
            LOGGER.warning("judge failed for %s: %s", item.id, exc)
            continue
        parsed = _extract_json(resp.text)
        score = None
        if isinstance(parsed, Mapping):
            raw_score = parsed.get("puan")
            if isinstance(raw_score, (int, float)) and 1 <= raw_score <= 5:
                score = float(raw_score)
        scores[item.id] = {
            "score": score,
            "reason": (parsed or {}).get("gerekce") if isinstance(parsed, Mapping) else None,
            "raw": resp.text[:400],
        }
    valid = [v["score"] for v in scores.values() if v["score"] is not None]
    return {
        "mode": "judge",
        "judge_model": model,
        "n_judged": len(valid),
        "mean_score": round(statistics.fmean(valid), 3) if valid else None,
        "scores": scores,
        "caveats": [
            "LLM-as-judge puanları doğrulanmamıştır; insan değerlendirmesiyle "
            "kalibre edilmeden model kalitesi kanıtı sayılmamalıdır.",
            "Aynı model ailesinden bir yargıç kendi çıktılarına yanlı olabilir.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_predictions(path: str | Path) -> dict[str, str]:
    records, issues = read_jsonl(path)
    if issues:
        raise ConfigError(f"{path}: {len(issues)} invalid line(s) ({issues[0].reason})")
    out: dict[str, str] = {}
    for rec in records:
        item_id = rec.get("id")
        response = rec.get("response", rec.get("output", rec.get("prediction")))
        if not isinstance(item_id, str) or not isinstance(response, str):
            raise ConfigError(
                f"{path}: each line needs string 'id' and one of "
                "'response'/'output'/'prediction'"
            )
        out[item_id] = response
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GökTürk-Ultra değerlendirme araçları")
    parser.add_argument("--mode", choices=["format", "score", "judge"], default="score")
    parser.add_argument("--eval-set", default="data/eval/turkish_eval.jsonl")
    parser.add_argument("--predictions", default=None, help="id+response JSONL")
    parser.add_argument("--data", default=None, help="format modu için JSONL yolu")
    parser.add_argument("--quality-config", default=None)
    parser.add_argument("--out", default=None, help="JSON rapor yolu")
    parser.add_argument("--markdown", default=None, help="Markdown rapor yolu")
    parser.add_argument("--judge-config", default=None, help="Öğretmen config (judge modu)")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-judge-items", type=int, default=20)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        quality = load_quality_config(args.quality_config)
    except ConfigError as exc:
        print(f"Kalite yapılandırma hatası: {exc}", file=sys.stderr)
        return 4

    if args.mode == "format":
        if not args.data:
            print("--data gerekli", file=sys.stderr)
            return 4
        report = format_check(args.data, quality)
        print(json.dumps(report, ensure_ascii=False, indent=2)[:8000])
        if args.out:
            Path(args.out).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0

    if not args.predictions:
        print(
            "--predictions gerekli: model çıktılarını (id+response JSONL) üretip "
            "buraya verin. Skorlama, tahmin üretiminden ayrıdır.",
            file=sys.stderr,
        )
        return 4

    try:
        items = load_eval_set(args.eval_set)
        predictions = _load_predictions(args.predictions)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 4

    report = score_predictions(
        items, predictions, eval_set_path=args.eval_set, predictions_path=args.predictions
    )
    payload: dict[str, Any] = report.to_dict()

    if args.mode == "judge":
        if not args.judge_config or not args.judge_model:
            print("judge modu için --judge-config ve --judge-model gerekli", file=sys.stderr)
            return 4
        from .config import load_teachers_config

        teachers = load_teachers_config(args.judge_config)
        if teachers.provider.api_key() is None:
            print(
                f"Durduruldu: {teachers.provider.api_key_env} yok; judge çalıştırılmadı.",
                file=sys.stderr,
            )
            return 3
        payload["judge"] = judge_predictions(
            items,
            predictions,
            TeacherClient(teachers.provider),
            args.judge_model,
            max_items=args.max_judge_items,
        )

    markdown = report.render_markdown()
    print(markdown)
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
