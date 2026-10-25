"""Synthetic Turkish data generation using teacher models.

Two-step pipeline (configurable):

1. **Draft** -- a teacher is asked to invent a realistic Turkish user prompt for
   a given task family and topic. The draft is parsed as JSON; malformed drafts
   are rejected, never repaired silently.
2. **Solve** -- a teacher answers the drafted prompt. The answer becomes the
   training target. It is filtered by :mod:`gokturk.data_quality` before being
   accepted, and it is *not* assumed to be correct: teacher outputs still need
   human spot-checking (see ``docs/architecture.md``).

Safety behaviour
----------------
* No API key -> the run stops before a single request is sent.
* Authentication failure (401/403) -> immediate abort.
* Repeated transport failures -> abort after
  ``provider.abort_after_consecutive_failures`` consecutive failures.
* Every accepted record is appended to disk immediately, so an aborted run
  keeps its partial data and can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    ProviderConfig,
    QualityConfig,
    TeacherModelConfig,
    TeachersConfig,
    load_quality_config,
    load_teachers_config,
)
from .data_quality import assess_record, read_jsonl
from .schema import SCHEMA_VERSION, TASK_TYPES
from .teachers import (
    RETRYABLE_ERRORS,
    TeacherAuthError,
    TeacherClient,
    TeacherConfigError,
    TeacherError,
    TeacherPool,
    TeacherRequestError,
)

LOGGER = logging.getLogger("gokturk.data_generation")

PROMPT_VERSION = "v1"

DRAFTER_SYSTEM = (
    "Sen Türkçe bir veri seti tasarımcısısın. Görevin, verilen görev türü ve konu "
    "için gerçek bir kullanıcının yazacağı tek turlu, doğal ve net bir Türkçe "
    "istek (prompt) üretmektir. Kullanıcı isteği kısa ama yeterince "
    "belirgin olmalı, ek bağlam gerektirmemeli ve modelin yanıtlayabileceği "
    "somut bir çıktı istemelidir. Yalnızca JSON döndür."
)

DRAFTER_USER = """Görev türü: {task_type}
Konu: {topic}
Zorluk: {level}
Varyasyon numarası: {variant}

Şu şemada SADECE JSON döndür (açıklama, markdown, kod bloğu yok):
{{"prompt": "<Türkçe kullanıcı isteği>"}}

Kurallar:
- İstek Türkçe olacak ve {task_type} görev türüne uygun olacak.
- İstek, cevabı verilecek asıl görevi içerecek; "sen bir uzmansın" gibi rol
  yönergeleri kullanıcı isteğinde yer almayacak.
- 10 ile 90 kelime arasında olacak.
- Daha önce üretilmiş örneklerden farklı olacak."""

ANSWER_SYSTEM = (
    "Sen GökTürk-Ultra için eğitim verisi üreten, alanında uzman bir Türkçe "
    "asistansın. Yanıtlarını her zaman Türkçe, doğru, eksiksiz ve doğrudan "
    "kullanıcının istediği biçimde verirsin. Gereksiz giriş cümlesi kurmazsın. "
    "Kod istendiğinde çalışır kod bloğu verirsin. Araç çağrısı istendiğinde "
    "istenen JSON şemasına birebir uyarsın. Bilmediğin bir şeyi uydurmazsın."
)

ANSWER_USER = """Kullanıcı isteği:
\"\"\"
{prompt}
\"\"\""""

TOOL_FORMALISM = (
    "\n\nBu örnek bir araç çağrısı görevidir. Yanıt, yalnızca aşağıdaki şemaya "
    "uyan bir JSON nesnesi olsun:\n"
    '{{"tool": "<araç_adı>", "arguments": {{...}}}}\n'
    "Açıklama, markdown veya ek metin ekleme."
)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
@dataclass
class SeedTask:
    id: str
    task_type: str
    topic: str
    level: str = "orta"
    prompt: str | None = None  # fixed prompt skips the draft step
    n_variants: int = 1
    notes: str | None = None

    def validate(self, index: int = 0) -> None:
        if not self.id.strip():
            raise ConfigError(f"seed[{index}]: id must not be empty")
        if self.task_type not in TASK_TYPES:
            raise ConfigError(
                f"seed[{index}] ({self.id}): task_type {self.task_type!r} not in {list(TASK_TYPES)}"
            )
        if not self.topic.strip() and not (self.prompt and self.prompt.strip()):
            raise ConfigError(f"seed[{index}] ({self.id}): need a topic or a fixed prompt")
        if self.n_variants < 1:
            raise ConfigError(f"seed[{index}] ({self.id}): n_variants must be >= 1")


def load_seeds(path: str | Path) -> list[SeedTask]:
    records, issues = read_jsonl(path)
    if issues:
        raise ConfigError(
            f"{path}: {len(issues)} invalid line(s); first: line {issues[0].line}: {issues[0].reason}"
        )
    known = {f for f in SeedTask.__dataclass_fields__}
    seeds: list[SeedTask] = []
    for i, rec in enumerate(records):
        unknown = set(rec) - known
        if unknown:
            raise ConfigError(f"{path}: seed[{i}] has unknown field(s) {sorted(unknown)}")
        seed = SeedTask(**{k: v for k, v in rec.items() if k in known})
        seed.validate(i)
        seeds.append(seed)
    if not seeds:
        raise ConfigError(f"{path}: no seeds found")
    ids = [s.id for s in seeds]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError(f"{path}: duplicate seed id(s) {sorted(dupes)}")
    return seeds


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def build_draft_request(seed: SeedTask, variant: int) -> tuple[str, str]:
    return (
        DRAFTER_SYSTEM,
        DRAFTER_USER.format(
            task_type=seed.task_type,
            topic=seed.topic,
            level=seed.level,
            variant=variant,
        ),
    )


def build_answer_request(seed: SeedTask, prompt: str) -> tuple[str, str]:
    user = ANSWER_USER.format(prompt=prompt)
    if seed.task_type == "tool_calling":
        user += TOOL_FORMALISM
    return ANSWER_SYSTEM, user


def parse_draft(text: str) -> str | None:
    """Extract ``prompt`` from a drafter response. Returns None if unusable."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = [ln for ln in candidate.splitlines() if not ln.strip().startswith("```")]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, Mapping):
        return None
    prompt = obj.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 10:
        return None
    return prompt.strip()


def make_record_id(seed_id: str, teacher: str, variant: int, task_type: str) -> str:
    digest = hashlib.sha1(f"{seed_id}|{teacher}|{variant}".encode()).hexdigest()[:10]
    return f"gtu-{task_type[:3]}-{digest}"


def build_record(
    *,
    seed: SeedTask,
    prompt: str,
    response: str,
    teacher: TeacherModelConfig,
    provider: ProviderConfig,
    variant: int,
    response_meta: Mapping[str, Any],
    draft_teacher: str | None = None,
) -> dict[str, Any]:
    return {
        "id": make_record_id(seed.id, teacher.name, variant, seed.task_type),
        "prompt": prompt,
        "response": response,
        "source": {
            "kind": "teacher",
            "teacher": teacher.name,
            "teacher_model": teacher.model,
            "endpoint": provider.base_url,
            "task_type": seed.task_type,
            "template_version": PROMPT_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_id": seed.id,
            "draft_teacher": draft_teacher,
            "finish_reason": response_meta.get("finish_reason"),
            "truncated": response_meta.get("finish_reason") == "length",
        },
        "meta": {
            "lang": "tr",
            "level": seed.level,
            "variant": variant,
            "latency_s": response_meta.get("latency_s"),
            "usage": response_meta.get("usage", {}),
        },
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@dataclass
class GenerationStats:
    requested: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    skipped_existing: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)
    teachers_used: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        key = reason.split("(")[0].split(":")[0].strip()[:80]
        self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationAborted(RuntimeError):
    """Raised when generation stops safely; carries the partial statistics."""

    def __init__(self, message: str, stats: GenerationStats | None = None) -> None:
        super().__init__(message)
        self.stats = stats


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    records, _ = read_jsonl(path)
    return {str(r.get("id", "")) for r in records if r.get("id")}


def generate(
    teachers: TeachersConfig,
    seeds: Sequence[SeedTask],
    *,
    out_path: str | Path,
    per_seed: int = 1,
    resume: bool = True,
    quality: QualityConfig | None = None,
    client: TeacherClient | None = None,
    pool: TeacherPool | None = None,
    sleep_between: float = 0.0,
    max_records: int | None = None,
) -> GenerationStats:
    """Run the pipeline. Returns stats; raises on fatal teacher errors."""
    qcfg = quality or QualityConfig()
    qcfg.validate()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = out.with_name(out.stem + ".rejects.jsonl")

    teacher_pool = pool or TeacherPool(teachers, client=client)
    stats = GenerationStats()
    existing = _existing_ids(out) if resume else set()
    if not resume and out.exists():
        out.unlink()

    abort_after = teachers.provider.abort_after_consecutive_failures

    with out.open("a", encoding="utf-8") as out_fh, rejects_path.open(
        "a", encoding="utf-8"
    ) as rej_fh:
        for seed in seeds:
            variants = max(per_seed, seed.n_variants)
            for variant in range(variants):
                if max_records and stats.accepted >= max_records:
                    LOGGER.info("max_records sınırına ulaşıldı (%d)", max_records)
                    return stats
                stats.requested += 1
                try:
                    _generate_one(
                        seed=seed,
                        variant=variant,
                        teachers=teachers,
                        pool=teacher_pool,
                        qcfg=qcfg,
                        stats=stats,
                        existing=existing,
                        out_fh=out_fh,
                        rej_fh=rej_fh,
                    )
                except (TeacherAuthError, TeacherConfigError) as exc:
                    stats.aborted = True
                    stats.abort_reason = f"{type(exc).__name__}: {exc}"
                    LOGGER.error("Öğretmen erişimi kritik hata verdi: %s", exc)
                    raise GenerationAborted(stats.abort_reason, stats) from exc
                except RETRYABLE_ERRORS as exc:
                    stats.failed += 1
                    consecutive = teacher_pool.register_failure()
                    LOGGER.warning("İstek başarısız: %s", exc)
                    if consecutive >= abort_after:
                        stats.aborted = True
                        stats.abort_reason = (
                            f"{consecutive} ardışık hata; öğretmen kullanılamıyor gibi görünüyor"
                        )
                        raise GenerationAborted(stats.abort_reason, stats) from exc
                except TeacherRequestError as exc:
                    # Bad request (e.g. wrong model id): fatal, retrying is pointless.
                    stats.aborted = True
                    stats.abort_reason = f"{type(exc).__name__}: {exc}"
                    raise GenerationAborted(stats.abort_reason, stats) from exc
                except TeacherError as exc:  # pragma: no cover - defensive
                    stats.failed += 1
                    LOGGER.warning("Beklenmeyen öğretmen hatası: %s", exc)
                if sleep_between:
                    time.sleep(sleep_between)
    return stats


def _generate_one(
    *,
    seed: SeedTask,
    variant: int,
    teachers: TeachersConfig,
    pool: TeacherPool,
    qcfg: QualityConfig,
    stats: GenerationStats,
    existing: set[str],
    out_fh,
    rej_fh,
) -> None:
    # ---- draft ---------------------------------------------------------
    if seed.prompt:
        prompt = seed.prompt
        draft_teacher = None
    else:
        sys_prompt, user_prompt = build_draft_request(seed, variant)
        draft_resp, draft_model = pool.ask(sys_prompt, user_prompt)
        prompt = parse_draft(draft_resp.text)
        if prompt is None:
            stats.reject("drafter returned unusable JSON")
            rej_fh.write(
                json.dumps(
                    {"seed_id": seed.id, "variant": variant, "reason": "bad_draft",
                     "raw_preview": draft_resp.text[:200]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            return
        draft_teacher = draft_model.name

    # ---- answer --------------------------------------------------------
    sys_prompt, user_prompt = build_answer_request(seed, prompt)
    answer_resp, teacher = pool.ask(sys_prompt, user_prompt)
    pool.reset_failures()

    record = build_record(
        seed=seed,
        prompt=prompt,
        response=answer_resp.text,
        teacher=teacher,
        provider=teachers.provider,
        variant=variant,
        response_meta=answer_resp.as_metadata(),
        draft_teacher=draft_teacher,
    )
    if record["id"] in existing:
        stats.skipped_existing += 1
        return

    problems = assess_record(record, qcfg)
    if problems:
        stats.reject(problems[0])
        rej_fh.write(
            json.dumps(
                {"id": record["id"], "reason": problems, "prompt": prompt[:300]},
                ensure_ascii=False,
            )
            + "\n"
        )
        rej_fh.flush()
        return

    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_fh.flush()
    existing.add(record["id"])
    stats.accepted += 1
    stats.teachers_used[teacher.name] = stats.teachers_used.get(teacher.name, 0) + 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Öğretmen modellerle sentetik veri üretimi")
    parser.add_argument("--teachers-config", default="configs/teachers.yaml")
    parser.add_argument("--seeds", default="data/seeds/turkish_seed_tasks.jsonl")
    parser.add_argument("--out", default="data/raw/generated.jsonl")
    parser.add_argument("--per-seed", type=int, default=1)
    parser.add_argument("--quality-config", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--health-check", action="store_true", help="Yalnızca bağlantıyı doğrula")
    parser.add_argument("--dry-run", action="store_true", help="Ağa çıkmadan planı yazdır")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        teachers = load_teachers_config(args.teachers_config)
        seeds = load_seeds(args.seeds)
        quality = load_quality_config(args.quality_config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Yapılandırma hatası: {exc}", file=sys.stderr)
        return 4

    provider = teachers.provider
    key_present = provider.api_key() is not None
    total_requests = sum(max(args.per_seed, s.n_variants) for s in seeds)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider": provider.name,
                    "base_url": provider.base_url or "(boş — doğrulanmalı)",
                    "api_key_env": provider.api_key_env,
                    "api_key_present": key_present,
                    "teachers": [t.to_public_dict() for t in teachers.models()],
                    "seeds": len(seeds),
                    "planned_records": total_requests,
                    "tasks": sorted({s.task_type for s in seeds}),
                    "out": args.out,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not provider.base_url:
            print(
                "\nUYARI: provider.base_url boş. "
                "configs/teachers.yaml içine sağlayıcının doğrulanmış adresini yazın.",
                file=sys.stderr,
            )
        if not key_present:
            print(
                f"\nUYARI: {provider.api_key_env} tanımlı değil; gerçek üretim "
                "bu ortam değişkeni olmadan başlamaz.",
                file=sys.stderr,
            )
        return 0

    if not provider.base_url:
        print(
            "Durduruldu: provider.base_url boş. Doğrulanmamış bir adrese istek "
            "gönderilmez. configs/teachers.yaml dosyasını doldurun.",
            file=sys.stderr,
        )
        return 3
    if not key_present:
        print(
            f"Durduruldu: {provider.api_key_env} ortam değişkeni yok. "
            "Anahtarı dosyaya yazmayın; ortam değişkeni veya Kaggle Secrets kullanın.",
            file=sys.stderr,
        )
        return 3

    pool = TeacherPool(teachers)
    if args.health_check:
        report = pool.health_check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 2

    started = time.time()
    try:
        stats = generate(
            teachers,
            seeds,
            out_path=args.out,
            per_seed=args.per_seed,
            resume=not args.no_resume,
            quality=quality,
            pool=pool,
            max_records=args.max_records,
        )
        exit_code = 0
    except GenerationAborted as exc:
        print(f"\nÜretim güvenli biçimde durduruldu: {exc}", file=sys.stderr)
        stats = exc.stats or GenerationStats()
        stats.aborted = True
        stats.abort_reason = str(exc)
        exit_code = 2

    summary = stats.to_dict()
    summary["duration_s"] = round(time.time() - started, 1)
    summary["out"] = args.out
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
