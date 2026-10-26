"""Turn the CoT dataset into train/valid/test files this repo can train on.

What the input looks like
-------------------------
``data/datasets/evren_cot_10k.jsonl`` (the file that was added to the repo) has
records shaped like::

    {"id": "evren_cot_10k_00001",
     "domain": "system_architecture",
     "instruction": "Kubernetes'te pod security nasıl yapılandırılır? (id: 4019)",
     "thinking_process": "1. Problem anlama: ...",
     "response": "K8s'te pod security için RBAC ...",
     "quality_metrics": {"relevance_score": 1.0, ...}}

This repo's training path (:mod:`gokturk.train`) and quality gate
(:mod:`gokturk.data_quality`) speak a different, simpler schema::

    {"prompt": "...", "response": "..."}

so the file cannot be fed to ``train.py`` as-is. This module does the mapping
and, just as importantly, **measures the dataset before anyone trains on it**.

Why measuring matters (2026-10-25)
----------------------------------
The file has 10,000 rows. After removing the ``(id: N)`` artifacts that make
look-alike instructions unique, it has ~920 distinct instructions and ~990
distinct answers: roughly ten copies of each. Training on all 10k rows for a
"10k dataset" is really training ~920 examples for ten epochs — it burns GPU
time and teaches memorisation. The report this module returns states that
number out loud instead of letting the row count imply diversity.

Usage
-----
    python -m gokturk.cot_dataset --input data/datasets/evren_cot_10k.jsonl \\
        --out-dir data/processed

    python -m gokturk.cot_dataset --input evren_cot_10k.jsonl --no-thinking \\
        --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_quality import build_datasets, normalize_for_dedup

#: Trailing artifacts produced by the generator, e.g. ``(id: 4019)`` or
#: ``(senaryo id: 10880)``. They leak internal counters into the prompt, so the
#: model would learn to imitate them.
_ID_ARTIFACT = re.compile(r"\s*\(\s*(?:senaryo\s+id|scenario\s+id|id)\s*:\s*\d+\s*\)\s*$", re.I)

#: Below this share of distinct instructions the dataset is template-generated
#: and row counts are misleading.
LOW_DIVERSITY_RATIO = 0.30

REQUIRED_FIELDS = ("instruction", "response")

#: The CoT file's ``domain`` values are not all in :data:`gokturk.schema.TASK_TYPES`.
#: Unmapped domains are left without a ``task_type`` rather than being forced into
#: a wrong bucket: the quality report then shows the true domain spread.
DOMAIN_TO_TASK_TYPE: dict[str, str] = {
    "system_architecture": "system_architecture",
    "logic_reasoning": "reasoning",
    "algorithms": "coding",
    "cybersecurity": "system_architecture",  # en yakın aile; güvenlik ayrı aile değil
}

#: Provenance written into every converted record (``source`` is required by the
#: schema). No credentials, no provider payloads.
DEFAULT_SOURCE: dict[str, str] = {
    "kind": "synthetic",
    "name": "evren_cot_10k",
    "template_version": "cot-v1",
}


class CotDatasetError(RuntimeError):
    """Raised when the CoT file cannot be read or lacks required fields."""


@dataclass
class CotReport:
    source: str
    input_rows: int = 0
    parse_errors: int = 0
    missing_fields: int = 0
    id_artifacts_stripped: int = 0
    unique_instructions_raw: int = 0
    unique_instructions_clean: int = 0
    unique_responses: int = 0
    unique_thinking: int = 0
    conflicting_answers: int = 0
    kept_after_dedupe: int = 0
    domains: dict[str, int] = field(default_factory=dict)
    include_thinking: bool = True
    dedupe: bool = True
    written: dict[str, int] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_uniqueness(self) -> float:
        if not self.input_rows:
            return 0.0
        return round(self.unique_instructions_clean / self.input_rows, 3)

    @property
    def low_diversity(self) -> bool:
        return self.effective_uniqueness < LOW_DIVERSITY_RATIO

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "input_rows": self.input_rows,
            "parse_errors": self.parse_errors,
            "missing_fields": self.missing_fields,
            "id_artifacts_stripped": self.id_artifacts_stripped,
            "unique_instructions_raw": self.unique_instructions_raw,
            "unique_instructions_clean": self.unique_instructions_clean,
            "unique_responses": self.unique_responses,
            "unique_thinking": self.unique_thinking,
            "conflicting_answers": self.conflicting_answers,
            "kept_after_dedupe": self.kept_after_dedupe,
            "effective_uniqueness": self.effective_uniqueness,
            "low_diversity": self.low_diversity,
            "domains": self.domains,
            "include_thinking": self.include_thinking,
            "dedupe": self.dedupe,
            "written": self.written,
            "quality": self.quality,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [
            f"Kaynak: {self.source}",
            f"Satır: {self.input_rows} (ayrıştırma hatası {self.parse_errors}, "
            f"eksik alan {self.missing_fields})",
            f"Temizlenen '(id: N)' kalıntısı: {self.id_artifacts_stripped}",
            f"TEKİL talimat: {self.unique_instructions_clean} "
            f"(ham: {self.unique_instructions_raw}) -> etkin çeşitlilik "
            f"%{self.effective_uniqueness * 100:.1f}",
            f"TEKİL yanıt: {self.unique_responses} | tekil düşünme zinciri: {self.unique_thinking}",
            f"Aynı talimata çelişkili yanıt: {self.conflicting_answers}",
            f"Tekilleştirme sonrası kalan: {self.kept_after_dedupe}",
        ]
        if self.domains:
            lines.append("Alan dağılımı: " + ", ".join(f"{k}={v}" for k, v in sorted(self.domains.items())))
        if self.written:
            lines.append("Yazılan: " + ", ".join(f"{k}={v}" for k, v in sorted(self.written.items())))
        for warning in self.warnings:
            lines.append(f"UYARI: {warning}")
        return "\n".join(lines)


def strip_id_artifacts(text: str) -> str:
    """Remove a trailing ``(id: N)`` / ``(senaryo id: N)`` marker."""
    return _ID_ARTIFACT.sub("", text).strip()


def _iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any] | None]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield number, None
                continue
            yield number, row if isinstance(row, dict) else None


def build_example(
    row: dict[str, Any],
    *,
    include_thinking: bool = True,
    index: int = 0,
) -> dict[str, Any] | None:
    """Map one CoT row onto this repo's schema (``None`` if unusable).

    The output carries ``id``, ``prompt``, ``response`` and ``source`` because
    :func:`gokturk.data_quality.validate_record` requires all four; omitting
    ``id``/``source`` is what made every record fail the quality gate on the
    first attempt.
    """
    instruction = strip_id_artifacts(str(row.get("instruction") or ""))
    answer = str(row.get("response") or "").strip()
    thinking = str(row.get("thinking_process") or "").strip()
    if not instruction or not answer:
        return None

    if include_thinking and thinking:
        target = f"{thinking}\n\n{answer}"
    else:
        target = answer
    record_id = str(row.get("id") or f"cot-{index:05d}").strip()
    domain = str(row.get("domain") or "").strip()

    source: dict[str, Any] = dict(DEFAULT_SOURCE)
    task_type = DOMAIN_TO_TASK_TYPE.get(domain)
    if task_type:
        source["task_type"] = task_type

    meta: dict[str, Any] = {"lang": "tr"}
    if domain:
        meta["domain"] = domain
    meta["has_thinking"] = bool(include_thinking and thinking)

    return {"id": record_id, "prompt": instruction, "response": target, "source": source, "meta": meta}


def convert_cot(
    source: str | Path,
    *,
    include_thinking: bool = True,
) -> tuple[list[dict[str, Any]], CotReport]:
    """Convert the CoT file into this repo's schema, with diversity statistics."""
    path = Path(source)
    if not path.is_file():
        raise CotDatasetError(
            f"CoT veri dosyası yok: {path}. Depoda `data/datasets/evren_cot_10k.jsonl` var; "
            "yolu ona verin ya da kendi dosyanızı gösterin."
        )
    report = CotReport(source=str(path), include_thinking=include_thinking)

    raw_instructions: list[str] = []
    clean_instructions: list[str] = []
    answers: list[str] = []
    thinkings: list[str] = []
    domains: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    answers_by_instruction: dict[str, set[str]] = {}

    for _number, row in _iter_rows(path):
        if row is None:
            report.parse_errors += 1
            continue
        report.input_rows += 1
        missing = [key for key in REQUIRED_FIELDS if not str(row.get(key) or "").strip()]
        if missing:
            report.missing_fields += 1
            continue

        raw_instruction = str(row["instruction"]).strip()
        clean = strip_id_artifacts(raw_instruction)
        if clean != raw_instruction:
            report.id_artifacts_stripped += 1
        raw_instructions.append(raw_instruction)
        clean_instructions.append(clean)
        answers.append(str(row["response"]).strip())
        thinkings.append(str(row.get("thinking_process") or "").strip())
        if row.get("domain"):
            domains[str(row["domain"])] += 1
        answers_by_instruction.setdefault(normalize_for_dedup(clean), set()).add(
            normalize_for_dedup(str(row["response"]))
        )

        example = build_example(
            row, include_thinking=include_thinking, index=report.input_rows
        )
        if example is not None:
            examples.append(example)

    report.unique_instructions_raw = len(set(raw_instructions))
    report.unique_instructions_clean = len(set(clean_instructions))
    report.unique_responses = len(set(answers))
    report.unique_thinking = len({t for t in thinkings if t})
    report.conflicting_answers = sum(1 for v in answers_by_instruction.values() if len(v) > 1)
    report.domains = dict(domains)
    report.kept_after_dedupe = len(examples)
    return examples, report


def dedupe_examples(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first example per (prompt, response) pair, order preserved.

    Duplicates with the same prompt but a *different* answer are ambiguous and
    are reported by :func:`convert_cot`; here the first one wins so the result
    is deterministic for a given file order.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for example in examples:
        key = (normalize_for_dedup(example["prompt"]), normalize_for_dedup(example["response"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(example)
    # ``id`` alanı dosyada tekil olmalı; tekilleştirme sonrası çakışma kalmasın.
    for position, example in enumerate(kept, start=1):
        if not example.get("id"):
            example["id"] = f"cot-{position:05d}"
    return kept


def prepare_cot_dataset(
    source: str | Path,
    out_dir: str | Path,
    *,
    include_thinking: bool = True,
    dedupe: bool = True,
    min_kept: int = 400,
) -> CotReport:
    """Convert, de-duplicate, quality-gate and split the CoT file.

    Writes ``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl`` plus
    ``cot_report.json`` into ``out_dir``. The report doubles as a warning: with
    ~920 distinct instructions the honest epoch budget is small and the model
    will not learn general Turkish instruction following from this file alone.
    """
    examples, report = convert_cot(source, include_thinking=include_thinking)
    report.dedupe = dedupe
    if dedupe:
        examples = dedupe_examples(examples)
    report.kept_after_dedupe = len(examples)

    if len(examples) < min_kept:
        raise CotDatasetError(
            f"Tekilleştirmeden sonra yalnızca {len(examples)} örnek kaldı "
            f"(eşik {min_kept}). Bu dosya eğitim için çok küçük; üretimi artırın."
        )

    if report.id_artifacts_stripped:
        report.warnings.append(
            f"{report.id_artifacts_stripped} kayıtta '(id: N)' kalıntısı vardı; temizlendi. "
            "Bunlar temizlenmeden eğitilseydi model sayaç taklidi öğrenirdi."
        )
    if report.low_diversity:
        report.warnings.append(
            f"ETKİN ÇEŞİTLİLİK DÜŞÜK: {report.input_rows} satırda yalnızca "
            f"{report.unique_instructions_clean} tekil talimat var "
            f"(%{report.effective_uniqueness * 100:.1f}). Bu dosya şablon üretimi; "
            "hepsiyle eğitim yapmak ~10 epoch aynı örnekleri tekrarlamak demektir. "
            "Tekilleştirilmiş hâliyle eğitin ve çeşitliliği EVREN üretimiyle artırın."
        )
    if report.conflicting_answers:
        report.warnings.append(
            f"{report.conflicting_answers} talimatın birden fazla farklı yanıtı var; "
            "ilk kayıt tutuldu. Talimat başına tek yanıt istiyorsanız üretimi düzeltin."
        )

    out = Path(out_dir)
    staging = out / "_cot_converted.jsonl"
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    try:
        quality = build_datasets(staging, out, allow_empty=False)
        report.quality = {
            "input_records": quality.input_records,
            "quality_rejected": quality.quality_rejected,
            "duplicates_removed": quality.duplicates_removed,
            "kept": quality.kept,
            "split": dict(quality.split_sizes) if hasattr(quality, "split_sizes") else {},
            "summary": quality.summary(),
        }
        report.written = {
            "train": _count_rows(out / "train.jsonl"),
            "valid": _count_rows(out / "valid.jsonl"),
            "test": _count_rows(out / "test.jsonl"),
        }
    finally:
        staging.unlink(missing_ok=True)

    (out / "cot_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def training_steps_for(examples: int, *, epochs: float = 2.0, batch: int = 2, grad_accum: int = 8) -> int:
    """Adım sayısı = örnek × epoch / (batch × grad_accum), 60–1500 arasına sıkıştırılmış."""
    if examples <= 0:
        return 60
    steps = round(examples * epochs / max(batch * grad_accum, 1))
    return max(60, min(1500, steps))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CoT JSONL'ini bu reponun prompt/response şemasına çevirip böl"
    )
    parser.add_argument(
        "--input",
        default="data/datasets/evren_cot_10k.jsonl",
        help="CoT dosyası (instruction/thinking_process/response)",
    )
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Yalnızca nihai yanıtı hedefle (düşünme zincirini at)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Tekilleştirme yapma (satır sayısı korunur, ~10x tekrar eğitilir)",
    )
    parser.add_argument(
        "--min-kept",
        type=int,
        default=400,
        help="Tekilleştirme sonrası kabul edilen en az örnek (varsayılan 400)",
    )
    parser.add_argument("--json", action="store_true", help="Raporu JSON olarak yaz")
    args = parser.parse_args(argv)

    try:
        report = prepare_cot_dataset(
            args.input,
            args.out_dir,
            include_thinking=not args.no_thinking,
            dedupe=not args.keep_duplicates,
            min_kept=args.min_kept,
        )
    except CotDatasetError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 4

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        print("\nYazıldı:", Path(args.out_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
