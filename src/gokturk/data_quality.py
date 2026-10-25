"""Data validation, quality filtering, de-duplication and leakage-aware splits.

Two distinct concerns are kept apart on purpose:

1. **Format validity** -- is the JSONL parseable and schema-conformant?
2. **Content quality** -- is the example useful for training?

Content checks here are heuristic only. Passing this module is *not* evidence
that a label is correct: teacher outputs are automatically generated and must
still be sampled and reviewed by a human (see ``docs/architecture.md``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import QualityConfig, load_quality_config
from .schema import ALLOWED_FIELDS, REQUIRED_FIELDS, SCHEMA_VERSION, SOURCE_KINDS, TASK_TYPES

LOGGER = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
@dataclass
class JsonlIssue:
    line: int
    reason: str
    preview: str = ""


def read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], list[JsonlIssue]]:
    """Read JSONL, collecting parse errors instead of raising on the first one."""
    issues: list[JsonlIssue] = []
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    JsonlIssue(lineno, f"invalid JSON: {exc.msg}", stripped[:120])
                )
                continue
            if not isinstance(obj, dict):
                issues.append(JsonlIssue(lineno, "top-level value is not an object"))
                continue
            records.append(obj)
    return records, issues


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(p)  # atomic: a crash never leaves a half-written dataset
    return count


# ---------------------------------------------------------------------------
# format validation
# ---------------------------------------------------------------------------
def validate_record(rec: Mapping[str, Any], *, strict_fields: bool = True) -> list[str]:
    """Return schema problems. An empty list means the record is well formed."""
    problems: list[str] = []
    if not isinstance(rec, Mapping):
        return ["record is not an object"]

    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    if missing:
        problems.append(f"missing required field(s): {missing}")
        return problems  # further checks would be noise

    if strict_fields:
        unknown = sorted(set(rec) - set(ALLOWED_FIELDS))
        if unknown:
            problems.append(f"unknown field(s): {unknown}")

    rec_id = rec.get("id")
    if not isinstance(rec_id, str) or not rec_id.strip():
        problems.append("'id' must be a non-empty string")

    for key in ("prompt", "response"):
        value = rec.get(key)
        if not isinstance(value, str):
            problems.append(f"'{key}' must be a string")
        elif not value.strip():
            problems.append(f"'{key}' is empty")

    source = rec.get("source")
    if isinstance(source, str):
        if not source.strip():
            problems.append("'source' is empty")
    elif isinstance(source, Mapping):
        kind = source.get("kind")
        if kind is not None and kind not in SOURCE_KINDS:
            problems.append(f"source.kind={kind!r} not in {list(SOURCE_KINDS)}")
        if not any(str(source.get(k, "")).strip() for k in ("kind", "teacher", "name")):
            problems.append("source must identify at least kind/teacher/name")
        task_type = source.get("task_type")
        if task_type is not None and task_type not in TASK_TYPES:
            problems.append(f"source.task_type={task_type!r} not in {list(TASK_TYPES)}")
        for secret_key in ("api_key", "authorization", "token", "password"):
            if secret_key in source:
                problems.append(f"source must not contain credentials ({secret_key})")
    else:
        problems.append("'source' must be a string or an object")

    meta = rec.get("meta")
    if meta is not None and not isinstance(meta, Mapping):
        problems.append("'meta' must be an object when present")

    messages = rec.get("messages")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            problems.append("'messages' must be a non-empty list when present")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, Mapping) or "role" not in msg or "content" not in msg:
                    problems.append(f"messages[{i}] must have 'role' and 'content'")
                    break

    return problems


# ---------------------------------------------------------------------------
# content quality
# ---------------------------------------------------------------------------
def normalize_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace and strip punctuation-ish noise."""
    lowered = text.casefold()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def shingles(text: str, n: int = 3) -> frozenset[str]:
    words = _WORD_RE.findall(normalize_for_dedup(text))
    if len(words) < n:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    counts = Counter(lines)
    return max(counts.values()) / len(lines)


def assess_record(
    rec: Mapping[str, Any],
    config: QualityConfig | None = None,
) -> list[str]:
    """Heuristic content problems. Empty list means "keep".

    This deliberately does *not* judge factual correctness. It only removes
    examples that cannot be good training targets (empty, truncated, refusing,
    looping, or wildly out of the configured length budget).
    """
    cfg = config or QualityConfig()
    problems: list[str] = list(validate_record(rec))
    if problems:
        return problems

    prompt = str(rec["prompt"])
    response = str(rec["response"])

    if len(prompt) < cfg.min_prompt_chars:
        problems.append(f"prompt shorter than {cfg.min_prompt_chars} chars")
    if len(prompt) > cfg.max_prompt_chars:
        problems.append(f"prompt longer than {cfg.max_prompt_chars} chars")
    if len(response) < cfg.min_response_chars:
        problems.append(f"response shorter than {cfg.min_response_chars} chars")
    if len(response) > cfg.max_response_chars:
        problems.append(f"response longer than {cfg.max_response_chars} chars")
    if len(_WORD_RE.findall(response)) < cfg.min_response_words:
        problems.append(f"response has fewer than {cfg.min_response_words} words")

    if normalize_for_dedup(prompt) == normalize_for_dedup(response):
        problems.append("response is identical to prompt")

    lowered = response.casefold()
    for marker in cfg.refusal_markers:
        if marker.casefold() in lowered:
            problems.append(f"response looks like a refusal ({marker!r})")
            break

    if _repeated_line_ratio(response) > cfg.max_repeated_line_ratio:
        problems.append("response is repetitive (same line repeated)")

    if len(_CODE_FENCE_RE.findall(response)) % 2 != 0:
        problems.append("unbalanced markdown code fence (likely truncated)")

    if any(ord(ch) < 9 for ch in response):
        problems.append("response contains control characters")

    source = rec.get("source")
    if isinstance(source, Mapping) and source.get("truncated") is True:
        problems.append("teacher marked the response as truncated")

    return problems


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------
@dataclass
class DedupResult:
    kept: list[dict[str, Any]]
    dropped: list[tuple[dict[str, Any], str]]


def deduplicate(
    records: Sequence[dict[str, Any]],
    config: QualityConfig | None = None,
) -> DedupResult:
    """Remove exact and near duplicates.

    * Exact duplicates: identical id, or identical normalised (prompt, response).
    * Near duplicates: word-shingle Jaccard >= ``near_duplicate_threshold``.
      Blocked by a cheap word-overlap bucket so this stays near-linear for the
      dataset sizes this project targets (10^4--10^5 rows).
    """
    cfg = config or QualityConfig()
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    # bucket -> list of (shingle_set, index_in_kept)
    buckets: dict[str, list[int]] = {}
    kept_shingles: list[frozenset[str]] = []

    for rec in records:
        rec_id = str(rec.get("id", ""))
        if rec_id in seen_ids:
            dropped.append((rec, "duplicate id"))
            continue
        pair = (
            normalize_for_dedup(str(rec.get("prompt", ""))),
            normalize_for_dedup(str(rec.get("response", ""))),
        )
        if pair in seen_pairs:
            dropped.append((rec, "exact duplicate (prompt+response)"))
            continue

        sh = shingles(pair[0] + " " + pair[1])
        words = _WORD_RE.findall(pair[0] + " " + pair[1])
        if not words:
            dropped.append((rec, "empty after normalisation"))
            continue
        bucket_keys = {
            hashlib.blake2b(w.encode("utf-8"), digest_size=6).hexdigest()
            for w in set(words[:24])
        }
        is_near = False
        candidate_hits: Counter[int] = Counter()
        for bk in bucket_keys:
            for idx in buckets.get(bk, ()):
                candidate_hits[idx] += 1
        # only compare against the strongest candidates
        for idx, _ in candidate_hits.most_common(8):
            if jaccard(sh, kept_shingles[idx]) >= cfg.near_duplicate_threshold:
                is_near = True
                break
        if is_near:
            dropped.append((rec, "near duplicate"))
            continue

        seen_ids.add(rec_id)
        seen_pairs.add(pair)
        idx = len(kept)
        kept.append(rec)
        kept_shingles.append(sh)
        for bk in bucket_keys:
            buckets.setdefault(bk, []).append(idx)

    return DedupResult(kept=kept, dropped=dropped)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def group_key(rec: Mapping[str, Any]) -> str:
    """Leakage-aware grouping key.

    Examples that share a seed task (``source.seed_id``) or that start with the
    same prompt prefix land in the same split, so paraphrases of one prompt can
    never be split across train and test.
    """
    source = rec.get("source")
    seed_id = None
    if isinstance(source, Mapping):
        seed_id = source.get("seed_id")
    if isinstance(seed_id, str) and seed_id.strip():
        return f"seed:{seed_id}"
    words = _WORD_RE.findall(normalize_for_dedup(str(rec.get("prompt", ""))))[:6]
    return "prompt:" + " ".join(words)


def split_records(
    records: Sequence[dict[str, Any]],
    config: QualityConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic grouped split into train/valid/test."""
    cfg = config or QualityConfig()
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(group_key(rec), []).append(rec)

    keys = sorted(groups)
    rng = random.Random(cfg.seed)
    rng.shuffle(keys)

    total = len(records)
    n_valid_target = max(1, int(round(total * cfg.valid_ratio))) if total >= 2 else 0
    n_test_target = max(1, int(round(total * cfg.test_ratio))) if total >= 2 else 0

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for key in keys:
        bucket = groups[key]
        if len(splits["valid"]) < n_valid_target:
            splits["valid"].extend(bucket)
        elif len(splits["test"]) < n_test_target:
            splits["test"].extend(bucket)
        else:
            splits["train"].extend(bucket)

    # Guarantee a non-empty train split when possible.
    if not splits["train"] and splits["valid"]:
        moved = splits["valid"].pop(0)
        splits["train"].append(moved)
    return splits


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
@dataclass
class DatasetReport:
    input_records: int = 0
    parse_errors: int = 0
    invalid_records: int = 0
    quality_rejected: int = 0
    duplicates_removed: int = 0
    kept: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    teachers: dict[str, int] = field(default_factory=dict)
    task_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_records": self.input_records,
            "parse_errors": self.parse_errors,
            "invalid_records": self.invalid_records,
            "quality_rejected": self.quality_rejected,
            "duplicates_removed": self.duplicates_removed,
            "kept": self.kept,
            "split_sizes": dict(self.split_sizes),
            "reject_reasons": dict(self.reject_reasons),
            "teachers": dict(self.teachers),
            "task_types": dict(self.task_types),
        }

    def summary(self) -> str:
        parts = [
            f"okundu={self.input_records}",
            f"geçersiz={self.invalid_records}",
            f"kalite_red={self.quality_rejected}",
            f"tekrar={self.duplicates_removed}",
            f"kalan={self.kept}",
        ]
        if self.split_sizes:
            parts.append(
                "bölünme="
                + "/".join(f"{k}:{v}" for k, v in self.split_sizes.items())
            )
        return ", ".join(parts)


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def build_datasets(
    input_path: str | Path,
    out_dir: str | Path,
    config: QualityConfig | None = None,
    *,
    allow_empty: bool = False,
) -> DatasetReport:
    """Validate -> filter -> de-duplicate -> split -> write JSONL files."""
    cfg = config or QualityConfig()
    cfg.validate()
    records, issues = read_jsonl(input_path)
    report = DatasetReport(input_records=len(records) + len(issues))
    report.parse_errors = len(issues)
    for issue in issues:
        LOGGER.warning("line %d: %s | %s", issue.line, issue.reason, issue.preview)
        report.reject_reasons[issue.reason.split(":")[0]] = (
            report.reject_reasons.get(issue.reason.split(":")[0], 0) + 1
        )

    valid: list[dict[str, Any]] = []
    for rec in records:
        problems = assess_record(rec, cfg)
        if problems:
            report.quality_rejected += 1
            key = problems[0].split("(")[0].strip()
            report.reject_reasons[key] = report.reject_reasons.get(key, 0) + 1
            continue
        rec.setdefault("schema_version", SCHEMA_VERSION)
        valid.append(rec)

    dedup = deduplicate(valid, cfg)
    report.duplicates_removed = len(dedup.dropped)
    for _rec, reason in dedup.dropped:
        report.reject_reasons[reason] = report.reject_reasons.get(reason, 0) + 1

    report.kept = len(dedup.kept)
    if not report.kept and not allow_empty:
        raise ValueError(
            f"no valid records left after filtering {input_path}; "
            "check the reject reasons in the report"
        )

    splits = split_records(dedup.kept, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        report.split_sizes[name] = len(rows)

    report.teachers = _count(
        str(rec.get("source", {}).get("teacher", "unknown"))
        if isinstance(rec.get("source"), Mapping)
        else "unknown"
        for rec in dedup.kept
    )
    report.task_types = _count(
        str(rec.get("source", {}).get("task_type", "unknown"))
        if isinstance(rec.get("source"), Mapping)
        else "unknown"
        for rec in dedup.kept
    )

    (out / "data_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "rejects.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": r.get("id"), "reason": reason, "prompt": str(r.get("prompt", ""))[:200]},
                ensure_ascii=False,
            )
            + "\n"
            for r, reason in dedup.dropped
        ),
        encoding="utf-8",
    )
    return report


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    records, _ = read_jsonl(path)
    yield from records


# ---------------------------------------------------------------------------
# human review sampling
# ---------------------------------------------------------------------------
def sample_for_review(
    path: str | Path,
    n: int = 30,
    seed: int = 0,
) -> list[dict]:
    """Draw a deterministic random sample for human inspection.

    Teacher output is never assumed correct; this is the entry point of the
    review process described in ``docs/architecture.md``.
    """
    records, _ = read_jsonl(path)
    if not records:
        return []
    rng = random.Random(seed)
    k = min(n, len(records))
    return rng.sample(records, k)


def render_review_markdown(records: Sequence[Mapping[str, Any]], source_path: str) -> str:
    lines = [
        "# İnsan incelemesi için örnek",
        "",
        f"- Kaynak: `{source_path}`",
        f"- Örnek sayısı: {len(records)}",
        "",
        "Her kayıt için üç şeyi işaretleyin: **(D)** doğru, **(K)** kısmen doğru/eksik,",
        "**(Y)** yanlış. Hata oranı kabul eşiğini aşarsa ilgili `template_version`",
        "ile üretilen veri atılmalıdır (bkz. docs/architecture.md §4).",
        "",
    ]
    for i, rec in enumerate(records, start=1):
        source = rec.get("source") if isinstance(rec.get("source"), Mapping) else {}
        lines += [
            f"## {i}. `{rec.get('id')}` — {source.get('teacher', '?')} / {source.get('task_type', '?')}",
            "",
            f"**İstem:** {str(rec.get('prompt', ''))[:600]}",
            "",
            f"**Yanıt:** {str(rec.get('response', ''))[:1200]}",
            "",
            "Değerlendirme: ( ) D  ( ) K  ( ) Y   — Not:",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: kalite raporu üret veya insan incelemesi için örnek çıkar."""
    import argparse

    parser = argparse.ArgumentParser(description="Veri kalitesi araçları")
    parser.add_argument("--data", required=True, help="JSONL dosyası")
    parser.add_argument("--out-dir", default=None, help="Bölme çıktıları için dizin")
    parser.add_argument("--quality-config", default=None)
    parser.add_argument("--review-sample", type=int, default=0, help="İnceleme için örnek sayısı")
    parser.add_argument("--review-out", default=None, help="İnceleme Markdown yolu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_quality_config(args.quality_config)
    if args.review_sample:
        records = sample_for_review(args.data, args.review_sample, args.seed)
        markdown = render_review_markdown(records, args.data)
        if args.review_out:
            Path(args.review_out).write_text(markdown, encoding="utf-8")
            print(f"İnceleme dosyası yazıldı: {args.review_out} ({len(records)} örnek)")
        else:
            print(markdown)
        return 0

    if args.out_dir:
        report = build_datasets(args.data, args.out_dir, config)
        print(report.summary())
        return 0

    print(json.dumps(format_report(args.data, config), ensure_ascii=False, indent=2)[:8000])
    return 0


def format_report(path: str | Path, config: QualityConfig | None = None) -> dict:
    """Schema + quality summary for a JSONL file (no splitting)."""
    cfg = config or QualityConfig()
    records, issues = read_jsonl(path)
    problems = {str(r.get("id", "<no-id>")): assess_record(r, cfg) for r in records}
    problems = {k: v for k, v in problems.items() if v}
    return {
        "path": str(path),
        "records": len(records),
        "parse_errors": [asdict_issue(i) for i in issues],
        "records_with_problems": len(problems),
        "problems_sample": dict(list(problems.items())[:50]),
    }


def asdict_issue(issue: JsonlIssue) -> dict:
    return {"line": issue.line, "reason": issue.reason, "preview": issue.preview}


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    _sys.exit(main())
