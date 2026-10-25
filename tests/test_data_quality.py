"""Schema validation, quality filters, de-duplication and split behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_record, write_jsonl
from gokturk.config import QualityConfig
from gokturk.data_quality import (
    assess_record,
    build_datasets,
    deduplicate,
    group_key,
    jaccard,
    normalize_for_dedup,
    read_jsonl,
    shingles,
    split_records,
    validate_record,
)
from gokturk.data_quality import (
    write_jsonl as write_records,
)


def test_valid_record_passes():
    assert validate_record(make_record()) == []
    assert assess_record(make_record()) == []


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda r: r.pop("id"), "missing required field"),
        (lambda r: r.__setitem__("prompt", "   "), "empty"),
        (lambda r: r.__setitem__("response", 5), "must be a string"),
        (lambda r: r.__setitem__("source", ""), "empty"),
        (lambda r: r.__setitem__("source", {"kind": "hacker"}), "source.kind"),
        (lambda r: r.__setitem__("source", {"api_key": "sk-x"}), "credentials"),
        (lambda r: r.__setitem__("meta", "nope"), "must be an object"),
        (lambda r: r.__setitem__("extra_field", 1), "unknown field"),
    ],
)
def test_schema_rejections(mutation, expected):
    rec = make_record()
    mutation(rec)
    problems = validate_record(rec)
    assert any(expected in p for p in problems), problems


def test_source_as_string_is_accepted():
    rec = make_record()
    rec["source"] = "human-written"
    assert validate_record(rec) == []


def test_quality_length_limits():
    cfg = QualityConfig(min_response_chars=50)
    rec = make_record(response="Kısa.")
    assert any("shorter" in p for p in assess_record(rec, cfg))

    cfg = QualityConfig(max_response_chars=20)
    rec = make_record(response="x " * 50)
    assert any("longer" in p for p in assess_record(rec, cfg))


def test_refusal_is_rejected():
    rec = make_record(response="Üzgünüm, bu isteği yerine getiremiyorum çünkü uygun değil.")
    assert any("refusal" in p for p in assess_record(rec))


def test_repetitive_response_is_rejected():
    line = "Bu satır sürekli tekrar ediyor ve anlamsız."
    rec = make_record(response="\n".join([line] * 8))
    assert any("repetitive" in p for p in assess_record(rec))


def test_unbalanced_code_fence_is_rejected():
    rec = make_record(response="İşte kod:\n```python\nprint(1)\n")
    assert any("code fence" in p for p in assess_record(rec))


def test_truncated_flag_is_rejected():
    rec = make_record(teacher="mimo")
    rec["source"]["truncated"] = True
    assert any("truncated" in p for p in assess_record(rec))


def test_normalisation_and_similarity():
    assert normalize_for_dedup("Merhaba,  Dünya!") == "merhaba dünya"
    a = shingles("bu bir deneme cümlesidir ve tekrar edecek")
    b = shingles("bu bir deneme cümlesidir ve tekrar edecek")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, shingles("tamamen farklı kelimeler burada")) < 0.2


def test_exact_and_near_duplicates_removed():
    base = dict(
        prompt="Ankara'nın tarihî önemi hakkında kısa bir paragraf yaz.",
        response="Ankara, Kurtuluş Savaşı'nın merkezi olmuş ve 1923'te başkent ilan edilmiştir.",
    )
    records = [
        make_record("a", **base),
        make_record("b", **base),  # exact duplicate
        make_record(
            "c",
            prompt=base["prompt"] + " Kısa olsun.",
            response=base["response"],  # near duplicate
        ),
        make_record(
            "d",
            prompt="Python'da dosya okuma nasıl yapılır?",
            response="`with open('yol') as f: f.read()` biçiminde okunur.",
        ),
    ]
    result = deduplicate(records)
    kept_ids = [r["id"] for r in result.kept]
    assert "a" in kept_ids
    assert "b" not in kept_ids
    assert "d" in kept_ids
    reasons = [reason for _, reason in result.dropped]
    assert any("exact duplicate" in r for r in reasons)


def test_duplicate_ids_removed():
    records = [make_record("x"), make_record("x", prompt="Bambaşka bir soru metni yaz buraya")]
    result = deduplicate(records)
    assert len(result.kept) == 1
    assert result.dropped[0][1] == "duplicate id"


def test_split_is_deterministic_and_group_aware():
    records = [
        make_record(f"id-{i}", prompt=f"Farklı konu {i} hakkında açıklama yaz.", seed_id=f"s{i%10}")
        for i in range(60)
    ]
    cfg = QualityConfig(valid_ratio=0.1, test_ratio=0.1, seed=7)
    first = split_records(records, cfg)
    second = split_records(records, cfg)
    assert [r["id"] for r in first["train"]] == [r["id"] for r in second["train"]]
    groups = {name: {group_key(r) for r in rows} for name, rows in first.items()}
    assert not (groups["train"] & groups["valid"])
    assert not (groups["train"] & groups["test"])


def test_group_key_prefers_seed_id():
    rec = make_record(seed_id="seed-42")
    assert group_key(rec) == "seed:seed-42"


def test_read_jsonl_reports_bad_lines(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(make_record("ok")) + "\n" + "{not json}\n" + "\n" + "[1,2,3]\n",
        encoding="utf-8",
    )
    records, issues = read_jsonl(path)
    assert len(records) == 1
    assert len(issues) == 2
    assert "invalid JSON" in issues[0].reason


def test_build_datasets_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    records = [
        make_record(
            f"id-{i}",
            prompt=f"Konu {i} için kısa ve özgün bir açıklama yaz lütfen.",
            response=f"Konu {i} hakkında yeterince uzun bir açıklama metni burada yer alıyor.",
            seed_id=f"s{i}",
        )
        for i in range(30)
    ]
    records.append(make_record("dup", **{k: records[0][k] for k in ("prompt", "response")}))
    records.append(make_record("empty", prompt="", response=""))
    write_jsonl(raw, records)

    report = build_datasets(raw, tmp_path / "out", QualityConfig(valid_ratio=0.1, test_ratio=0.1))
    assert report.kept == 30
    assert report.duplicates_removed >= 1
    assert report.quality_rejected >= 1
    for split in ("train", "valid", "test"):
        assert (tmp_path / "out" / f"{split}.jsonl").is_file()
    assert report.split_sizes["valid"] >= 1
    assert (tmp_path / "out" / "data_report.json").is_file()
    assert (tmp_path / "out" / "rejects.jsonl").is_file()

    # every written line must be valid JSON and schema-conformant
    written, issues = read_jsonl(tmp_path / "out" / "train.jsonl")
    assert not issues
    assert all(validate_record(r) == [] for r in written)


def test_build_datasets_fails_loudly_when_everything_is_filtered(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [make_record("bad", response="Kısa.")])
    with pytest.raises(ValueError, match="no valid records"):
        build_datasets(raw, tmp_path / "out")


def test_write_jsonl_is_atomic(tmp_path: Path):
    target = tmp_path / "out.jsonl"
    n = write_records(target, [make_record("1"), make_record("2")])
    assert n == 2
    assert target.is_file()
    assert not (tmp_path / "out.jsonl.tmp").exists()
    assert len(target.read_text().strip().splitlines()) == 2


def test_review_sample_is_deterministic_and_rendered(tmp_path: Path):
    from gokturk.data_quality import render_review_markdown, sample_for_review

    path = write_jsonl(
        tmp_path / "data.jsonl",
        [
            make_record(
                f"r{i}",
                prompt=f"{i}. konu hakkında kısa bir açıklama yazar mısın?",
                response=f"{i}. konu ile ilgili yeterince uzun bir açıklama metni burada.",
                seed_id=f"s{i}",
            )
            for i in range(20)
        ],
    )
    first = sample_for_review(path, n=5, seed=3)
    second = sample_for_review(path, n=5, seed=3)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 5
    md = render_review_markdown(first, str(path))
    assert "İnsan incelemesi için örnek" in md
    assert md.count("Değerlendirme:") == 5


def test_quality_cli_review_mode(tmp_path: Path, capsys):
    from gokturk.data_quality import main as quality_main

    path = write_jsonl(tmp_path / "data.jsonl", [make_record("a"), make_record("b", seed_id="s2")])
    out = tmp_path / "review.md"
    code = quality_main(["--data", str(path), "--review-sample", "1", "--review-out", str(out)])
    assert code == 0
    assert out.is_file()
    assert "İnsan incelemesi için örnek" in out.read_text(encoding="utf-8")


def test_quality_cli_split_mode(tmp_path: Path):
    from gokturk.data_quality import main as quality_main

    records = [
        make_record(
            f"c{i}",
            prompt=f"{i}. farklı konuda kısa bir açıklama yazar mısın?",
            response=f"{i}. farklı konuda yeterince uzun bir açıklama metni burada yer alıyor.",
            seed_id=f"s{i}",
        )
        for i in range(20)
    ]
    path = write_jsonl(tmp_path / "data.jsonl", records)
    out = tmp_path / "processed"
    assert quality_main(["--data", str(path), "--out-dir", str(out)]) == 0
    assert (out / "train.jsonl").is_file()
