"""Tests for :mod:`gokturk.cot_dataset` — the CoT → repo-schema bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gokturk.cot_dataset import (
    DEFAULT_SOURCE,
    LOW_DIVERSITY_RATIO,
    CotDatasetError,
    build_example,
    convert_cot,
    dedupe_examples,
    prepare_cot_dataset,
    strip_id_artifacts,
    training_steps_for,
)
from gokturk.data_quality import validate_record

# Quality gate needs >=12 chars of prompt, >=20 chars / >=4 words of response;
# deliberately realistic so the fixtures exercise the real gate, not a stub.
PROMPT = "Kurumsal ağda DNS tünellemesi hangi filtrelerle tespit edilir?"
ANSWER = "Önce DNS sorgu hacmi baseline ile karşılaştırılır, sonra uzun TXT kayıtları incelenir."
THINKING = "Problemi ağ katmanına indir, ardından istatistiksel ve kural tabanlı yöntemleri sırala."


def _row(
    instruction: str = PROMPT,
    response: str = ANSWER,
    *,
    domain: str = "cybersecurity",
    thinking: str = THINKING,
    record_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "domain": domain,
        "instruction": instruction,
        "thinking_process": thinking,
        "response": response,
        "quality_metrics": {"relevance": 1.0, "coherence": 1.0, "pii_clean": True},
    }
    if record_id:
        row["id"] = record_id
    return row


def _write(path: Path, rows: list[object]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


# --------------------------------------------------------------------------- #
# strip_id_artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"{PROMPT} (id: 123)", PROMPT),
        ("Senaryoyu çöz (senaryo id: 7)", "Senaryoyu çöz"),
        ("Sonda kalıntı yok", "Sonda kalıntı yok"),
        ("  boşluklu metin (id: 4)  ", "boşluklu metin"),
    ],
)
def test_strip_id_artifacts(raw: str, expected: str) -> None:
    assert strip_id_artifacts(raw) == expected


# --------------------------------------------------------------------------- #
# build_example
# --------------------------------------------------------------------------- #


def test_build_example_emits_every_required_field() -> None:
    example = build_example(_row(f"{PROMPT} (id: 9)"), index=1)

    assert example is not None
    # The quality gate rejects records without id/prompt/response/source; this is
    # the exact regression that made the first CLI run drop all 10k rows.
    assert validate_record(example) == []
    assert example["id"] == "cot-00001"
    assert example["prompt"] == PROMPT
    assert example["response"] == f"{THINKING}\n\n{ANSWER}"
    assert example["source"]["name"] == DEFAULT_SOURCE["name"]
    assert example["source"]["task_type"] == "system_architecture"
    assert example["meta"] == {"lang": "tr", "domain": "cybersecurity", "has_thinking": True}


def test_build_example_keeps_file_id_and_maps_domain() -> None:
    example = build_example(_row(domain="logic_reasoning", record_id="evren_1"))

    assert example is not None
    assert example["id"] == "evren_1"
    assert example["source"]["task_type"] == "reasoning"


def test_build_example_without_thinking_targets_answer_only() -> None:
    example = build_example(_row(), include_thinking=False)

    assert example is not None
    assert example["response"] == ANSWER
    assert example["meta"]["has_thinking"] is False


def test_build_example_unknown_domain_leaves_task_type_unset() -> None:
    example = build_example(_row(domain="astronomi"))

    assert example is not None
    assert "task_type" not in example["source"]
    assert example["meta"]["domain"] == "astronomi"


@pytest.mark.parametrize(
    "row",
    [
        {"response": ANSWER},
        {"instruction": PROMPT},
        {"instruction": "   ", "response": ANSWER},
    ],
)
def test_build_example_returns_none_for_unusable_rows(row: dict[str, object]) -> None:
    assert build_example(row) is None


# --------------------------------------------------------------------------- #
# dedupe_examples
# --------------------------------------------------------------------------- #


def test_dedupe_examples_ignores_whitespace_only_differences() -> None:
    examples = [
        {"id": "a", "prompt": PROMPT, "response": ANSWER},
        {"id": "b", "prompt": f"{PROMPT} ", "response": f"{ANSWER}\n"},
        {"id": "c", "prompt": PROMPT, "response": f"{ANSWER} Ardından rapor yazılır."},
    ]

    kept = dedupe_examples(examples)

    assert [example["id"] for example in kept] == ["a", "c"]


def test_dedupe_examples_keeps_first_occurrence_order() -> None:
    examples = [
        {"id": f"id-{i}", "prompt": f"{PROMPT} Varyant {i}", "response": ANSWER}
        for i in range(5)
    ]

    kept = dedupe_examples(list(reversed(examples)))

    assert [example["id"] for example in kept] == [f"id-{i}" for i in reversed(range(5))]


# --------------------------------------------------------------------------- #
# convert_cot
# --------------------------------------------------------------------------- #


def test_convert_cot_counts_errors_missing_and_conflicts(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "cot.jsonl",
        [
            _row(f"{PROMPT} A (id: 1)", f"{ANSWER} A"),
            _row(f"{PROMPT} A (id: 2)", f"{ANSWER} B"),  # aynı talimat, farklı yanıt
            _row(f"{PROMPT} B (id: 3)", f"{ANSWER} A"),
            {"instruction": "", "response": ANSWER},  # eksik alan
            "{bozuk json",  # ayrıştırma hatası
            "",
        ],
    )

    examples, report = convert_cot(path)

    assert report.input_rows == 4
    assert report.parse_errors == 1
    assert report.missing_fields == 1
    assert report.id_artifacts_stripped == 3
    assert report.unique_instructions_clean == 2
    assert report.conflicting_answers == 1
    assert len(examples) == 3
    assert report.domains == {"cybersecurity": 3}


def test_convert_cot_flags_low_diversity(tmp_path: Path) -> None:
    rows = [_row(f"{PROMPT} (id: {i})") for i in range(20)]
    examples, report = convert_cot(_write(tmp_path / "cot.jsonl", rows))

    assert len(examples) == 20
    assert report.unique_instructions_clean == 1
    assert report.effective_uniqueness == pytest.approx(1 / 20, abs=1e-3)
    assert report.effective_uniqueness < LOW_DIVERSITY_RATIO
    assert report.low_diversity is True


def test_convert_cot_missing_file_raises() -> None:
    with pytest.raises(CotDatasetError, match="CoT veri dosyası yok"):
        convert_cot("/nonexistent/cot.jsonl")


# --------------------------------------------------------------------------- #
# prepare_cot_dataset
# --------------------------------------------------------------------------- #


def test_prepare_cot_dataset_writes_split_and_report(tmp_path: Path) -> None:
    rows = [_row(f"{PROMPT} Varyant {i} (id: {i})", f"{ANSWER} Varyant {i}") for i in range(80)]
    source = _write(tmp_path / "cot.jsonl", rows)
    out_dir = tmp_path / "processed"

    report = prepare_cot_dataset(source, out_dir, min_kept=10)

    for name in ("train", "valid", "test"):
        assert (out_dir / f"{name}.jsonl").is_file()
    written = json.loads((out_dir / "cot_report.json").read_text(encoding="utf-8"))
    assert written["kept_after_dedupe"] == 80
    assert sum(written["written"].values()) > 0
    assert report.quality["quality_rejected"] == 0
    assert report.written["train"] > 0
    assert not (out_dir / "_cot_converted.jsonl").exists(), "geçici dosya silinmeli"


def test_prepare_cot_dataset_dedupes_by_default(tmp_path: Path) -> None:
    rows = [_row(f"{PROMPT} (id: {i})") for i in range(30)]
    source = _write(tmp_path / "cot.jsonl", rows)

    report = prepare_cot_dataset(source, tmp_path / "out", min_kept=1)

    assert report.kept_after_dedupe == 1
    assert any("ETKİN ÇEŞİTLİLİK DÜŞÜK" in warning for warning in report.warnings)


def test_prepare_cot_dataset_rejects_too_small_dataset(tmp_path: Path) -> None:
    source = _write(tmp_path / "cot.jsonl", [_row()])

    with pytest.raises(CotDatasetError, match="eşik"):
        prepare_cot_dataset(source, tmp_path / "out", min_kept=5)


# --------------------------------------------------------------------------- #
# training_steps_for
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("examples", "expected"),
    [
        (0, 60),
        (4, 60),  # 4 * 2 / 16 = 0.5 -> alt sınır
        (1326, 166),  # 1326 * 2 / 16
        (100_000, 1500),  # üst sınır
    ],
)
def test_training_steps_for_clamps(examples: int, expected: int) -> None:
    assert training_steps_for(examples) == expected


def test_training_steps_for_respects_epochs_and_batch() -> None:
    assert training_steps_for(320, epochs=3, batch=2, grad_accum=8) == 60
    assert training_steps_for(3200, epochs=3, batch=2, grad_accum=8) == 600


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_returns_4_on_missing_input(tmp_path: Path) -> None:
    from gokturk.cot_dataset import main

    assert main(["--input", str(tmp_path / "yok.jsonl"), "--out-dir", str(tmp_path / "o")]) == 4


def test_cli_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gokturk.cot_dataset import main

    rows = [_row(f"{PROMPT} Varyant {i} (id: {i})", f"{ANSWER} Varyant {i}") for i in range(12)]
    source = _write(tmp_path / "cot.jsonl", rows)

    code = main(
        ["--input", str(source), "--out-dir", str(tmp_path / "out"), "--min-kept", "1", "--json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input_rows"] == 12
    assert payload["written"]["train"] > 0


# --------------------------------------------------------------------------- #
# depo sözleşmesi: dosya yolu, config ve notebook aynı hikâyeyi anlatmalı
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]


def test_cli_default_input_matches_tracked_dataset_location() -> None:
    """Varsayılan yol gerçekten var olan dosyayı göstermeli.

    Dosya eskiden depo kökündeydi ve varsayılan yol `data/datasets/...` idi; bu
    uyuşmazlık CLI'yi doğrudan `CotDatasetError` ile düşürüyordu.
    """
    from gokturk.cot_dataset import main as _main  # noqa: F401  (import kanıtı)

    dataset = ROOT / "data" / "datasets" / "evren_cot_10k.jsonl"
    assert dataset.is_file(), "CoT veri seti data/datasets/ altında izlenmiyor"
    assert not (ROOT / "evren_cot_10k.jsonl").exists(), "dosya kökte kalmamalı"


def test_cot_training_config_targets_processed_split() -> None:
    import yaml

    cfg = yaml.safe_load(
        (ROOT / "configs" / "training_qwen3_5_4b_cot_t4.yaml").read_text(encoding="utf-8")
    )

    assert cfg["model"]["name_or_path"] == "Qwen/Qwen3.5-4B"
    assert cfg["data"]["train_file"] == "data/processed/train.jsonl"
    assert cfg["data"]["eval_file"] == "data/processed/valid.jsonl"
    # llama.cpp LoRA dönüştürücüsü embed/lm_head ile uyumlu değil.
    assert "embed_tokens" not in cfg["lora"]["target_modules"]
    assert "lm_head" not in cfg["lora"]["target_modules"]


def test_notebook_wires_cot_conversion_and_computed_steps() -> None:
    import json as _json

    notebook = _json.loads(
        (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    joined = "\n".join("".join(c["source"]) for c in code_cells)

    # Her kod hücresi derlenebilmeli: bozuk bir hücre Kaggle'da ancak çalışırken belli olur.
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"cell{index}", "exec")

    assert "prepare_cot_dataset" in joined
    assert "training_steps_for" in joined
    assert "qwen3.5-4b-cot" in joined
    assert "max_steps = 60" not in joined, "adım sayısı veriden hesaplanmalı, sabit olmamalı"


def test_dataset_doc_is_linked_and_states_the_measured_diversity() -> None:
    doc = ROOT / "docs" / "veri-seti-hazirlama.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    for needle in ("920", "%9,2", "train:1271", "cot_report.json", "training_steps_for"):
        assert needle in text, f"belgede eksik: {needle}"
    assert "veri-seti-hazirlama.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "veri-seti-hazirlama.md" in (ROOT / "data" / "README.md").read_text(encoding="utf-8")
