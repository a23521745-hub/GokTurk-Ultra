"""Çevrimdışı eğitim kapısı: `train.py --offline` ağa çıkmadan karar vermeli."""

from __future__ import annotations

import json
from pathlib import Path

from gokturk import train as gtu_train

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training_qwen3_5_4b_cot_t4.yaml"


def _fake_model(root: Path) -> Path:
    target = root / "Qwen3.5-4B"
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    return target


def _run(argv, monkeypatch, tmp_path, *, model_dir=None):
    if model_dir is not None:
        monkeypatch.setenv("GTU_MODEL_SEARCH_ROOTS", str(model_dir))
    called = {}

    def fake_run_training(cfg, **kwargs):
        called["cfg"] = cfg
        called["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(gtu_train, "run_training", fake_run_training)
    code = gtu_train.main(argv)
    return code, called


def test_offline_refuses_when_weights_are_not_local(monkeypatch, tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    code, called = _run(
        [
            "--offline",
            "--config",
            str(CONFIG),
            "--train-file",
            str(data),
            "--eval-file",
            str(data),
        ],
        monkeypatch,
        tmp_path,
        model_dir=tmp_path / "empty",
    )

    assert code == 4, "ağ yokken sessizce indirmeye çalışmamalı"
    assert called == {}
    out = capsys.readouterr()
    assert "Çevrimdışı eğitim denetimi" in out.out
    assert "AĞ İHTİYACI   : VAR" in out.out
    assert "model yerel değil" in out.err


def test_offline_proceeds_with_local_weights(monkeypatch, tmp_path, capsys):
    weights = tmp_path / "weights"
    _fake_model(weights)
    data = tmp_path / "train.jsonl"
    data.write_text('{"id":"1","prompt":"p","response":"r","source":{"name":"x"}}\n', encoding="utf-8")

    code, called = _run(
        [
            "--offline",
            "--config",
            str(CONFIG),
            "--train-file",
            str(data),
            "--eval-file",
            str(data),
            "--plan-only",
        ],
        monkeypatch,
        tmp_path,
        model_dir=weights,
    )

    assert code == 0, "yerel ağırlık + veri varken devam etmeli"
    assert called, "run_training çağrılmalı"
    out = capsys.readouterr().out
    assert "YEREL" in out
    assert "SONUÇ         : Bu koşu tamamen çevrimdışı çalışabilir." in out


def test_offline_refuses_when_data_is_missing(monkeypatch, tmp_path, capsys):
    weights = tmp_path / "weights"
    _fake_model(weights)

    code, called = _run(
        ["--offline", "--config", str(CONFIG), "--train-file", str(tmp_path / "yok.jsonl")],
        monkeypatch,
        tmp_path,
        model_dir=weights,
    )

    assert code == 4
    assert called == {}
    assert "veri dosyaları eksik" in capsys.readouterr().err


def test_offline_sets_hf_env_switches(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    _fake_model(weights)
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)

    _run(
        ["--offline", "--config", str(CONFIG), "--train-file", str(data), "--eval-file", str(data)],
        monkeypatch,
        tmp_path,
        model_dir=weights,
    )

    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_without_offline_flag_no_network_check_is_printed(monkeypatch, tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    code, called = _run(
        ["--config", str(CONFIG), "--train-file", str(data), "--eval-file", str(data), "--plan-only"],
        monkeypatch,
        tmp_path,
    )

    assert code == 0
    assert called
    assert "Çevrimdışı eğitim denetimi" not in capsys.readouterr().out


def test_offline_doc_is_linked_and_states_the_two_phases():
    doc = ROOT / "docs" / "cevrimdisi-egitim.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    for needle in ("Veri üretimi", "Eğitim", "--offline", "HF_HUB_OFFLINE",
                   "GTU_MODEL_SEARCH_ROOTS", "resume"):
        assert needle in body, f"belgede eksik: {needle}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "cevrimdisi-egitim.md" in readme
