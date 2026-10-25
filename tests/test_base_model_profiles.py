"""Alternative base models: configs load, preflight knows them, docs stay honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gokturk.config import load_training_config
from gokturk.env_check import KNOWN_MODELS, resolve_known_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "configs" / "training_qwen3_5_9b_t4.yaml",
    ROOT / "configs" / "training_qwen3_14b_t4.yaml",
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_profile_loads_and_is_4bit(path: Path):
    cfg = load_training_config(path)
    assert cfg.model.load_in_4bit is True
    assert cfg.optimization.method == "qlora"
    assert cfg.model.name_or_path.startswith("Qwen/")
    assert cfg.data.max_seq_len >= 512
    assert "q_proj" in cfg.lora.target_modules


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_profile_is_a_measurement_run_first(path: Path):
    cfg = load_training_config(path)
    # 60-step measurement run before any long run.
    assert cfg.optimization.max_steps == 60


def test_qwen_models_are_known_to_the_preflight():
    for model in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3-14B"):
        facts = resolve_known_model(model)
        assert facts is not None, model
        assert facts["params"] > 1e9
        assert facts["license"] == "Apache-2.0"


def test_mirror_ids_resolve_to_the_same_facts():
    assert resolve_known_model("unsloth/Qwen3.5-9B-GGUF")["params"] == 9.0e9


def test_profile_choice_is_documented():
    doc = (ROOT / "docs" / "taban-model-secimi.md").read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-9B" in doc
    assert "Qwen/Qwen3-14B" in doc
    # The Spaces reality must stay documented: free Spaces cannot host 24/7 chat.
    assert "ZeroGPU" in doc
    assert "5 dakika" in doc
    # Multimodal caveat must not be dropped from the doc.
    assert "AutoProcessor" in doc


def test_known_models_table_matches_publish_sizes():
    """env_check (training) and publish (serving) must agree on parameter counts."""
    from gokturk.publish import BASE_PARAMS_B

    for model, facts in KNOWN_MODELS.items():
        if model in BASE_PARAMS_B:
            assert BASE_PARAMS_B[model] * 1e9 == pytest.approx(facts["params"])


def test_no_profile_uses_a_hardcoded_token():
    for path in CONFIGS:
        text = path.read_text(encoding="utf-8")
        assert "hf_" not in text
        assert "api_key" not in text.lower()
        # Configs must be loadable JSON-free YAML with no secrets.
        json.dumps(text)
