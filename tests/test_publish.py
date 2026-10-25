"""Publishing: model card honesty, dry-run push, GGUF feasibility, secrets."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from gokturk.publish import (
    DEFAULT_QUANT,
    QUANTS,
    PublishError,
    build_model_card,
    estimate_sizes,
    main,
    plan_gguf,
    plan_push,
    push_adapter,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 16}', encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 2048)
    return d


MANIFEST = {
    "kind": "training",
    "backend": "unsloth",
    "config": {
        "model": {"name_or_path": "unsloth/gpt-oss-20b"},
        "optimization": {"method": "qlora"},
        "data": {"max_seq_len": 1024},
    },
    "data": {"train_examples": 9000, "eval_examples": 500},
    "metrics": {"train_loss": 0.9143, "eval_loss": 1.552, "global_step": 60},
}


# ---------------------------------------------------------------------------
# model card
# ---------------------------------------------------------------------------
def test_card_reports_manifest_numbers():
    card = build_model_card(MANIFEST, repo_id="user/gokturk-lora")
    assert "unsloth/gpt-oss-20b" in card
    assert "9000" in card
    assert "0.9143" in card
    assert "1.552" in card


def test_card_states_it_has_no_measured_benchmark():
    card = build_model_card(MANIFEST, repo_id="user/x")
    lowered = card.lower()
    assert "benchmark" in lowered
    # The exact honesty markers the project uses elsewhere.
    assert "ölçülmüş bir benchmark sonucu yoktur" in lowered
    for claim in ("accuracy", "doğruluk oranı %", "mmlu", "hellaswag"):
        assert claim not in lowered


def test_card_without_manifest_says_bilinmiyor():
    card = build_model_card(None, repo_id="user/x")
    assert "bilinmiyor" in card
    assert "ölçülmedi" in card


def test_card_never_contains_a_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecretvalue123456")
    card = build_model_card(MANIFEST, repo_id="user/x")
    assert "hf_supersecretvalue123456" not in card
    assert "HF_TOKEN" not in card.replace("`", "")


# ---------------------------------------------------------------------------
# push planning (no network)
# ---------------------------------------------------------------------------
def test_plan_push_ok(adapter_dir, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    plan = plan_push(adapter_dir, "user/gokturk-lora", private=True)
    assert plan.ok
    assert plan.token_present is True
    assert "adapter_config.json" in plan.files
    assert plan.total_bytes > 0
    assert "hf_dummy" not in json.dumps(plan.as_dict())


def test_plan_push_without_token_explains_how(adapter_dir, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    plan = plan_push(adapter_dir, "user/x")
    assert not plan.ok
    assert any("HF_TOKEN" in p for p in plan.problems)


def test_plan_push_rejects_bad_repo_id(adapter_dir, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    for bad in ("nodash", "a/b/c", "/x", "x/"):
        plan = plan_push(adapter_dir, bad)
        assert not plan.ok, bad
        assert any("repo_id" in p for p in plan.problems)


def test_plan_push_rejects_non_adapter_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    empty = tmp_path / "empty"
    empty.mkdir()
    plan = plan_push(empty, "user/x")
    assert not plan.ok
    assert any("PEFT" in p or "boş" in p for p in plan.problems)


# ---------------------------------------------------------------------------
# push with an injected fake client
# ---------------------------------------------------------------------------
class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


def test_push_adapter_uploads_and_writes_card(adapter_dir, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    fake = _FakeApi()
    result = push_adapter(
        adapter_dir,
        "user/gokturk-lora",
        private=True,
        manifest=MANIFEST,
        api=fake,
    )
    assert result["ok"] is True
    assert result["url"] == "https://huggingface.co/user/gokturk-lora"
    assert fake.created[0][0] == "user/gokturk-lora"
    assert fake.created[0][1]["repo_type"] == "model"
    assert fake.created[0][1]["private"] is True
    assert fake.uploaded[0]["repo_id"] == "user/gokturk-lora"
    card = (adapter_dir / "README.md").read_text(encoding="utf-8")
    assert card.startswith("# user/gokturk-lora")
    assert "hf_dummy" not in card


def test_push_adapter_never_calls_api_when_plan_fails(adapter_dir, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = _FakeApi()
    with pytest.raises(PublishError):
        push_adapter(adapter_dir, "user/x", api=fake)
    assert fake.created == [] and fake.uploaded == []


def test_push_adapter_lazy_imports_hub(adapter_dir, monkeypatch):
    """With an injected client, huggingface_hub must not be imported at all."""

    class _Boom(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover - only on failure
            raise AssertionError("huggingface_hub was imported although api= was given")

    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _Boom("huggingface_hub"))
    result = push_adapter(adapter_dir, "user/x", api=_FakeApi())
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# GGUF planning
# ---------------------------------------------------------------------------
def test_estimate_sizes_matches_published_magnitudes():
    sizes = estimate_sizes(20.9, quant="Q4_K_M")
    assert 40 <= sizes["merged_gb"] <= 44      # 20.9B * 2 bytes
    # Published gpt-oss-20b builds: Q4_K_M ≈ 11.7 GB, MXFP4 ≈ 12.1 GB, because
    # the MoE experts stay at MXFP4 in every quant.
    assert 10.0 <= sizes["final_gb"] <= 12.0
    assert 10.0 <= estimate_sizes(20.9, quant="mxfp4")["final_gb"] <= 12.0


def test_estimate_sizes_rejects_unknown_quant_and_dtype():
    with pytest.raises(PublishError, match="quant"):
        estimate_sizes(20.9, quant="Q9_Z")
    with pytest.raises(PublishError, match="dtype"):
        estimate_sizes(20.9, merge_dtype="float8")


def test_default_quant_is_the_native_one():
    assert DEFAULT_QUANT == "mxfp4"
    assert QUANTS["mxfp4"]["llama_quantize_type"] == "MXFP4_MOE"
    # Q4_K_M stays available but must warn about the FFN experts.
    assert "MXFP4" in QUANTS["Q4_K_M"]["note"]


def test_plan_gguf_refuses_on_kaggle_sized_machine(adapter_dir):
    plan = plan_gguf(
        adapter_dir,
        available_ram_gb=29.0,
        available_disk_gb=20.0,
        out_dir=str(adapter_dir.parent / "gguf"),
    )
    assert plan["feasible"] is False
    assert any("RAM" in p for p in plan["problems"])
    assert any("Disk" in p for p in plan["problems"])
    # Commands are still emitted so the operator can run them elsewhere.
    assert len(plan["steps"]) >= 4


def test_plan_gguf_accepts_a_big_machine(adapter_dir):
    plan = plan_gguf(
        adapter_dir,
        available_ram_gb=128.0,
        available_disk_gb=500.0,
        out_dir=str(adapter_dir.parent / "gguf"),
    )
    assert plan["feasible"] is True
    assert plan["problems"] == []
    commands = " ".join(" ".join(step["command"]) for step in plan["steps"])
    assert "convert_hf_to_gguf.py" in commands
    assert "llama-quantize" in commands
    assert "MXFP4_MOE" in commands
    # The converter has no --outtype mxfp4; bf16 is the documented intermediate.
    assert "--outtype bf16" in commands
    assert "mxfp4" not in commands.split("--outtype")[1].split()[0]


def test_plan_gguf_never_ships_the_intermediate(adapter_dir):
    plan = plan_gguf(
        adapter_dir,
        available_ram_gb=128.0,
        available_disk_gb=500.0,
        out_dir=str(adapter_dir.parent / "gguf"),
    )
    last = plan["steps"][-1]
    assert last["command"][0] == "rm"
    assert "model-bf16.gguf" in " ".join(last["command"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_card_only_prints_card(capsys):
    code = main(["--card-only", "--repo-id", "user/x"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("# user/x")
    assert "benchmark" in out.lower()


def test_cli_dry_run_push_reports_plan(adapter_dir, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code = main(["--adapter", str(adapter_dir), "--repo-id", "user/x"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2                     # token missing -> not ok
    assert payload["dry_run"] is True
    assert payload["token_present"] is False
    assert payload["problems"]


def test_cli_gguf_plan_exit_codes(adapter_dir, capsys):
    code = main(
        ["--gguf-plan", "--adapter", str(adapter_dir), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    # The test machine is small; whatever it reports, the code must match.
    assert code == (0 if payload["feasible"] else 2)
    assert payload["sizes_gb"]["merged_gb"] > 0


def test_cli_gguf_plan_requires_adapter(capsys):
    assert main(["--gguf-plan"]) == 4
    assert "adapter" in capsys.readouterr().err


def test_cli_missing_arguments(capsys):
    assert main([]) == 4
    assert "repo-id" in capsys.readouterr().err


def test_publish_module_has_no_hardcoded_token():
    import re

    text = (ROOT / "src" / "gokturk" / "publish.py").read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "api_key" not in text.lower()
    # The token may only be read from the environment.
    assert 'os.environ.get(HF_TOKEN_ENV' in text
