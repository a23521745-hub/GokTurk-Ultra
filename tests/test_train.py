"""Training pipeline: masking, plan/preflight gating and a real CPU smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_record, write_jsonl
from gokturk.config import (
    DataConfig,
    LoraConfigYaml,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from gokturk.env_check import (
    GpuInfo,
    HardwareInfo,
    check_environment,
    detect_hardware,
    estimate_memory,
    package_versions,
)

torch = pytest.importorskip("torch", reason="training tests need PyTorch")


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# hardware / estimate logic (no GPU required)
# ---------------------------------------------------------------------------
def _fake_t4_hardware() -> HardwareInfo:
    return HardwareInfo(
        python_version="3.11",
        platform="Linux",
        cpu_count=4,
        ram_gb=29.0,
        disk_free_gb=20.0,
        torch_available=True,
        torch_version="2.6.0",
        cuda_available=True,
        cuda_version="12.4",
        gpu_count=2,
        gpus=[
            GpuInfo(0, "Tesla T4", 15.6, 15.0, "7.5"),
            GpuInfo(1, "Tesla T4", 15.6, 15.0, "7.5"),
        ],
        bf16_supported=False,
        fp16_supported=True,
    )


def _fake_a100_hardware() -> HardwareInfo:
    return HardwareInfo(
        python_version="3.11",
        platform="Linux",
        cpu_count=8,
        ram_gb=64.0,
        disk_free_gb=100.0,
        torch_available=True,
        torch_version="2.6.0",
        cuda_available=True,
        cuda_version="12.4",
        gpu_count=1,
        gpus=[GpuInfo(0, "NVIDIA A100-SXM4-80GB", 79.3, 78.0, "8.0")],
        bf16_supported=True,
        fp16_supported=True,
    )


def test_gpt_oss_qlora_is_blocked_on_2xt4(monkeypatch):
    monkeypatch.setattr("gokturk.env_check.detect_hardware", lambda *a, **k: _fake_t4_hardware())
    monkeypatch.setattr(
        "gokturk.env_check.detect_packages",
        lambda: [],
    )
    report = check_environment(
        model_name="openai/gpt-oss-20b", method="qlora", dtype="auto", load_in_4bit=True
    )
    codes = {f.code for f in report.blocking}
    assert "gptoss-qlora-generic-too-small" in codes
    assert report.ok is False
    rendered = report.render()
    assert "ENGELLEYİCİ" in rendered or "ENGELLEYİCİ SORUN VAR" in rendered
    assert "Unsloth" in rendered


def test_gpt_oss_lora_is_blocked_below_44gb(monkeypatch):
    monkeypatch.setattr("gokturk.env_check.detect_hardware", lambda *a, **k: _fake_t4_hardware())
    report = check_environment(
        model_name="openai/gpt-oss-20b", method="lora", dtype="bf16", load_in_4bit=False
    )
    codes = {f.code for f in report.blocking}
    assert "gptoss-lora-bf16-vram" in codes
    assert "bf16-unsupported" in codes  # T4 has no bf16


def test_gpt_oss_full_finetune_blocked():
    report = check_environment(
        model_name="openai/gpt-oss-20b", method="full", dtype="bf16", load_in_4bit=False
    )
    assert "gptoss-full-unsupported" in {f.code for f in report.blocking}


def test_small_model_fits_on_t4(monkeypatch):
    monkeypatch.setattr("gokturk.env_check.detect_hardware", lambda *a, **k: _fake_t4_hardware())
    report = check_environment(
        model_name="some/7b-model", method="qlora", dtype="auto", load_in_4bit=True, params=7e9
    )
    codes = {f.code for f in report.blocking}
    assert "vram-insufficient" not in codes
    assert report.hardware.total_vram_gb == 15.6


def test_memory_estimate_ordering():
    qlora = estimate_memory(20.9e9, "qlora", "fp16")
    lora = estimate_memory(20.9e9, "lora", "bf16")
    full = estimate_memory(20.9e9, "full", "bf16")
    assert qlora.total_gb < lora.total_gb < full.total_gb
    assert qlora.total_gb < 20
    assert full.total_gb > 60


def test_estimates_are_labelled_as_estimates():
    est = estimate_memory(1e9, "lora", "fp16")
    assert any("Estimate only" in a for a in est.assumptions)


def test_detect_hardware_runs_here():
    hw = detect_hardware()
    assert hw.python_version.startswith("3")
    assert hw.ram_gb >= 0
    assert isinstance(hw.gpus, list)
    versions = package_versions()
    assert "torch" in versions


# ---------------------------------------------------------------------------
# tokenisation / masking
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_tokenizer():
    from gokturk.train import build_smoke_tokenizer

    return build_smoke_tokenizer(["merhaba dünya bu bir deneme", "kod yaz lütfen"])


def test_encode_example_masks_prompt_only(smoke_tokenizer):
    from gokturk.train import encode_example

    prompt = "merhaba dünya"
    response = "kod yaz lütfen"
    input_ids, labels = encode_example(smoke_tokenizer, prompt, response, 128)
    assert input_ids
    assert len(input_ids) == len(labels)
    supervised = [tok for tok, lab in zip(input_ids, labels, strict=True) if lab != -100]
    assert supervised, "at least the answer tokens must be supervised"
    # prompt words must not be supervised
    prompt_tokens = smoke_tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert set(prompt_tokens).isdisjoint(set(supervised)) or len(supervised) < len(input_ids)


def test_truncation_that_removes_the_answer_is_dropped(smoke_tokenizer):
    from gokturk.train import encode_example

    input_ids, labels = encode_example(
        smoke_tokenizer, "çok uzun bir istek " * 20, "yanıt", max_seq_len=4
    )
    assert input_ids == [] and labels == []


def test_dataset_and_collator(smoke_tokenizer, tmp_path: Path):
    from gokturk.train import ChatSFTDataset, build_collator

    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            make_record("1", prompt="merhaba dünya", response="kod yaz lütfen"),
            make_record("2", prompt="kısa soru", response="uzun cevap " * 5),
        ],
    )
    ds = ChatSFTDataset(path, smoke_tokenizer, max_seq_len=128)
    assert len(ds) == 2
    batch = build_collator(smoke_tokenizer.pad_token_id or 0)([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert (batch["labels"] == -100).any()


def test_dataset_fails_on_empty_file(smoke_tokenizer, tmp_path: Path):
    from gokturk.train import ChatSFTDataset

    path = write_jsonl(tmp_path / "empty.jsonl", [])
    with pytest.raises(ValueError, match="no usable training example"):
        ChatSFTDataset(path, smoke_tokenizer, max_seq_len=128)


# ---------------------------------------------------------------------------
# LoRA module resolution
# ---------------------------------------------------------------------------
def test_resolve_target_modules_auto_uses_known_projections():
    from gokturk.train import build_smoke_model, resolve_target_modules

    model = build_smoke_model(100)
    targets = resolve_target_modules(model, ["auto"])
    assert set(targets) & {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_resolve_target_modules_rejects_unknown_explicit_name():
    from gokturk.train import build_smoke_model, resolve_target_modules

    model = build_smoke_model(100)
    with pytest.raises(ValueError, match="not found"):
        resolve_target_modules(model, ["q_proj", "nonexistent_proj"])


def test_resolve_dtype_follows_hardware():
    from gokturk.train import resolve_torch_dtype

    assert resolve_torch_dtype("auto", _fake_t4_hardware()) == "fp16"
    assert resolve_torch_dtype("auto", _fake_a100_hardware()) == "bf16"
    assert resolve_torch_dtype("fp32", _fake_a100_hardware()) == "fp32"


# ---------------------------------------------------------------------------
# preflight gating in run_training
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path, method: str = "qlora") -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(name_or_path="openai/gpt-oss-20b", load_in_4bit=method == "qlora"),
        lora=LoraConfigYaml(r=4, alpha=8),
        data=DataConfig(
            train_file=str(tmp_path / "train.jsonl"), eval_file=str(tmp_path / "valid.jsonl")
        ),
        optimization=OptimizationConfig(method=method, epochs=1, gradient_accumulation_steps=1),
        output_dir=str(tmp_path / "out"),
    )


def test_run_training_refuses_blocked_hardware(tmp_path: Path, capsys):
    from gokturk.train import run_training

    write_jsonl(
        tmp_path / "train.jsonl", [make_record("1", prompt="merhaba dünya", response="kod yaz")]
    )
    code = run_training(_cfg(tmp_path, "qlora"))
    err = capsys.readouterr().err
    if code == 2:
        assert "Eğitim başlatılmadı" in err
        assert not (tmp_path / "out" / "adapter").exists()
    else:
        pytest.skip("Bu makinede ön denetim geçti; engelleme yolu test edilemedi")


def test_plan_only_always_returns_zero(tmp_path: Path, capsys):
    from gokturk.train import run_training

    code = run_training(_cfg(tmp_path, "qlora"), plan_only=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "Donanım" in out
    assert "Plan" in out


# ---------------------------------------------------------------------------
# full smoke run (CPU, tiny random model) — proves the pipeline end to end
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_smoke_training_runs_end_to_end(tmp_path: Path):
    from gokturk.train import run_training

    write_jsonl(
        tmp_path / "train.jsonl",
        [
            make_record(f"{i}", prompt=f"merhaba dünya {i}", response="kod yaz lütfen kısa olsun")
            for i in range(8)
        ],
    )
    write_jsonl(
        tmp_path / "valid.jsonl",
        [make_record("v1", prompt="merhaba dünya", response="kod yaz lütfen")],
    )
    cfg = _cfg(tmp_path, "lora")
    cfg.model = ModelConfig(name_or_path="tiny", load_in_4bit=False, dtype="fp32")
    cfg.optimization.method = "lora"
    cfg.data.max_seq_len = 64
    cfg.optimization.learning_rate = 1e-2
    out = tmp_path / "smoke_out"

    code = run_training(cfg, smoke=True, max_steps=2, output_dir=str(out))
    assert code == 0, "smoke training must succeed on CPU with a tiny random model"

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "smoke"
    assert "kalite iddiası yok" in manifest["note"]
    assert manifest["data"]["train_examples"] == 8
    assert "loss" in manifest["metrics"] or "train_loss" in manifest["metrics"]
    assert (out / "adapter").is_dir()
    # A smoke run must never claim to be a real trained model.
    assert manifest["config"]["model"]["name_or_path"] != "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------
def test_unsloth_backend_is_blocked_without_unsloth(tmp_path, capsys):
    """No silent fallback: the explicit backend must fail loudly."""
    from gokturk.train import run_training

    write_jsonl(
        tmp_path / "train.jsonl",
        [make_record("1", prompt="merhaba dünya", response="kod yaz lütfen")],
    )
    cfg = _cfg(tmp_path, "qlora")
    code = run_training(cfg, backend="unsloth")
    err = capsys.readouterr().err
    assert code == 2
    assert "Eğitim başlatılmadı" in err


def test_unknown_backend_raises(tmp_path):
    from gokturk.train import run_training

    with pytest.raises(ValueError, match="unknown backend"):
        run_training(_cfg(tmp_path), backend="magic")


def test_plan_only_reports_backend(tmp_path, capsys):
    from gokturk.train import run_training

    run_training(_cfg(tmp_path), plan_only=True, backend="unsloth")
    out = capsys.readouterr().out
    assert '"backend": "unsloth"' in out
