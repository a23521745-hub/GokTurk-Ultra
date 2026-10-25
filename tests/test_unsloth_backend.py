"""Unsloth backend: honest preflight, API drift handling, no silent fallback.

Unsloth itself is a heavy optional dependency and is *not* installed in CI, so
these tests inject a stub module into ``sys.modules``. That exercises the real
code paths (signature filtering, error messages, preflight findings) without
pretending a GPU run happened.
"""

from __future__ import annotations

import sys
import types

import pytest

from gokturk import unsloth_backend
from gokturk.env_check import GpuInfo, HardwareInfo, check_environment, estimate_memory


def _t4_hardware() -> HardwareInfo:
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


def _a100_hardware() -> HardwareInfo:
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
        gpus=[GpuInfo(0, "NVIDIA A100", 79.3, 78.0, "8.0")],
        bf16_supported=True,
        fp16_supported=True,
    )


class _StubModel:
    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


class _StubTokenizer:
    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


def _install_stub(monkeypatch, *, version="2026.1.1", symbol="FastLanguageModel",
                  from_pretrained_sig=None, peft_sig=None):
    """Install a fake ``unsloth`` module that records calls."""
    calls: dict[str, list[dict]] = {"from_pretrained": [], "get_peft_model": []}
    model = _StubModel()

    if from_pretrained_sig is None:

        def from_pretrained(model_name=None, max_seq_length=None, load_in_4bit=None, dtype=None):
            calls["from_pretrained"].append(dict(locals()))
            return model, _StubTokenizer()

    else:  # emulate an older/newer signature

        def from_pretrained(model_name=None, **kwargs):  # type: ignore[misc]
            calls["from_pretrained"].append({"model_name": model_name, **kwargs})
            return model, _StubTokenizer()

        from_pretrained.__signature__ = from_pretrained_sig  # type: ignore[attr-defined]

    if peft_sig is None:

        def get_peft_model(model=None, r=None, lora_alpha=None, lora_dropout=None,
                           bias=None, target_modules=None,
                           use_gradient_checkpointing=None, random_state=None,
                           max_seq_length=None):
            calls["get_peft_model"].append(dict(locals()))
            return model

    else:

        def get_peft_model(model=None, **kwargs):  # type: ignore[misc]
            calls["get_peft_model"].append(dict(kwargs))
            return model

        get_peft_model.__signature__ = peft_sig  # type: ignore[attr-defined]

    fast = types.SimpleNamespace(from_pretrained=from_pretrained, get_peft_model=get_peft_model)
    module = types.ModuleType("unsloth")
    module.__version__ = version
    setattr(module, symbol, fast)
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return calls, model


class _Cfg:
    """Minimal stand-in for TrainingConfig used by the backend functions."""

    class model:
        name_or_path = "unsloth/gpt-oss-20b"
        dtype = "auto"
        load_in_4bit = True

    class lora:
        r = 16
        alpha = 32
        dropout = 0.0
        bias = "none"
        target_modules = ["auto"]

    class optimization:
        seed = 42


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def test_missing_unsloth_is_reported_not_raised():
    status = unsloth_backend.detect_unsloth()
    assert status.available is False
    assert "No module named 'unsloth'" in status.detail


def test_detect_prefers_fast_language_model(monkeypatch):
    _install_stub(monkeypatch, version="9.9.9")
    status = unsloth_backend.detect_unsloth()
    assert status.available is True
    assert status.version == "9.9.9"
    assert status.fast_model_symbol == "FastLanguageModel"
    assert "get_peft_model" in status.methods


def test_detect_reports_incomplete_module(monkeypatch):
    module = types.ModuleType("unsloth")
    monkeypatch.setitem(sys.modules, "unsloth", module)
    status = unsloth_backend.detect_unsloth()
    assert status.available is False
    assert "FastLanguageModel/FastModel" in status.detail


def test_load_without_unsloth_raises_actionable_error():
    with pytest.raises(unsloth_backend.UnslothUnavailableError) as excinfo:
        unsloth_backend.load_model_and_tokenizer(_Cfg(), max_seq_length=1024)
    message = str(excinfo.value)
    assert "pip install unsloth" in message
    assert "isteğe bağlı" in message  # makes clear it is optional


# ---------------------------------------------------------------------------
# load / adapters
# ---------------------------------------------------------------------------
def test_load_passes_documented_kwargs(monkeypatch):
    calls, _ = _install_stub(monkeypatch)
    model, tokenizer = unsloth_backend.load_model_and_tokenizer(_Cfg(), max_seq_length=2048)
    assert isinstance(model, _StubModel)
    call = calls["from_pretrained"][0]
    assert call["model_name"] == "unsloth/gpt-oss-20b"
    assert call["load_in_4bit"] is True
    assert call["max_seq_length"] == 2048
    assert call["dtype"] is None  # "auto" maps to Unsloth's own detection


def test_apply_lora_uses_unsloth_targets_and_checkpointing(monkeypatch):
    calls, _ = _install_stub(monkeypatch)
    unsloth_backend.apply_lora(_StubModel(), _Cfg(), max_seq_length=1024)
    call = calls["get_peft_model"][0]
    assert call["r"] == 16
    assert call["lora_alpha"] == 32
    assert call["use_gradient_checkpointing"] == "unsloth"
    assert "q_proj" in call["target_modules"] and "gate_proj" in call["target_modules"]
    assert call["random_state"] == 42


def test_unknown_kwargs_are_filtered_not_crashed(monkeypatch):
    """Older Unsloth moved max_seq_length out of get_peft_model."""
    import inspect

    def old_signature(r=None, lora_alpha=None, target_modules=None,
                      use_gradient_checkpointing=None, random_state=None):
        ...

    _install_stub(monkeypatch, peft_sig=inspect.signature(old_signature))
    model = unsloth_backend.apply_lora(_StubModel(), _Cfg(), max_seq_length=1024)
    assert isinstance(model, _StubModel)


def test_save_adapter_writes_peft_layout(monkeypatch, tmp_path):
    _install_stub(monkeypatch)
    model, tokenizer = unsloth_backend.load_model_and_tokenizer(_Cfg(), max_seq_length=64)
    unsloth_backend.save_adapter(model, tokenizer, str(tmp_path))
    assert model.saved_to == str(tmp_path)
    assert tokenizer.saved_to == str(tmp_path)


def test_incompatible_api_raises_compatibility_error(monkeypatch):
    module = types.ModuleType("unsloth")
    module.__version__ = "0.0.1"
    module.FastModel = types.SimpleNamespace(from_pretrained=None)  # missing both
    monkeypatch.setitem(sys.modules, "unsloth", module)
    with pytest.raises(unsloth_backend.UnslothCompatibilityError):
        unsloth_backend.load_model_and_tokenizer(_Cfg(), max_seq_length=64)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def test_preflight_blocks_when_unsloth_missing(monkeypatch):
    monkeypatch.setattr("gokturk.env_check.detect_hardware", lambda *a, **k: _t4_hardware())
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=True, hardware=_t4_hardware(), max_seq_length=1024
    )
    codes = {f.code for f in report.blocking}
    assert "unsloth" in codes
    assert report.method == "unsloth-qlora"


def test_preflight_passes_on_16gb_with_unsloth_installed(monkeypatch):
    _install_stub(monkeypatch)
    monkeypatch.setattr("gokturk.env_check.detect_hardware", lambda *a, **k: _t4_hardware())
    monkeypatch.setattr(
        "gokturk.env_check.detect_packages",
        lambda: [],
    )
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=True, hardware=_t4_hardware(), max_seq_length=1024
    )
    blocking = {f.code for f in report.blocking}
    assert "unsloth-vram" not in blocking
    assert "gptoss-qlora-generic-too-small" not in blocking
    warnings = {f.code for f in report.warnings}
    assert "unsloth-fp16-moe" in warnings  # T4 has no bf16
    infos = {f.code for f in report.findings}
    assert "unsloth-single-gpu" in infos


def test_preflight_blocks_bf16_lora_on_16gb(monkeypatch):
    _install_stub(monkeypatch)
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=False, hardware=_t4_hardware(), max_seq_length=1024
    )
    assert "unsloth-vram" in {f.code for f in report.blocking}


def test_preflight_reports_vram_ok_on_a100(monkeypatch):
    _install_stub(monkeypatch)
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=False, hardware=_a100_hardware(), max_seq_length=2048
    )
    codes = {f.code for f in report.findings}
    assert "unsloth-vram-ok" in codes
    assert not [f for f in report.blocking if f.code.startswith("unsloth")]


def test_preflight_does_not_require_bitsandbytes(monkeypatch):
    _install_stub(monkeypatch)
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=True, hardware=_t4_hardware(), max_seq_length=1024
    )
    assert "bnb-missing" not in {f.code for f in report.findings}


def test_generic_backend_still_blocks_gpt_oss_qlora():
    """The honest default must not be weakened by adding the Unsloth backend."""
    report = check_environment(
        model_name="openai/gpt-oss-20b",
        method="qlora",
        dtype="auto",
        load_in_4bit=True,
        hardware=_t4_hardware(),
        backend="transformers",
    )
    assert "gptoss-qlora-generic-too-small" in {f.code for f in report.blocking}


def test_estimate_is_labelled_for_unsloth_reference_numbers():
    estimate = estimate_memory(20.9e9, "qlora", "fp16")
    report = unsloth_backend.preflight_unsloth(
        "unsloth/gpt-oss-20b", load_in_4bit=True, hardware=_t4_hardware(), max_seq_length=1024
    )
    assert report.estimate is not None
    assert any("Unsloth" in note for note in report.estimate.assumptions)
    assert estimate.total_gb > 0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        check_environment(
            model_name="x",
            method="qlora",
            dtype="auto",
            load_in_4bit=True,
            hardware=_t4_hardware(),
            backend="magic",
        )
