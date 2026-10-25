"""Unsloth backend for 4-bit QLoRA fine-tuning of gpt-oss on consumer GPUs.

Why this module exists
----------------------
The generic transformers+peft path in :mod:`gokturk.train` cannot fit
``openai/gpt-oss-20b`` on a 16 GB card: the base checkpoint is stored in MXFP4
and must be upcast before gradient training (≈42 GB in bf16, ≈44 GB VRAM for
bf16 LoRA by Unsloth's published numbers). Unsloth ships fused 4-bit QLoRA
kernels that fit the 20B model in ≈14 GB VRAM, which is what makes Kaggle's
2×T4 free tier plausible.

This backend therefore *delegates* model loading and adapter creation to
Unsloth's public API and keeps everything else (data tokenisation, prompt
masking, validation, manifests) in this repository. That means:

* No unsupported claim is made about the generic path.
* Unsloth remains an **optional** dependency: importing this module never
  imports Unsloth, and every entry point raises a clear error when it is absent.
* The API calls used are the documented ones (``FastLanguageModel`` /
  ``FastModel`` ``from_pretrained`` + ``get_peft_model``). There is **no
  hard-coded fallback to arbitrary kwargs**: unsupported parameters are filtered
  against the installed function's signature, because Unsloth's kwargs have
  changed across releases (e.g. ``max_seq_length`` moved to ``get_peft_model``
  in newer versions).

Verified constraints from Unsloth's gpt-oss documentation (see
``docs/architecture.md`` §6 for sources):

* ``load_in_4bit=True`` → ≈14 GB VRAM for gpt-oss-20b QLoRA.
* ``load_in_4bit=False`` → requires ``unsloth/gpt-oss-20b-BF16`` and ≈44 GB.
* fp16-only GPUs (T4) fall back to float32 for the MoE path; long runs on T4
  were historically unstable. The preflight warns instead of hiding this.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from .env_check import HardwareInfo, PreflightReport

LOGGER = logging.getLogger("gokturk.unsloth_backend")

#: Published VRAM figures for gpt-oss-20b (Unsloth documentation).
UNSLOTH_VRAM_REFERENCES_GB = {"qlora_4bit": 14.0, "lora_16bit": 44.0}
#: Unsloth publishes 4-bit builds of gpt-oss; model ids must still be verified.
SUGGESTED_4BIT_MODEL = "unsloth/gpt-oss-20b"
SUGGESTED_BF16_MODEL = "unsloth/gpt-oss-20b-BF16"

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class UnslothUnavailableError(RuntimeError):
    """Unsloth is not installed (or not importable on this platform)."""


class UnslothCompatibilityError(RuntimeError):
    """Installed Unsloth version does not expose the expected API."""


@dataclass
class UnslothStatus:
    available: bool
    version: str | None
    fast_model_symbol: str | None
    detail: str
    methods: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "fast_model_symbol": self.fast_model_symbol,
            "detail": self.detail,
            "methods": self.methods,
        }


def detect_unsloth() -> UnslothStatus:
    """Probe the installed Unsloth without raising (used by env_check)."""
    try:
        module = importlib.import_module("unsloth")
    except Exception as exc:  # ImportError, OSError (Triton/CUDA), RuntimeError…
        return UnslothStatus(
            available=False,
            version=None,
            fast_model_symbol=None,
            detail=f"{type(exc).__name__}: {exc}",
        )

    version = getattr(module, "__version__", None)
    symbol = None
    for name in ("FastLanguageModel", "FastModel"):
        if hasattr(module, name):
            symbol = name
            break
    if symbol is None:
        return UnslothStatus(
            available=False,
            version=version,
            fast_model_symbol=None,
            detail="unsloth import edildi ama FastLanguageModel/FastModel bulunamadı",
        )
    fast = getattr(module, symbol)
    methods = [m for m in ("from_pretrained", "get_peft_model", "for_inference") if hasattr(fast, m)]
    return UnslothStatus(
        available=True,
        version=version,
        fast_model_symbol=symbol,
        detail="hazır",
        methods=methods,
    )


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the installed function does not accept (API drift guard)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - buildins
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    accepted = {name for name in params if name != "self"}
    dropped = sorted(set(kwargs) - accepted)
    for key in dropped:
        LOGGER.warning("Unsloth bu sürümde '%s' parametresini kabul etmiyor; atlandı", key)
    return {k: v for k, v in kwargs.items() if k in accepted}


def preflight_unsloth(
    model_name: str,
    *,
    load_in_4bit: bool,
    hardware: HardwareInfo,
    max_seq_length: int,
) -> PreflightReport:
    """Hardware/version checks specific to the Unsloth 4-bit path."""
    # Imported lazily to avoid a circular import at module load time.
    from .env_check import (
        SEVERITY_BLOCKING,
        SEVERITY_INFO,
        SEVERITY_WARNING,
        Finding,
        check_environment,
    )

    report = check_environment(
        model_name=model_name,
        method="qlora" if load_in_4bit else "lora",
        dtype="auto",
        load_in_4bit=False,  # bitsandbytes is not used by the Unsloth path
        seq_len=max_seq_length,
        check_hub=False,
        backend="unsloth",
        hardware=hardware,
    )
    add = report.findings.append
    status = detect_unsloth()
    add(
        Finding(
            "unsloth",
            SEVERITY_INFO if status.available else SEVERITY_BLOCKING,
            f"Unsloth {'hazır: ' + str(status.version) if status.available else 'kurulu değil'}",
            status.detail if not status.available else None,
        )
    )

    single_gpu = hardware.total_vram_gb
    required = (
        UNSLOTH_VRAM_REFERENCES_GB["qlora_4bit"]
        if load_in_4bit
        else UNSLOTH_VRAM_REFERENCES_GB["lora_16bit"]
    )
    reference = (
        "Unsloth gpt-oss eğitim belgeleri: 4-bit QLoRA ≈14 GB, bf16 LoRA ≈44 GB (20B)"
    )
    if hardware.cuda_available and single_gpu < required:
        add(
            Finding(
                "unsloth-vram",
                SEVERITY_BLOCKING,
                f"Unsloth yolu için yayımlanmış gereksinim ≈{required:.0f} GB VRAM, "
                f"bu GPU {single_gpu:.1f} GB.",
                reference,
            )
        )
    elif hardware.cuda_available:
        add(
            Finding(
                "unsloth-vram-ok",
                SEVERITY_INFO,
                f"VRAM {single_gpu:.1f} GB ≥ yayımlanmış ≈{required:.0f} GB gereksinim.",
                reference,
            )
        )

    if hardware.gpu_count > 1:
        add(
            Finding(
                "unsloth-single-gpu",
                SEVERITY_INFO,
                f"{hardware.gpu_count} GPU görünüyor; Unsloth eğitimi tek GPU'da çalışır. "
                "İkinci T4 bu betikte kullanılmaz.",
            )
        )
    if hardware.cuda_available and not hardware.bf16_supported:
        add(
            Finding(
                "unsloth-fp16-moe",
                SEVERITY_WARNING,
                "GPU bfloat16 desteklemiyor (T4). Unsloth gpt-oss'ta MoE katmanlarını "
                "float32'ye taşıyarak fp16 taşmasını engeller; yine de uzun koşularda "
                "kayıp kararlılığını izleyin.",
                "Unsloth: 'gpt-oss precision issues' belgeleri.",
            )
        )
    if not load_in_4bit and "bf16" not in model_name.lower():
        add(
            Finding(
                "unsloth-16bit-model-id",
                SEVERITY_WARNING,
                f"16-bit LoRA için Unsloth, MXFP4 taban yerine bf16 yayınını öneriyor "
                f"({SUGGESTED_BF16_MODEL}). Varsayılan kimlik doğrulanmalı.",
            )
        )

    # Recompute estimates with the Unsloth-specific reference number so the
    # report does not mix two different measurement methods.
    if report.estimate is not None:
        report.estimate.assumptions.append(
            f"Unsloth yolu için yayımlanmış rakam ≈{required:.0f} GB; "
            "yukarıdaki generic tahmin bu yol için geçerli değildir."
        )
    report.method = "unsloth-qlora" if load_in_4bit else "unsloth-lora"
    return report


def load_model_and_tokenizer(
    cfg: Any,
    *,
    max_seq_length: int,
) -> tuple[Any, Any]:
    """Load the base model + tokenizer through Unsloth.

    ``cfg`` is a :class:`gokturk.config.TrainingConfig`; ``cfg.model`` supplies
    the checkpoint id, dtype policy and 4-bit flag.
    """
    status = detect_unsloth()
    if not status.available:
        raise UnslothUnavailableError(
            "Unsloth kullanılamıyor: "
            + status.detail
            + ". Kaggle'da `pip install unsloth` gerekir; bu isteğe bağlı bir "
            "bağımlılıktır ve generic yol onsuz da çalışır (yalnızca daha çok VRAM ister)."
        )

    import unsloth  # noqa: F401  (registers patches on import)

    fast = getattr(unsloth, status.fast_model_symbol)
    if not hasattr(fast, "from_pretrained") or not hasattr(fast, "get_peft_model"):
        raise UnslothCompatibilityError(
            f"{status.fast_model_symbol} beklenen from_pretrained/get_peft_model "
            f"metotlarını sağlamıyor (sürüm {status.version}); kurulum sürümünü "
            "Unsloth belgeleriyle karşılaştırın."
        )

    dtype = cfg.model.dtype if cfg.model.dtype in ("fp16", "bf16", "fp32") else None
    kwargs = {
        "model_name": cfg.model.name_or_path,
        "max_seq_length": max_seq_length,
        "load_in_4bit": bool(cfg.model.load_in_4bit),
        "dtype": dtype,
    }
    model, tokenizer = fast.from_pretrained(**_filter_kwargs(fast.from_pretrained, kwargs))
    return model, tokenizer


def apply_lora(model: Any, cfg: Any, *, max_seq_length: int) -> Any:
    """Attach LoRA adapters with Unsloth's memory-optimised settings."""
    status = detect_unsloth()
    if not status.available:  # pragma: no cover - guarded by load_model_and_tokenizer
        raise UnslothUnavailableError("Unsloth kullanılmıyor")
    import unsloth

    fast = getattr(unsloth, status.fast_model_symbol)
    configured = list(cfg.lora.target_modules)
    targets = DEFAULT_TARGET_MODULES if configured == ["auto"] else configured
    kwargs = {
        "r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "bias": cfg.lora.bias,
        "target_modules": list(targets),
        "use_gradient_checkpointing": "unsloth",
        "random_state": cfg.optimization.seed,
        "max_seq_length": max_seq_length,
    }
    return fast.get_peft_model(model, **_filter_kwargs(fast.get_peft_model, kwargs))


def save_adapter(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Save adapters in the PEFT layout (``save_pretrained``)."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
