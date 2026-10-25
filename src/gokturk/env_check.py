"""Environment, hardware and compatibility checks for training.

Everything in this module is *diagnostic*. It answers three questions:

1. What hardware and packages do we actually have?
2. Does the configured base checkpoint fit (rough estimate, with assumptions)?
3. Is the requested training method (full / LoRA / QLoRA, dtype, quantization)
   actually supported for this architecture on this hardware?

The answer to (3) is derived from published documentation and from runtime
probing, never from assumptions. Estimates are labelled as estimates; blocking
problems are labelled as blocking. See ``docs/architecture.md`` for sources.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

GB = 1024**3

# ---------------------------------------------------------------------------
# Known model facts (from public model cards; verify before relying on them)
# ---------------------------------------------------------------------------
#: ``params`` = total parameters, ``native_format`` = how the released
#: checkpoint is stored, ``source`` = where the number comes from.
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b": {
        "params": 20.9e9,
        "active_params": 3.6e9,
        "architecture": "GptOssForCausalLM",
        "native_format": "mxfp4 (MoE experts) + bf16 (rest)",
        "gated": False,
        "license": "Apache-2.0",
        "source": "openai/gpt-oss-20b model card (21B total / 3.6B active, MXFP4)",
    },
    "openai/gpt-oss-120b": {
        "params": 116.8e9,
        "active_params": 5.1e9,
        "architecture": "GptOssForCausalLM",
        "native_format": "mxfp4 (MoE experts) + bf16 (rest)",
        "gated": False,
        "license": "Apache-2.0",
        "source": "openai/gpt-oss-120b model card",
    },
    # --- Qwen tabanları (v1 adayları) -------------------------------------
    # Kimlikler ve parametre sayıları Hugging Face model API'sinden okundu
    # (2026-10-25). Qwen3.5 ailesi görüntü+metin (image-text-to-text) mimarisi
    # kullanır ama metin SFT'si için AutoProcessor GEREKMEZ: transformers'ta
    # ``Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected`` hem ``^mtp.*``
    # hem ``^model.visual.*`` kalıbını taşır; görüntü kulesi ağırlıkları
    # yüklenirken yok sayılır. Yani ``AutoModelForCausalLM`` + ``AutoTokenizer``
    # metin yolu yeterlidir (kaynak: transformers
    # ``models/qwen3_5/modeling_qwen3_5.py``, 2026-10-25). Bu bir *kod okuma*
    # bulgusudur; gerçek checkpoint üzerinde doğrulaması Kaggle koşusunun ilk
    # adımıdır (yükleme logunda ``model.visual.*`` "unexpected key" görünür).
    "Qwen/Qwen3.5-9B": {
        "params": 9.0e9,
        "active_params": 9.0e9,
        "architecture": "Qwen3_5ForConditionalGeneration (görüntü+metin)",
        "native_format": "bf16",
        "gated": False,
        "license": "Apache-2.0",
        "source": "huggingface.co/api/models/Qwen/Qwen3.5-9B (2026-10-25)",
    },
    "Qwen/Qwen3.5-4B": {
        # safetensors toplamı: 4_659_865_088 (BF16); HF depolama ≈9,33 GB
        "params": 4.66e9,
        "active_params": 4.66e9,
        "architecture": "Qwen3_5ForConditionalGeneration (görüntü+metin)",
        "native_format": "bf16",
        "gated": False,
        "license": "Apache-2.0",
        "source": "huggingface.co/api/models/Qwen/Qwen3.5-4B (2026-10-25)",
    },
    "Qwen/Qwen3.5-27B": {
        "params": 27.0e9,
        "active_params": 27.0e9,
        "architecture": "Qwen3_5ForConditionalGeneration (görüntü+metin)",
        "native_format": "bf16",
        "gated": False,
        "license": "Apache-2.0",
        "source": "huggingface.co/api/models/Qwen/Qwen3.5-27B (2026-10-25)",
    },
    "Qwen/Qwen3-0.6B": {
        "params": 0.6e9,
        "active_params": 0.6e9,
        "architecture": "Qwen3ForCausalLM (metin)",
        "native_format": "bf16",
        "gated": False,
        "license": "Apache-2.0",
        "source": "huggingface.co/api/models/Qwen/Qwen3-0.6B (2026-10-25)",
    },
    "Qwen/Qwen3-14B": {
        "params": 14.8e9,
        "active_params": 14.8e9,
        "architecture": "Qwen3ForCausalLM (metin)",
        "native_format": "bf16",
        "gated": False,
        "license": "Apache-2.0",
        "source": "Qwen/Qwen3-14B model card (14.8B, 13.2B non-embedding)",
    },
}

#: Published minimum VRAM for gpt-oss fine-tuning (Unsloth documentation).
#: BF16 LoRA = 44 GB for 20B; their patched 4-bit QLoRA = ~14 GB but it is an
#: Unsloth-specific kernel path, not what transformers+bitsandbytes does here.
GPT_OSS_REFERENCE_VRAM_GB = {"lora_bf16": 44, "qlora_unsloth_4bit": 14}

def resolve_known_model(model_name: str) -> dict[str, Any] | None:
    """Match a checkpoint id against :data:`KNOWN_MODELS` (substring, case-free).

    Community mirrors such as ``unsloth/gpt-oss-20b`` are the same architecture
    and the same released weights, so planning numbers must not silently vanish
    just because the repository prefix differs.
    """
    key = model_name.strip().casefold()
    for known_name, facts in KNOWN_MODELS.items():
        short = known_name.split("/")[-1].casefold()
        if known_name.casefold() in key or short in key:
            return facts
    return None


#: GPU compute capabilities that have bfloat16 support.
BF16_MIN_COMPUTE_CAPABILITY = 8.0  # Ampere (A100/3090) and newer
MXFP4_MIN_COMPUTE_CAPABILITY = 9.0  # Hopper (H100) and newer


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------
@dataclass
class GpuInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float | None
    compute_capability: str | None

    @property
    def compute_capability_tuple(self) -> tuple[int, int] | None:
        if not self.compute_capability:
            return None
        try:
            major, minor = self.compute_capability.split(".")[:2]
            return int(major), int(minor)
        except ValueError:  # pragma: no cover
            return None


@dataclass
class HardwareInfo:
    python_version: str
    platform: str
    cpu_count: int
    ram_gb: float
    disk_free_gb: float
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_count: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)
    bf16_supported: bool = False
    fp16_supported: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        """Largest *single* GPU. Cross-GPU sharding is not automatic here."""
        return max((g.total_vram_gb for g in self.gpus), default=0.0)

    @property
    def combined_vram_gb(self) -> float:
        return sum(g.total_vram_gb for g in self.gpus)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def _mem_gb() -> float:
    try:
        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return round(pages / GB, 1)
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return 0.0


def detect_hardware(probe_path: str | None = None) -> HardwareInfo:
    info = HardwareInfo(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        cpu_count=os.cpu_count() or 0,
        ram_gb=_mem_gb(),
        disk_free_gb=round(
            shutil.disk_usage(probe_path or os.getcwd()).free / GB, 1
        ),
    )

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        info.notes.append(
            "PyTorch is not installed: no training can run (data tooling still works)."
        )
        return info

    import torch

    info.torch_available = True
    info.torch_version = torch.__version__
    info.cuda_available = bool(torch.cuda.is_available())
    if info.cuda_available:
        info.cuda_version = getattr(torch.version, "cuda", None)
        info.cudnn_version = (
            str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        )
        info.gpu_count = torch.cuda.device_count()
        for idx in range(info.gpu_count):
            props = torch.cuda.get_device_properties(idx)
            free_bytes = None
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except Exception:  # pragma: no cover - older drivers
                pass
            info.gpus.append(
                GpuInfo(
                    index=idx,
                    name=props.name,
                    total_vram_gb=round(props.total_memory / GB, 1),
                    free_vram_gb=round(free_bytes / GB, 1) if free_bytes else None,
                    compute_capability=f"{props.major}.{props.minor}",
                )
            )
        try:
            info.bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:  # pragma: no cover
            info.bf16_supported = False
        info.fp16_supported = all(
            (g.compute_capability_tuple or (0, 0)) >= (6, 0) for g in info.gpus
        )
        if info.gpu_count > 1:
            info.notes.append(
                f"{info.gpu_count} GPUs detected. This repository does not implement "
                "tensor-parallel or FSDP sharding: a single model must fit on one GPU "
                "unless you add an accelerate/FSDP configuration yourself."
            )
        if not info.bf16_supported:
            info.notes.append(
                "This GPU does not support bfloat16 (pre-Ampere, e.g. Tesla T4). "
                "Training will fall back to float16, which is numerically less stable "
                "for MoE models such as gpt-oss."
            )
    else:
        info.notes.append("CUDA is not available: only CPU planning/smoke runs are possible.")
    return info


# ---------------------------------------------------------------------------
# packages
# ---------------------------------------------------------------------------
@dataclass
class PackageInfo:
    name: str
    installed: bool
    version: str | None
    purpose: str


def detect_packages() -> list[PackageInfo]:
    specs = [
        ("torch", "tensors/GPU kernels; required for training"),
        ("transformers", "model loading, Trainer; >=4.55 for gpt-oss support"),
        ("peft", "LoRA/QLoRA adapters"),
        ("accelerate", "device placement, mixed precision"),
        ("bitsandbytes", "4-bit NF4 quantization (QLoRA); CUDA only"),
        ("datasets", "optional dataset utilities (not required by this repo)"),
        ("trl", "optional SFT trainer (not required by this repo)"),
        ("unsloth", "optional alternative 4-bit gpt-oss training path"),
        ("yaml", "configuration files"),
    ]
    out: list[PackageInfo] = []
    for name, purpose in specs:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            out.append(PackageInfo(name, True, str(version) if version else "unknown", purpose))
        except Exception:
            out.append(PackageInfo(name, False, None, purpose))
    return out


def package_versions() -> dict[str, str]:
    return {p.name: p.version or "missing" for p in detect_packages()}


# ---------------------------------------------------------------------------
# memory estimation
# ---------------------------------------------------------------------------
@dataclass
class MemoryEstimate:
    method: str
    dtype: str
    base_weights_gb: float
    optimizer_and_grads_gb: float
    activations_gb: float
    total_gb: float
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_memory(
    params: float,
    method: str,
    dtype: str,
    *,
    batch_size: int = 1,
    seq_len: int = 1024,
) -> MemoryEstimate:
    """Rough VRAM estimate in GB.

    Assumptions (documented, intentionally conservative):
    * bf16/fp16 weights: 2 bytes/param; fp32: 4 bytes/param.
    * QLoRA (NF4): 0.55 bytes/param for the frozen base + a small adapter
      (~1% of params in bf16) + activations.
    * LoRA on a bf16 base: 2 bytes/param + ~1% adapter in bf16.
    * Full fine-tuning: weights + grads (same dtype) + Adam moments
      (8 bytes/param) + fp32 master copy (4 bytes/param).
    * Activations: ~0.6 GB per 1B active parameters at seq_len=1024,
      batch=1, with gradient checkpointing enabled. This term is the least
      reliable part of the estimate; it scales with seq_len and batch size.
    """
    notes = [
        "Estimate only: real usage depends on sequence length, batch size, "
        "attention implementation, gradient checkpointing and allocator behaviour.",
    ]
    adapter_gb = params * 0.01 * 2 / GB
    act_gb = 0.6 * (params / 1e9) * (batch_size * seq_len / 1024) * 0.35

    if method == "qlora":
        base = params * 0.55 / GB
        opt = adapter_gb * 4  # adapter grads + Adam states (small)
        notes.append("QLoRA path assumes bitsandbytes NF4 with compute dtype fp16/bf16.")
        notes.append(
            "This does NOT model Unsloth's fused 4-bit kernels, which fit "
            f"gpt-oss-20b in ~{GPT_OSS_REFERENCE_VRAM_GB['qlora_unsloth_4bit']} GB."
        )
    elif method == "lora":
        base = params * (2 if dtype in ("bf16", "fp16") else 4) / GB
        opt = adapter_gb * 4
        notes.append("LoRA keeps the base weights in full precision in memory.")
    elif method == "full":
        bytes_per_param = 2 if dtype in ("bf16", "fp16") else 4
        base = params * bytes_per_param / GB
        opt = params * (2 * bytes_per_param + 12) / GB
        notes.append("Full fine-tuning is not viable at 20B on 16 GB cards.")
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"unknown method {method!r}")

    total = base + opt + act_gb
    return MemoryEstimate(
        method=method,
        dtype=dtype,
        base_weights_gb=round(base, 1),
        optimizer_and_grads_gb=round(opt, 1),
        activations_gb=round(act_gb, 1),
        total_gb=round(total, 1),
        assumptions=notes,
    )


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bf16_label(gpu: GpuInfo, hardware: HardwareInfo) -> str:
    """Human-readable bf16 support for one GPU (Ampere+ and runtime support)."""
    capability = gpu.compute_capability_tuple or (0, 0)
    upper = int(BF16_MIN_COMPUTE_CAPABILITY)
    supported = capability >= (upper, 0) and hardware.bf16_supported
    return "destekli" if supported else "desteklenmiyor"


@dataclass
class PreflightReport:
    hardware: HardwareInfo
    packages: list[PackageInfo]
    estimate: MemoryEstimate | None
    findings: list[Finding] = field(default_factory=list)
    model_name: str = ""
    method: str = ""
    dtype: str = ""
    backend: str = "transformers"

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_BLOCKING]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "method": self.method,
            "backend": self.backend,
            "dtype": self.dtype,
            "ok": self.ok,
            "hardware": self.hardware.to_dict(),
            "packages": [asdict(p) for p in self.packages],
            "estimate": self.estimate.to_dict() if self.estimate else None,
            "findings": [f.to_dict() for f in self.findings],
        }

    def render(self) -> str:
        lines: list[str] = []
        hw = self.hardware
        lines.append("=== Donanım ===")
        lines.append(f"Python {hw.python_version} | {hw.platform} | CPU {hw.cpu_count} çekirdek")
        lines.append(f"RAM {hw.ram_gb} GB | boş disk {hw.disk_free_gb} GB")
        if hw.cuda_available:
            lines.append(
                f"CUDA {hw.cuda_version} | torch {hw.torch_version} | {hw.gpu_count} GPU"
            )
            for g in hw.gpus:
                lines.append(
                    f"  [{g.index}] {g.name}: {g.total_vram_gb} GB, "
                    f"compute {g.compute_capability}, "
                    f"bf16={_bf16_label(g, hw)}"
                )
        else:
            lines.append(f"CUDA yok | torch {hw.torch_version or 'kurulu değil'}")

        lines.append("")
        lines.append("=== Paketler ===")
        for p in self.packages:
            lines.append(
                f"  {p.name:<14} {'✓ ' + str(p.version) if p.installed else '✗ yok':<22} {p.purpose}"
            )

        if self.estimate:
            e = self.estimate
            lines.append("")
            lines.append(f"=== Bellek tahmini ({self.method}, {self.dtype}) ===")
            lines.append(
                f"  temel ağırlıklar {e.base_weights_gb} GB + optimizer/grad "
                f"{e.optimizer_and_grads_gb} GB + aktivasyon {e.activations_gb} GB "
                f"≈ {e.total_gb} GB"
            )
            for a in e.assumptions:
                lines.append(f"  - {a}")

        lines.append("")
        lines.append("=== Bulgular ===")
        if not self.findings:
            lines.append("  (bulgu yok)")
        for f in self.findings:
            icon = {"blocking": "✗", "warning": "!", "info": "i"}[f.severity]
            lines.append(f"  [{icon}] {f.code}: {f.message}")
            if f.detail:
                for chunk in f.detail.splitlines():
                    lines.append(f"      {chunk}")
        lines.append("")
        lines.append("SONUÇ: " + ("EĞİTİM İÇİN UYGUN" if self.ok else "ENGELLEYİCİ SORUN VAR"))
        return "\n".join(lines)


def _resolve_architecture(model_name: str, trust_remote_code: bool = False) -> str | None:
    """Try to read the architecture from a local path or the HF hub (best effort)."""
    try:
        from transformers import AutoConfig
    except ModuleNotFoundError:  # pragma: no cover
        return None
    try:
        cfg = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=False
        )
        archs = getattr(cfg, "architectures", None) or []
        return str(archs[0]) if archs else None
    except Exception:
        return None


def check_environment(
    *,
    model_name: str,
    method: str,
    dtype: str,
    load_in_4bit: bool,
    params: float | None = None,
    seq_len: int = 1024,
    batch_size: int = 1,
    probe_path: str | None = None,
    check_hub: bool = False,
    trust_remote_code: bool = False,
    cpu_ok: bool = False,
    cpu_mini: bool = False,
    backend: str = "transformers",
    hardware: HardwareInfo | None = None,
) -> PreflightReport:
    """Collect hardware/package/model information and derive findings.

    ``hardware`` can be injected (tests, or callers that already probed the
    machine) instead of probing again.
    """
    hardware = hardware or detect_hardware(probe_path)
    packages = detect_packages()
    known = resolve_known_model(model_name)
    if params is None:
        params = known["params"] if known else None

    report = PreflightReport(
        hardware=hardware,
        packages=packages,
        estimate=None,
        model_name=model_name,
        method=method,
        dtype=dtype,
    )
    if backend not in ("transformers", "unsloth"):
        raise ValueError(f"unknown backend {backend!r}; use 'transformers' or 'unsloth'")
    report.backend = backend
    add = report.findings.append

    installed = {p.name: p for p in packages}
    if not hardware.torch_available:
        add(Finding("torch-missing", SEVERITY_BLOCKING, "PyTorch kurulu değil; eğitim başlatılamaz."))
    if not installed.get("transformers", PackageInfo("", False, None, "")).installed:
        add(Finding("transformers-missing", SEVERITY_BLOCKING, "transformers kurulu değil."))
    if method in ("lora", "qlora") and not installed.get("peft", PackageInfo("", False, None, "")).installed:
        add(Finding("peft-missing", SEVERITY_BLOCKING, "peft kurulu değil; LoRA/QLoRA uygulanamaz."))
    if method == "qlora" and backend == "transformers":
        if not installed.get("bitsandbytes", PackageInfo("", False, None, "")).installed:
            add(
                Finding(
                    "bnb-missing",
                    SEVERITY_BLOCKING,
                    "QLoRA için bitsandbytes gerekli (yalnızca CUDA üzerinde çalışır).",
                    "pip install bitsandbytes",
                )
            )
        if not hardware.cuda_available:
            add(
                Finding(
                    "qlora-no-cuda",
                    SEVERITY_BLOCKING,
                    "QLoRA 4-bit eğitim CUDA olmadan çalışmaz.",
                )
            )
    if backend == "unsloth":
        add(
            Finding(
                "unsloth-backend",
                SEVERITY_INFO,
                "Eğitim arka ucu: Unsloth 4-bit QLoRA (bitsandbytes yolu kullanılmıyor).",
            )
        )

    if not hardware.cuda_available:
        if cpu_mini:
            add(
                Finding(
                    "cpu-mini",
                    SEVERITY_INFO,
                    "GPU yok; CPU mini koşusu bilinçli olarak onaylandı (--cpu-ok). "
                    "Bu, ≤1B sınıfı bir taban modelde kısa bir LoRA koşusudur: mekaniği "
                    "kanıtlar, hız ve kalite açısından üretim koşusu değildir.",
                    "GitHub Actions ubuntu-latest: 4 vCPU / 16 GB RAM / 14 GB disk, GPU yok.",
                )
            )
            if params and params > 1.5e9:
                add(
                    Finding(
                        "cpu-mini-too-big",
                        SEVERITY_WARNING,
                        f"CPU mini koşusu için model çok büyük (≈{params / 1e9:.1f}B); "
                        "4 vCPU'da adım süresi dakikalara çıkar.",
                        "≤1B bir taban model kullanın (ör. Qwen/Qwen3-0.6B) veya GPU'lu ortama geçin.",
                    )
                )
        elif cpu_ok:
            add(
                Finding(
                    "cpu-smoke",
                    SEVERITY_INFO,
                    "GPU yok; CPU üzerinde yalnızca küçük rastgele modelle duman testi "
                    "yapılabilir. Bu mod gerçek bir eğitim değildir.",
                )
            )
        else:
            add(
                Finding(
                    "no-gpu",
                    SEVERITY_BLOCKING,
                    "GPU bulunamadı: büyük model eğitimi bu ortamda yapılamaz.",
                    "CPU üzerinde yalnızca küçük rastgele modelle duman testi (--smoke) çalıştırılabilir.",
                )
            )

    # dtype vs hardware
    effective_dtype = dtype
    if dtype == "auto":
        if not hardware.cuda_available:
            effective_dtype = "fp32"  # CPU-only: bf16/fp16 bring no benefit here
        else:
            effective_dtype = "bf16" if hardware.bf16_supported else "fp16"
    if effective_dtype == "bf16" and hardware.cuda_available and not hardware.bf16_supported:
        add(
            Finding(
                "bf16-unsupported",
                SEVERITY_BLOCKING,
                "Bu GPU bfloat16 desteklemiyor; bf16 eğitim isteği geçersiz.",
            )
        )
    if effective_dtype == "fp16" and hardware.cuda_available and not hardware.bf16_supported:
        add(
            Finding(
                "fp16-fallback",
                SEVERITY_WARNING,
                "bfloat16 yok; float16 kullanılacak. MoE modellerde fp16 eğitim "
                "kayıp ıraksamasına açıktır (bkz. docs/architecture.md).",
            )
        )
    report.dtype = effective_dtype

    # model size estimate vs available memory
    if params and params > 0:
        est = estimate_memory(params, method, effective_dtype, batch_size=batch_size, seq_len=seq_len)
        report.estimate = est
        if hardware.cuda_available:
            single = hardware.total_vram_gb
            if est.total_gb > single:
                add(
                    Finding(
                        "vram-insufficient",
                        SEVERITY_BLOCKING,
                        f"Tahmini gereksinim ≈{est.total_gb} GB, tek GPU'da {single} GB var "
                        f"(toplam {hardware.combined_vram_gb} GB olsa da model tek GPU'ya sığmalı).",
                        "Bu repo otomatik model paralelliği uygulamıyor. Daha büyük GPU, "
                        "daha küçük model veya Unsloth benzeri özel 4-bit çekirdekler gerekir.",
                    )
                )
            elif est.total_gb > single * 0.85:
                add(
                    Finding(
                        "vram-tight",
                        SEVERITY_WARNING,
                        f"Tahmini gereksinim ≈{est.total_gb} GB, GPU {single} GB: sınırda.",
                        "max_seq_len / batch size düşürün veya gradient checkpointing açık tutun.",
                    )
                )

        # indirme boyutu vs boş disk: bf16 ağırlıklar ≈ 2 bayt/parametre.
        # VRAM hesabı tek başına yetmiyor: Kaggle'da 20 GB disk, 14B'nin 29,6 GB
        # bf16 indirmesini zaten alamaz (2026-10-25'te fark edilen tuzak).
        if hardware.disk_free_gb:
            download_gb = params * 2 / 1e9
            if download_gb > hardware.disk_free_gb:
                add(
                    Finding(
                        "weights-disk-insufficient",
                        SEVERITY_BLOCKING,
                        f"Taban ağırlıkları bf16 ≈{download_gb:.1f} GB, boş disk "
                        f"{hardware.disk_free_gb:.1f} GB: indirme tamamlanamaz.",
                        "Daha küçük taban seçin, hazır 4-bit bir depo kullanın veya "
                        "HF_HOME'u daha büyük bir diske taşıyın. Bu denetim yalnızca "
                        "çalışma dizininin diskini ölçer.",
                    )
                )
            elif download_gb > hardware.disk_free_gb * 0.8:
                add(
                    Finding(
                        "weights-disk-tight",
                        SEVERITY_WARNING,
                        f"Taban ağırlıkları bf16 ≈{download_gb:.1f} GB, boş disk "
                        f"{hardware.disk_free_gb:.1f} GB: sınırda (adaptör, log ve "
                        "llama.cpp kopyası için pay kalmıyor).",
                        "max_seq_len'i düşürün, ara çıktıları silin veya daha küçük tabana geçin.",
                    )
                )

    # architecture-specific rules
    if "gpt-oss" in model_name.lower():
        add(
            Finding(
                "gptoss-native-mxfp4",
                SEVERITY_INFO,
                "gpt-oss ağırlıkları MXFP4 (MoE) olarak yayınlanmıştır; yerel MXFP4 ile "
                "eğitim ana akım kütüphanelerde desteklenmez, eğitim için bf16/fp16'ya "
                "yükseltme gerekir.",
                "Kaynak: OpenAI gpt-oss README; NVIDIA Model Optimizer gpt-oss notları.",
            )
        )
        if method == "full":
            add(
                Finding(
                    "gptoss-full-unsupported",
                    SEVERITY_BLOCKING,
                    "gpt-oss-20b tam parametreli eğitim 2×T4 (32 GB) üzerinde mümkün değil; "
                    "bf16 ağırlıklar tek başına ≈42 GB tutar.",
                )
            )
        if method == "lora" and backend == "transformers":
            add(
                Finding(
                    "gptoss-lora-bf16-vram",
                    SEVERITY_BLOCKING,
                    "gpt-oss-20b bf16 LoRA için yayımlanmış gereksinim ≈44 GB VRAM "
                    "(Unsloth belgeleri); mevcut tek GPU kapasitesi yetmiyor."
                    if hardware.total_vram_gb < GPT_OSS_REFERENCE_VRAM_GB["lora_bf16"]
                    else "gpt-oss-20b bf16 LoRA için ≈44 GB VRAM gerekir; mevcut GPU sınırda.",
                    "Bf16 yükseltme olmadan LoRA uygulanamaz.",
                )
            )
        if method == "qlora" and backend == "transformers":
            single = hardware.total_vram_gb
            if single < 24:
                add(
                    Finding(
                        "gptoss-qlora-generic-too-small",
                        SEVERITY_BLOCKING,
                        f"Bu repodaki transformers+bitsandbytes QLoRA yolu gpt-oss-20b'yi "
                        f"{single} GB VRAM'e sığdıramaz; yayımlanmış ~14 GB rakamı "
                        "Unsloth'un özel 4-bit çekirdeklerine aittir.",
                        "Seçenekler: (a) `--backend unsloth` ile Unsloth 4-bit yolu "
                        "(bu repoda desteklenir), (b) daha büyük GPU (>=48 GB) ve bf16 LoRA, "
                        "(c) 7-8B sınıfı bir taban modelle altyapıyı doğrulayıp ölçek büyütme.",
                    )
                )
            if not (hardware.bf16_supported):
                add(
                    Finding(
                        "gptoss-fp16-instability",
                        SEVERITY_WARNING,
                        "T4 üzerinde bf16 yok: gpt-oss için fp16 eğitim tarihsel olarak "
                        "kayıp ıraksamasına yol açtı (Unsloth hata notları).",
                    )
                )
        if hardware.gpu_count and any(
            (g.compute_capability_tuple or (0, 0)) < (MXFP4_MIN_COMPUTE_CAPABILITY, 0)
            for g in hardware.gpus
        ):
            add(
                Finding(
                    "mxfp4-kernel-unsupported",
                    SEVERITY_INFO,
                    "Yerel MXFP4 çekirdekleri compute capability >= 9.0 ister; bu GPU'da "
                    "model yüklenirken bf16/fp16'ya çözülür (VRAM ~4x artar).",
                    "transformers, MXFP4 ağırlıkları desteklenmeyen GPU'da dequantize eder.",
                )
            )

    if check_hub:
        arch = _resolve_architecture(model_name, trust_remote_code)
        if arch:
            add(Finding("arch", SEVERITY_INFO, f"Model mimarisi: {arch}"))
            try:
                import transformers

                if not hasattr(transformers, arch):
                    add(
                        Finding(
                            "arch-missing",
                            SEVERITY_BLOCKING,
                            f"transformers sürümü {arch} sınıfını içermiyor; paketi yükseltin.",
                        )
                    )
            except Exception:  # pragma: no cover
                pass
        else:
            add(
                Finding(
                    "hub-unreachable",
                    SEVERITY_WARNING,
                    f"'{model_name}' için model kartı okunamadı (ağ kapalı, kimlik doğrulama "
                    "veya model yolu hatalı olabilir).",
                )
            )

    if known and known.get("license"):
        add(
            Finding(
                "license",
                SEVERITY_INFO,
                f"Temel model lisansı: {known['license']} ({known['source']}). "
                "Bu repo kendi lisansını (Apache-2.0) korur; model ağırlıklarının lisansı ayrıdır.",
            )
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m gokturk.env_check``."""
    import argparse

    parser = argparse.ArgumentParser(description="Eğitim ortamı denetimi")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--method", default="qlora", choices=["full", "lora", "qlora"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--params", type=float, default=None, help="Toplam parametre sayısı")
    parser.add_argument("--check-hub", action="store_true", help="Model kartını çözmeyi dene")
    parser.add_argument("--json", action="store_true", help="JSON çıktısı")
    args = parser.parse_args(argv)

    report = check_environment(
        model_name=args.model,
        method=args.method,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        params=args.params,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        check_hub=args.check_hub,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.render())
    return 0 if report.ok else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
