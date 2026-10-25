"""Publish a trained adapter, and plan a GGUF build -- honestly.

Two separate jobs live here:

1. **Hub push.** Upload a LoRA adapter directory to the Hugging Face Hub,
   together with a generated model card. The token is read *only* from the
   ``HF_TOKEN`` environment variable (or Kaggle Secrets feeding it); it is
   never written to disk, never placed in the card, and never logged.

2. **GGUF plan.** Decide whether a ``.gguf`` build is physically possible on
   the machine you are on, and print the exact commands if it is. This module
   does **not** run llama.cpp: building it needs ``cmake``, disk and hours, so
   the plan is emitted for the operator (or a later cell) to execute.

Why a plan and not a one-liner
------------------------------
Merging a LoRA into ``gpt-oss-20b`` materialises the full model in bf16
(~42 GB) before any conversion. Kaggle's free tier has ~29 GB RAM and a 20 GB
``/kaggle/working`` quota, so a GGUF build **cannot** happen there. Saying
"just run convert_hf_to_gguf.py" would be a lie by omission; the planner
refuses with numbers instead.

Accuracy note (important)
-------------------------
``gpt-oss`` ships natively in MXFP4 for its MoE expert weights, which are
>90% of the parameters. A Q4_K_M build therefore re-quantises weights that
were already 4-bit -- it is supported (community builds exist, ~11 GB), but it
adds rounding on top of the original rounding. For this model family MXFP4 is
both smaller and more faithful. The planner documents both and defaults to
``mxfp4``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Environment variable that may hold a Hugging Face write token.
HF_TOKEN_ENV = "HF_TOKEN"

#: Sizes are estimates in gigabytes (10**9 bytes) and are labelled as such
#: everywhere they surface. They exist to answer "will this fit?", not to
#: predict exact file sizes.
GB = 1_000_000_000

#: Bytes per parameter when a checkpoint is materialised in a given dtype.
DTYPE_BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "int8": 1.0, "mxfp4": 0.53}

#: Approximate parameter counts of the supported bases.
BASE_PARAMS_B = {
    "openai/gpt-oss-20b": 20.9,
    "unsloth/gpt-oss-20b": 20.9,
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit": 20.9,
    "openai/gpt-oss-120b": 116.8,
    # Qwen tabanları (v1 adayları); kimlikler HF model API'sinden doğrulandı.
    "Qwen/Qwen3.5-9B": 9.0,
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen3.5-27B": 27.0,
    "Qwen/Qwen3-14B": 14.8,
}

#: Effective memory bandwidth assumed per vCPU when running a quantised model
#: on CPU. A cloud vCPU is not a desktop core: sub-10 GB/s is normal, and the
#: range below is deliberately wide so the report shows an interval instead of
#: a single confident number. Token generation on CPU is bandwidth-bound, so
#: tokens/s ≈ bandwidth / model_bytes.
CPU_BANDWIDTH_GBPS_RANGE = (4.0, 10.0)

#: Quantisation targets understood by the planner.
#: ``bits`` is the effective packed size used for the size estimate.
QUANTS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "bits": 4.25,
        "note": (
            "gpt-oss'un doğal biçimi; ≈12 GB. llama.cpp'nin gpt-oss için önerdiği hedef "
            "(llama-quantize türü MXFP4_MOE). Bu model ailesinde uzman katmanları "
            "MXFP4 dışına taşımak kaliteyi belirgin biçimde düşürür."
        ),
        "llama_quantize_type": "MXFP4_MOE",
    },
    "Q4_K_M": {
        "bits": 4.89,
        "note": (
            "En yaygın varsayılan; ≈11 GB. gpt-oss'ta desteklenir ama önerilmez: "
            "uzman (FFN) katmanları MXFP4 dışına çıkar ve bu katmanlar başka "
            "quant'larda iyi davranmıyor. Topluluk quant'ları Q4_K_M etiketini "
            "kullanırken FFN'i MXFP4'te tutar; düz `llama-quantize Q4_K_M` bunu "
            "yapmaz."
        ),
        "llama_quantize_type": "Q4_K_M",
    },
    "Q5_K_M": {"bits": 5.65, "note": "Daha büyük, ≈13 GB.", "llama_quantize_type": "Q5_K_M"},
    "Q8_0": {"bits": 8.5, "note": "≈22 GB; bu modelde gereksiz.", "llama_quantize_type": "Q8_0"},
}

DEFAULT_QUANT = "mxfp4"

#: Fraction of gpt-oss parameters that live in the MoE expert (FFN) tensors.
#: Published builds keep these at MXFP4 for *any* target quant, which is why a
#: "Q4_K_M" gpt-oss file (≈11.7 GB) is not smaller than the MXFP4 one (≈12.1 GB).
#: The share is an approximation for the size model below, not a measurement.
MOE_EXPERT_SHARE = 0.93
MXFP4_BITS = 4.25

#: llama.cpp repository used in the plan (public, pinned by branch name only).
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


class PublishError(RuntimeError):
    """Raised when publishing or planning cannot proceed."""


# ---------------------------------------------------------------------------
# model card
# ---------------------------------------------------------------------------
_ADAPTER_LABELS = {"qlora": "QLoRA", "lora": "LoRA", "full": "tam fine-tune"}


def _fmt(value: Any, default: str = "bilinmiyor") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_model_card(
    manifest: dict[str, Any] | None,
    *,
    repo_id: str | None = None,
    base_model: str | None = None,
    dataset_name: str | None = None,
) -> str:
    """Render a Turkish model card that makes no unmeasured claim.

    The card is generated from ``run_manifest.json`` when available, so the
    numbers in it are the numbers the run actually produced. Anything unknown
    is written as *bilinmiyor* rather than guessed.
    """
    manifest = manifest or {}
    config = manifest.get("config") or {}
    model_cfg = config.get("model") or {}
    metrics = manifest.get("metrics") or {}
    resolved_base = base_model or model_cfg.get("name_or_path") or "openai/gpt-oss-20b"
    adapter_type = (config.get("optimization") or {}).get("method", "qlora")

    train_loss = metrics.get("train_loss")
    if train_loss is None:
        train_loss = metrics.get("loss")
    eval_loss = metrics.get("eval_loss")
    steps = metrics.get("global_step") or metrics.get("max_steps")
    examples = (manifest.get("data") or {}).get("train_examples")

    title = repo_id or "GökTürk-Ultra (LoRA adaptörü)"
    lines = [
        f"# {title}",
        "",
        "Bu depo, **GökTürk-Ultra** projesinin Türkçe talimat/araç kullanımı için",
        f"eğittiği bir **{_ADAPTER_LABELS.get(str(adapter_type).lower(), _fmt(adapter_type))}"
        " adaptörüdür**. Temel model",
        f"[`{resolved_base}`](https://huggingface.co/{resolved_base}) üzerine",
        "LoRA olarak uygulanır; tek başına çalışmaz.",
        "",
        "## Ne var, ne yok",
        "",
        "| | |",
        "| --- | --- |",
        f"| Temel model | `{resolved_base}` |",
        f"| Adaptör türü | {_fmt(adapter_type)} |",
        f"| Eğitim örneği | {_fmt(examples)} |",
        f"| Adım | {_fmt(steps)} |",
        f"| Eğitim kaybı | {_fmt(train_loss)} |",
        f"| Değerlendirme kaybı | {_fmt(eval_loss, 'ölçülmedi')} |",
        f"| Eğitim arka ucu | {_fmt(manifest.get('backend'))} |",
        f"| Veri seti | {_fmt(dataset_name, 'sentetik, öğretmen modellerle üretildi')} |",
        "",
        "## Dürüst sınırlar",
        "",
        "* **Ölçülmüş bir benchmark sonucu yoktur.** Yukarıdaki kayıp değerleri",
        "  eğitimin çalıştığını gösterir, modelin iyi Türkçe konuştuğunu değil.",
        "* Değerlendirme (`gokturk.evaluate` + insan incelemesi) yapılmadan bu",
        "  adaptör üretim için önerilmez.",
        "* Eğitim verisi sentetiktir (öğretmen modeller tarafından üretilmiş,",
        "  şema ve kalite kurallarından geçmiştir); insan yazımı bir set değildir.",
        "* Temel modelin kendi lisansı ve kullanım politikası geçerlidir.",
        "",
        "## Kullanım",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'tok = AutoTokenizer.from_pretrained("{resolved_base}")',
        f'model = AutoModelForCausalLM.from_pretrained("{resolved_base}")',
        f'model = PeftModel.from_pretrained(model, "{repo_id or "<bu-depo>"}")',
        "```",
        "",
        "Sistem mesajı eğitimde şu cümleyle verildi:",
        "",
        "> Sen GökTürk-Ultra'sın: Türkçe konuşan, talimatları izleyen ve",
        "> gerektiğinde araç çağıran bir yapay zekâ asistanısın.",
        "",
        "Bu cümle eğitim hedefi değil **bağlamdır** (`-100` ile maskelenir); model",
        "onu taklit etmeyi öğrenmez. Kimlik davranışı bu yüzden garanti değildir.",
        "",
        "## GGUF",
        "",
        "Bu depo **GGUF içermez**. gpt-oss-20b'nin GGUF dönüşümü LoRA birleştirmesi",
        "için ≈42 GB bellek ister; Kaggle ücretsiz katmanında (29 GB RAM) mümkün",
        "değildir. Komutlar ve boyut hesabı için:",
        "",
        "```bash",
        "python -m gokturk.publish --gguf-plan --adapter <adaptor-dizini>",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# hub push
# ---------------------------------------------------------------------------
@dataclass
class PushPlan:
    """What a push would do -- computed without any network access."""

    adapter_dir: str
    repo_id: str
    private: bool
    files: list[str] = field(default_factory=list)
    total_bytes: int = 0
    token_present: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_dir": self.adapter_dir,
            "repo_id": self.repo_id,
            "private": self.private,
            "files": self.files,
            "total_mb": round(self.total_bytes / 1_000_000, 2),
            # Only presence is reported; the token itself never leaves the env.
            "token_present": self.token_present,
            "problems": self.problems,
        }


_ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors")


def plan_push(adapter_dir: str | os.PathLike[str], repo_id: str, *, private: bool = True) -> PushPlan:
    """Validate the adapter directory and the repo id without touching the network."""
    path = Path(adapter_dir)
    plan = PushPlan(
        adapter_dir=str(path),
        repo_id=repo_id,
        private=private,
        token_present=bool(os.environ.get(HF_TOKEN_ENV, "").strip()),
    )
    if not path.is_dir():
        plan.problems.append(f"adaptör dizini yok: {path}")
        return plan
    files = sorted(p for p in path.rglob("*") if p.is_file())
    plan.files = [str(p.relative_to(path)) for p in files]
    plan.total_bytes = sum(p.stat().st_size for p in files)
    if not any(m in plan.files for m in _ADAPTER_MARKERS):
        plan.problems.append(
            "dizin bir PEFT adaptörü gibi görünmüyor (adapter_config.json / "
            "adapter_model.safetensors yok)"
        )
    if not plan.files:
        plan.problems.append("dizin boş")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        plan.problems.append(f"repo_id 'kullanıcı/depo' biçiminde olmalı, alınan: {repo_id!r}")
    if not plan.token_present:
        plan.problems.append(
            f"{HF_TOKEN_ENV} tanımlı değil; yazma izinli bir token olmadan push yapılamaz "
            "(token yalnızca ortam değişkeninden okunur)"
        )
    return plan


def _import_hub():
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - lazy: optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PublishError(
            "huggingface_hub kurulu değil. Kurulum: pip install 'gokturk-ultra[hub]'"
        ) from exc
    return HfApi


def push_adapter(
    adapter_dir: str | os.PathLike[str],
    repo_id: str,
    *,
    private: bool = True,
    manifest: dict[str, Any] | None = None,
    base_model: str | None = None,
    dataset_name: str | None = None,
    commit_message: str = "GökTürk-Ultra adaptörü",
    api: Any = None,
) -> dict[str, Any]:
    """Upload an adapter directory plus a generated model card.

    ``api`` exists so tests can inject a fake ``HfApi``-like object; production
    calls leave it ``None`` and the real client is imported lazily.

    Raises :class:`PublishError` when the plan has problems, so a bad repo id or
    a missing token fails *before* any network call.
    """
    plan = plan_push(adapter_dir, repo_id, private=private)
    if not plan.ok:
        raise PublishError("; ".join(plan.problems))

    token = os.environ.get(HF_TOKEN_ENV, "").strip()
    if not token:
        raise PublishError(f"{HF_TOKEN_ENV} boş; push yapılmadı")

    client = api if api is not None else _import_hub()(token=token)

    card_path = Path(adapter_dir) / "README.md"
    card = build_model_card(
        manifest,
        repo_id=repo_id,
        base_model=base_model,
        dataset_name=dataset_name,
    )
    card_path.write_text(card, encoding="utf-8")

    # A model repo is created (not a dataset repo) so `from_pretrained` works.
    client.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    client.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    return {"ok": True, "repo_id": repo_id, "url": url, "private": private, "files": plan.files}


# ---------------------------------------------------------------------------
# GGUF planning
# ---------------------------------------------------------------------------
def estimate_sizes(
    params_b: float,
    *,
    quant: str = DEFAULT_QUANT,
    merge_dtype: str = "bfloat16",
    moe_expert_share: float = 0.0,
) -> dict[str, float]:
    """Return the three sizes a GGUF build needs, in GB.

    * ``merged_gb`` -- the LoRA-merged checkpoint written by transformers.
    * ``gguf_unquantized_gb`` -- intermediate produced by ``convert_hf_to_gguf.py``
      (bf16 for a merged checkpoint: the MXFP4 experts are already float by then,
      so the intermediate is full-size, ≈42 GB for 20B).
    * ``final_gb`` -- the quantised file you actually ship.
    """
    if quant not in QUANTS:
        raise PublishError(f"bilinmeyen quant {quant!r}; desteklenen: {sorted(QUANTS)}")
    if merge_dtype not in DTYPE_BYTES:
        raise PublishError(f"bilinmeyen dtype {merge_dtype!r}; desteklenen: {sorted(DTYPE_BYTES)}")
    if not 0.0 <= moe_expert_share <= 1.0:
        raise PublishError(f"moe_expert_share must be in [0, 1], got {moe_expert_share}")
    params = params_b * 1e9
    # For gpt-oss the experts stay at MXFP4 whatever the target quant is, so the
    # final size is dominated by the expert share. Dense models (Qwen) have no
    # such share and are sized directly by the target quant.
    expert_bits = MXFP4_BITS * moe_expert_share
    other_bits = QUANTS[quant]["bits"] * (1.0 - moe_expert_share)
    return {
        "merged_gb": round(params * DTYPE_BYTES[merge_dtype] / GB, 1),
        "gguf_unquantized_gb": round(params * DTYPE_BYTES["float16"] / GB, 1),
        "final_gb": round(params * (expert_bits + other_bits) / 8 / GB, 1),
    }


def plan_gguf(
    adapter_dir: str | os.PathLike[str],
    *,
    base_model: str = "openai/gpt-oss-20b",
    quant: str = DEFAULT_QUANT,
    merge_dtype: str = "bfloat16",
    out_dir: str = "outputs/gguf",
    available_ram_gb: float | None = None,
    available_disk_gb: float | None = None,
    params_b: float | None = None,
) -> dict[str, Any]:
    """Decide whether a GGUF build fits, and emit the commands if it does.

    ``available_ram_gb`` / ``available_disk_gb`` default to the real values of
    the current machine (psutil if present, else ``/proc/meminfo`` and
    ``shutil.disk_usage``), so running this on Kaggle answers the question with
    Kaggle's numbers rather than with optimism.
    """
    if params_b is None:
        params_b = BASE_PARAMS_B.get(base_model, BASE_PARAMS_B["openai/gpt-oss-20b"])
    # Only gpt-oss ships MXFP4 experts; every other family is dense at the
    # target quant, so the expert correction must not leak into them.
    expert_share = MOE_EXPERT_SHARE if "gpt-oss" in base_model else 0.0
    sizes = estimate_sizes(
        params_b, quant=quant, merge_dtype=merge_dtype, moe_expert_share=expert_share
    )

    ram = available_ram_gb if available_ram_gb is not None else detect_ram_gb()
    disk = available_disk_gb if available_disk_gb is not None else detect_disk_gb(Path(out_dir))

    # Merge peak: the model twice-ish (source shards + merged copy) plus copy
    # buffers. 1.6x is a deliberately conservative factor, and it is stated as
    # an estimate wherever it appears.
    needed_ram = round(sizes["merged_gb"] * 1.6, 1)
    needed_disk = round(
        sizes["merged_gb"] + sizes["gguf_unquantized_gb"] + sizes["final_gb"] + 5.0, 1
    )

    problems: list[str] = []
    if ram is not None and ram < needed_ram:
        problems.append(
            f"RAM yetersiz: birleştirme ≈{needed_ram} GB ister, makinede ≈{ram:.1f} GB var. "
            "LoRA'yı tam hassasiyette birleştirmek modelin tamamını belleğe alır."
        )
    if disk is not None and disk < needed_disk:
        problems.append(
            f"Disk yetersiz: ≈{needed_disk} GB gerekir, boş ≈{disk:.1f} GB."
        )

    adapter = Path(adapter_dir)
    if not adapter.is_dir():
        problems.append(f"adaptör dizini yok: {adapter}")

    out = Path(out_dir)
    merged = out / "merged"
    # llama.cpp's converter has no --outtype mxfp4: it writes bf16/f16/f32 and
    # the 4-bit target is reached with llama-quantize afterwards.
    intermediate = out / "model-bf16.gguf"
    final = out / f"model-{quant}.gguf"

    steps = [
        {
            "step": 1,
            "what": "LoRA'yı temel modele birleştir (tam hassasiyet)",
            "command": [
                sys.executable,
                "-m",
                "gokturk.publish",
                "--merge",
                "--adapter",
                str(adapter),
                "--base-model",
                base_model,
                "--merge-out",
                str(merged),
                "--merge-dtype",
                merge_dtype,
            ],
        },
        {
            "step": 2,
            "what": "llama.cpp'yi derle (cmake gerekir)",
            "command": [
                "bash",
                "-c",
                f"git clone --depth 1 {LLAMA_CPP_REPO} llama.cpp && "
                "cd llama.cpp && cmake -B build && cmake --build build --config Release -j",
            ],
        },
        {
            "step": 3,
            "what": "HF checkpoint -> GGUF (bf16 ara dosya; kaynak dtype'ı korur)",
            "command": [
                "python",
                "llama.cpp/convert_hf_to_gguf.py",
                str(merged),
                "--outfile",
                str(intermediate),
                "--outtype",
                "bf16",
            ],
        },
        {
            "step": 4,
            "what": f"quantise -> {quant}",
            "command": [
                "llama.cpp/build/bin/llama-quantize",
                str(intermediate),
                str(final),
                QUANTS[quant]["llama_quantize_type"],
            ],
        },
        {
            "step": 5,
            "what": "ara dosyayı sil (≈42 GB)",
            "command": ["rm", "-f", str(intermediate)],
        },
    ]

    return {
        "feasible": not problems,
        "problems": problems,
        "base_model": base_model,
        "params_b": params_b,
        "quant": quant,
        "quant_note": QUANTS[quant]["note"],
        "merge_dtype": merge_dtype,
        "sizes_gb": sizes,
        "needed_ram_gb": needed_ram,
        "needed_disk_gb": needed_disk,
        "available_ram_gb": ram,
        "available_disk_gb": disk,
        "steps": steps,
        "caveat": (
            "RAM ve disk değerleri tahmindir (≈). Dönüşüm bu modülde çalıştırılmaz; "
            "komutlar sizin makinanizde/notebook'ta çalıştırılır."
        ),
    }


# ---------------------------------------------------------------------------
# CPU serving feasibility (Hugging Face Spaces CPU Basic, small VMs)
# ---------------------------------------------------------------------------
#: KV-cache bytes per token per *million* parameters, approximate. Derived from
#: the candidate families' GQA geometry: 40 layers, 8 KV heads, head_dim 128,
#: fp16 → 2 · 40 · 8 · 128 · 2 B ≈ 164 kB/token for a 14.8B model, i.e. ≈11 B per
#: (token · million params). Rounded up to 12; real values move with layer count,
#: GQA ratio, dtype and context.
KV_BYTES_PER_TOKEN_PER_M_PARAM = 12.0


def plan_cpu_serving(
    params_b: float,
    *,
    quant: str = "Q4_K_M",
    context_tokens: int = 4096,
    ram_gb: float | None = 16.0,
    vcpus: int = 2,
    answer_tokens: int = 300,
    moe_expert_share: float = 0.0,
) -> dict[str, Any]:
    """Answer "can this model be served on a free CPU box?" with numbers.

    Models the three things that decide the answer:

    * **weights** -- parameters x bits/8 (same model as :func:`estimate_sizes`);
    * **KV cache** -- grows with context; this is what kills long chats on 16 GB;
    * **speed** -- CPU generation is memory-bandwidth bound, so tokens/s is
      approximated as bandwidth / weights_bytes and reported as a *range*.

    The verdict is deliberately blunt: sub-2 tokens/s is not a chat experience,
    it is a demo that takes minutes per answer.
    """
    if quant not in QUANTS:
        raise PublishError(f"bilinmeyen quant {quant!r}; desteklenen: {sorted(QUANTS)}")
    sizes = estimate_sizes(params_b, quant=quant, moe_expert_share=moe_expert_share)
    weights = sizes["final_gb"]
    kv = round(
        params_b * 1000 * KV_BYTES_PER_TOKEN_PER_M_PARAM * context_tokens / GB, 2
    )
    total = round(weights + kv, 2)

    low_bw, high_bw = CPU_BANDWIDTH_GBPS_RANGE
    slow = round(low_bw * max(vcpus, 1) / weights, 2) if weights else 0.0
    fast = round(high_bw * max(vcpus, 1) / weights, 2) if weights else 0.0
    # Optimistic end of the range is quoted for the "how long do I wait" answer.
    seconds = round(answer_tokens / fast, 1) if fast else None

    problems: list[str] = []
    if ram_gb is not None and total > ram_gb * 0.9:
        problems.append(
            f"bellek yetersiz: ağırlık {weights} GB + KV {kv} GB = {total} GB, "
            f"kullanılabilir ≈{ram_gb} GB (×0.9 emniyet payı)"
        )
    # The verdict uses the *pessimistic* end of the bandwidth range for the
    # "unusable" call and the optimistic end for "comfortable": a model that is
    # only fast when bandwidth is best is not comfortable.
    if slow < 1.0:
        problems.append(
            f"hız yetersiz: {slow}–{fast} token/sn tahmini; {answer_tokens} token'lık "
            "bir yanıt birkaç dakika sürer, sohbet deneyimi olmaz"
        )
    elif slow < 2.0:
        problems.append(
            f"hız sınırda: {slow}–{fast} token/sn tahmini. Kısa yanıtlarda kabul "
            "edilebilir, uzun yanıtlarda dakikalar sürer."
        )

    if any("bellek" in p for p in problems):
        verdict = "sigmiyor"
    elif any("yetersiz" in p for p in problems):
        verdict = "cok-yavas"
    elif problems:
        verdict = "sinirda"
    else:
        verdict = "uygun"

    return {
        "verdict": verdict,
        "params_b": params_b,
        "quant": quant,
        "context_tokens": context_tokens,
        "weights_gb": weights,
        "kv_cache_gb": kv,
        "total_gb": total,
        "ram_gb": ram_gb,
        "vcpus": vcpus,
        "tokens_per_s_range": [slow, fast],
        "seconds_per_answer": seconds,
        "answer_tokens": answer_tokens,
        "problems": problems,
        "note": (
            "Tahmin: CPU üretimi bellek bant genişliğine bağlıdır; bant genişliği "
            f"{low_bw}–{high_bw} GB/sn/vCPU varsayıldı. Gerçek hız çekirdek "
            "tipine, llama.cpp derleme bayraklarına ve bağlam uzunluğuna göre "
            "değişir; bu bir ölçüm değildir."
        ),
    }


def detect_ram_gb() -> float | None:
    """Total RAM in GB, or ``None`` if it cannot be determined."""
    try:
        import psutil  # noqa: PLC0415 - optional

        return round(psutil.virtual_memory().total / GB, 1)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return round(kb * 1024 / GB, 1)
        except (OSError, ValueError, IndexError):  # pragma: no cover - defensive
            return None
    return None


def detect_disk_gb(path: Path) -> float | None:
    """Free space on the filesystem holding ``path`` (or its nearest parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / GB, 1)
    except OSError:  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# merge (the only part that needs a GPU-less full model in memory)
# ---------------------------------------------------------------------------
def merge_adapter(
    adapter_dir: str | os.PathLike[str],
    base_model: str,
    out_dir: str | os.PathLike[str],
    *,
    dtype: str = "bfloat16",
    token: str | None = None,
) -> Path:
    """Merge a PEFT adapter into its base model and save a plain HF checkpoint.

    Heavy: the base model is loaded in ``dtype`` (≈42 GB for gpt-oss-20b in
    bf16). Call :func:`plan_gguf` first and respect its verdict.
    """
    try:
        import torch  # noqa: PLC0415 - optional dependency
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PublishError(
            "Birleştirme için torch/transformers/peft gerekir: pip install '.[train]'"
        ) from exc

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise PublishError(f"bilinmeyen dtype {dtype!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, token=token
    )
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise PublishError(f"manifest bulunamadı: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublishError(f"{p}: geçersiz JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gokturk.publish",
        description="Adaptörü Hugging Face'e yükle veya GGUF derleme planı üret.",
    )
    parser.add_argument("--adapter", help="Eğitim çıktısındaki adapter/ dizini")
    parser.add_argument("--base-model", default="openai/gpt-oss-20b")
    parser.add_argument("--manifest", help="run_manifest.json (model kartı için)")
    parser.add_argument("--repo-id", help="Hugging Face deposu: kullanıcı/ad")
    parser.add_argument("--public", action="store_true", help="Depoyu herkese açık yap")
    parser.add_argument("--push", action="store_true", help="Gerçekten yükle (yoksa yalnızca plan)")
    parser.add_argument("--card-only", action="store_true", help="Model kartını stdout'a yaz")
    parser.add_argument("--gguf-plan", action="store_true", help="GGUF derleme planını üret")
    parser.add_argument("--merge", action="store_true", help="LoRA'yı birleştir (ağır)")
    parser.add_argument("--merge-out", default="outputs/gguf/merged")
    parser.add_argument("--merge-dtype", default="bfloat16", choices=sorted(DTYPE_BYTES))
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        choices=sorted(QUANTS),
        help="GGUF hedefi (varsayılan mxfp4: gpt-oss'un doğal biçimi)",
    )
    parser.add_argument(
        "--serve-plan",
        action="store_true",
        help="CPU'da servis edilebilir mi? (RAM/bağlam/hız hesabı)",
    )
    parser.add_argument("--context", type=int, default=4096, help="Bağlam token sayısı")
    parser.add_argument(
        "--serve-quant",
        default="Q4_K_M",
        choices=sorted(QUANTS),
        help="CPU servisi için quant (varsayılan Q4_K_M; MXFP4 gpt-oss'a özgü)",
    )
    parser.add_argument("--ram-gb", type=float, default=None, help="Kullanılabilir RAM (GB)")
    parser.add_argument("--vcpu", type=int, default=2, help="CPU çekirdek sayısı")
    parser.add_argument("--json", action="store_true", help="Makine okunur çıktı")
    args = parser.parse_args(argv)

    try:
        if args.serve_plan:
            params_b = BASE_PARAMS_B.get(args.base_model)
            if params_b is None:
                print(
                    f"Bilinmeyen taban model: {args.base_model}. "
                    f"Bilinenler: {sorted(BASE_PARAMS_B)}",
                    file=sys.stderr,
                )
                return 4
            plan = plan_cpu_serving(
                params_b,
                quant=args.serve_quant,
                context_tokens=args.context,
                ram_gb=args.ram_gb if args.ram_gb is not None else detect_ram_gb(),
                vcpus=args.vcpu,
                moe_expert_share=(
                    MOE_EXPERT_SHARE if "gpt-oss" in args.base_model else 0.0
                ),
            )
            if args.json:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
            else:
                _print_serve_plan(plan, args.base_model)
            return 0 if plan["verdict"] == "uygun" else 2

        if args.card_only:
            card = build_model_card(
                _load_manifest(args.manifest),
                repo_id=args.repo_id,
                base_model=args.base_model,
            )
            print(card)
            return 0

        if args.gguf_plan:
            if not args.adapter:
                print("--adapter gerekli", file=sys.stderr)
                return 4
            plan = plan_gguf(
                args.adapter,
                base_model=args.base_model,
                quant=args.quant,
                merge_dtype=args.merge_dtype if args.merge_dtype != "float32" else "float32",
            )
            if args.json:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
            else:
                _print_gguf_plan(plan)
            return 0 if plan["feasible"] else 2

        if args.merge:
            if not args.adapter:
                print("--adapter gerekli", file=sys.stderr)
                return 4
            plan = plan_gguf(
                args.adapter,
                base_model=args.base_model,
                merge_dtype=args.merge_dtype,
            )
            if not plan["feasible"]:
                print("Birleştirme bu makinede güvenli değil:", file=sys.stderr)
                for problem in plan["problems"]:
                    print("  -", problem, file=sys.stderr)
                return 2
            out = merge_adapter(
                args.adapter,
                args.base_model,
                args.merge_out,
                dtype=args.merge_dtype,
                token=os.environ.get(HF_TOKEN_ENV) or None,
            )
            print(f"Birleştirilmiş checkpoint: {out}")
            return 0

        if not args.repo_id or not args.adapter:
            print("--repo-id ve --adapter gerekli (veya --gguf-plan / --card-only)", file=sys.stderr)
            return 4

        private = not args.public
        if not args.push:
            plan = plan_push(args.adapter, args.repo_id, private=private)
            payload = plan.as_dict()
            payload["dry_run"] = True
            payload["ok"] = plan.ok
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if not plan.ok:
                for problem in plan.problems:
                    print("  -", problem, file=sys.stderr)
            return 0 if plan.ok else 2

        result = push_adapter(
            args.adapter,
            args.repo_id,
            private=private,
            manifest=_load_manifest(args.manifest),
            base_model=args.base_model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PublishError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 4


def _print_serve_plan(plan: dict[str, Any], model: str) -> None:
    verdicts = {
        "uygun": "UYGUN — bu donanımda rahat servis edilir",
        "sinirda": "SINIRDA — çalışır ama yanıtlar yavaş",
        "cok-yavas": "ÇOK YAVAŞ — sığar, sohbet deneyimi olmaz",
        "sigmiyor": "SIĞMIYOR — bellek yetmiyor",
    }
    print("=" * 72)
    print("CPU SERVİS PLANI —", model)
    print("=" * 72)
    print(f"{'KARAR':<17}: {verdicts.get(plan['verdict'], plan['verdict'])}")
    print(f"{'Ağırlıklar':<17}: ≈{plan['weights_gb']} GB ({plan['quant']})")
    print(
        f"{'KV önbelleği':<17}: ≈{plan['kv_cache_gb']} GB "
        f"({plan['context_tokens']} token bağlam)"
    )
    print(f"{'Toplam':<17}: ≈{plan['total_gb']} GB / RAM {plan['ram_gb']} GB")
    low, high = plan["tokens_per_s_range"]
    print(f"{'Hız':<17}: {low}–{high} token/sn ({plan['vcpus']} vCPU)")
    if plan["seconds_per_answer"]:
        print(
            f"{'Bir yanıt':<17}: ≈{plan['seconds_per_answer']} sn "
            f"({plan['answer_tokens']} token)"
        )
    for problem in plan["problems"]:
        print("  ✗", problem)
    print("\n" + plan["note"])


def _print_gguf_plan(plan: dict[str, Any]) -> None:
    sizes = plan["sizes_gb"]
    print("=" * 72)
    print("GGUF DERLEME PLANI —", plan["base_model"], f"({plan['params_b']}B parametre)")
    print("=" * 72)
    print(f"Hedef quant      : {plan['quant']}")
    print(f"Not              : {plan['quant_note']}")
    print("Boyut tahminleri :")
    print(f"  birleşik checkpoint : ≈{sizes['merged_gb']} GB")
    print(f"  bf16 ara GGUF      : ≈{sizes['gguf_unquantized_gb']} GB")
    print(f"  nihai {plan['quant']:<10}     : ≈{sizes['final_gb']} GB")
    print(
        f"Gereken          : ≈{plan['needed_ram_gb']} GB RAM, "
        f"≈{plan['needed_disk_gb']} GB disk"
    )
    print(
        f"Mevcut           : RAM {plan['available_ram_gb']}, disk {plan['available_disk_gb']}"
    )
    print()
    if plan["feasible"]:
        print("KARAR: YAPILABİLİR — adımlar:")
    else:
        print("KARAR: BU MAKİNEDE YAPILAMAZ")
        for problem in plan["problems"]:
            print("  ✗", problem)
        print("\nYine de komutlar (daha büyük bir makinede çalıştırın):")
    for step in plan["steps"]:
        print(f"\n  {step['step']}. {step['what']}")
        command = step["command"]
        # `bash -c "<script>"` prints as one shell line instead of a token soup.
        rendered = command[2] if command[:2] == ["bash", "-c"] else " ".join(command)
        print("     " + rendered)
    print("\n" + plan["caveat"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
