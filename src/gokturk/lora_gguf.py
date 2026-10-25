"""Adapter-only GGUF (llama.cpp LoRA) — plan, and optionally run the conversion.

Why this module exists
----------------------
:func:`gokturk.publish.plan_gguf` answers a different question: *can this
machine merge a LoRA into the base and ship one file?* That needs the whole
model in RAM (≈67 GB for a 4B in fp16) and ≈100 GB of disk, so on Kaggle it
always answers "no".

There is a second, much cheaper route: ``convert_lora_to_gguf.py`` converts the
**adapter alone**. It reads the base model's *config* (never its weights) and
writes a small ``.gguf`` that llama.cpp applies at load time with
``llama-server -m base.gguf --lora adapter.gguf``. Memory is then the adapter
plus a config, i.e. ~2 GB, and no merge happens anywhere.

This module plans that route honestly: it checks the adapter shape against what
llama.cpp's converter accepts, states the base GGUF you still need to download,
estimates time and disk, and prints the exact commands. ``--run`` executes the
conversion when a llama.cpp checkout is already present.

Verified facts (2026-10-25, llama.cpp master)
---------------------------------------------
* ``conversion/qwen.py`` registers
  ``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5ForCausalLM`` →
  ``MODEL_ARCH.QWEN35``; ``src/models/qwen35.cpp`` exists, so conversion **and**
  runtime support are both present.
* ``MODEL_TENSORS[QWEN35]`` contains ``ATTN_Q/K/V/OUT`` and
  ``FFN_GATE/DOWN/UP`` — exactly the modules this repo trains.
* ``src/llama-adapter.cpp`` only requires that the adapter's architecture
  matches the base model's ("model arch and LoRA arch mismatch"); there is no
  per-architecture LoRA restriction.
* ``convert_lora_to_gguf.py`` refuses adapters that contain
  ``.embed_tokens``/``.lm_head`` tensors, and has no DoRA support.
* It needs ``--base-model-id`` for a Hub model: ``--base`` expects a *directory*.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .publish import BASE_PARAMS_B, GB, QUANTS, detect_disk_gb, detect_ram_gb

#: Repro of the llama.cpp source used for the support claims above.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

#: llama.cpp architecture name per HF base family, plus whether each stage is
#: present in llama.cpp. ``conversion`` = ``convert_lora_to_gguf.py`` accepts the
#: arch, ``runtime`` = the C++ graph exists, ``lora`` = adapters can be applied.
ARCH_SUPPORT: dict[str, dict[str, Any]] = {
    "qwen3_5": {
        "llama_arch": "QWEN35",
        "conversion": True,
        "runtime": True,
        "lora": True,
        "hf_classes": ["Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"],
        "note": "conversion/qwen.py + src/models/qwen35.cpp; vision tower yok sayılır",
    },
    "qwen3_5_moe": {
        "llama_arch": "QWEN35MOE",
        "conversion": True,
        "runtime": True,
        "lora": True,
        "hf_classes": ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"],
        "note": "conversion/qwen.py + src/models/qwen35moe.cpp",
    },
    "qwen3": {
        "llama_arch": "QWEN3",
        "conversion": True,
        "runtime": True,
        "lora": True,
        "hf_classes": ["Qwen3ForCausalLM"],
        "note": "conversion/qwen.py + src/models/qwen3.cpp",
    },
    "gpt_oss": {
        "llama_arch": "GPT_OSS",
        "conversion": True,
        "runtime": True,
        "lora": True,
        "hf_classes": ["GptOssForCausalLM"],
        "note": "MXFP4 taban; adaptör yolu destekli",
    },
}

#: HF base id -> (repo with ready-made GGUFs, recommended quant). Downloading a
#: prebuilt base saves the whole bf16 conversion step.
KNOWN_BASE_GGUF: dict[str, str] = {
    "Qwen/Qwen3.5-4B": "unsloth/Qwen3.5-4B-GGUF",
    "Qwen/Qwen3.5-9B": "unsloth/Qwen3.5-9B-GGUF",
    "Qwen/Qwen3.5-27B": "unsloth/Qwen3.5-27B-GGUF",
    "Qwen/Qwen3-14B": "unsloth/Qwen3-14B-GGUF",
    "Qwen/Qwen3-0.6B": "unsloth/Qwen3-0.6B-GGUF",
    "openai/gpt-oss-20b": "unsloth/gpt-oss-20b-GGUF",
}

#: Modules the converter refuses outright (they change the vocab/head shape).
FORBIDDEN_TARGETS = ("embed_tokens", "lm_token", "lm_head")

#: Conservative memory floor: Python + torch-free conversion + adapter + config.
BASE_RAM_GB = 2.0


class AdapterGGUFError(RuntimeError):
    """Raised when the adapter cannot be converted at all (shape/format)."""


def detect_arch(base_model: str) -> str | None:
    """Map an HF model id to an :data:`ARCH_SUPPORT` key, without the network."""
    lowered = base_model.lower()
    if "gpt-oss" in lowered:
        return "gpt_oss"
    if "qwen3.5" in lowered:
        return "qwen3_5_moe" if any(t in lowered for t in ("a3b", "moe")) else "qwen3_5"
    if "qwen3" in lowered:
        return "qwen3"
    return None


def read_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """Read ``adapter_config.json``; raise with a Turkish message if missing."""
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise AdapterGGUFError(
            f"{adapter_dir} bir PEFT adaptörü gibi görünmüyor: adapter_config.json yok. "
            "Eğitim çıktısındaki `adapter/` dizinini verin."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterGGUFError(f"adapter_config.json okunamadı: {exc}") from exc


def adapter_weight_mb(adapter_dir: Path) -> float:
    """Size of ``adapter_model.safetensors`` (or ``.bin``) in MB."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.is_file():
            return round(path.stat().st_size / 1e6, 1)
    raise AdapterGGUFError(
        f"{adapter_dir} içinde adapter_model.safetensors/.bin yok; dönüştürülecek ağırlık bulunamadı."
    )


def check_adapter_compatibility(adapter_dir: Path, config: dict[str, Any]) -> list[str]:
    """Reasons llama.cpp would refuse this adapter. Empty list == convertible."""
    problems: list[str] = []
    targets = config.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]
    lowered = [str(t).lower() for t in targets]
    bad = [t for t in lowered if any(forbidden in t for forbidden in FORBIDDEN_TARGETS)]
    if bad:
        problems.append(
            f"Adaptör gömme/başlık katmanlarını hedefliyor ({', '.join(sorted(set(bad)))}); "
            "llama.cpp dönüştürücüsü bunları reddeder (sözlük değişmiş demektir)."
        )
    if str(config.get("use_dora", False)).lower() in ("true", "1"):
        problems.append(
            "DoRA adaptörü: convert_lora_to_gguf.py yalnızca LoRA A/B destekler. "
            "Eğitimi `use_dora: false` ile tekrarlayın."
        )
    if config.get("modules_to_save"):
        problems.append(
            "modules_to_save dolu (ör. lm_head/embed): adaptörle birlikte tam katman "
            "kaydedilmiş; llama.cpp bunları LoRA olarak uygulayamaz."
        )
    rank = config.get("r")
    if isinstance(rank, int) and rank > 0:
        # llama-server must be started with a matching --lora the model loads;
        # rank itself is read from the file, this is only informational.
        pass
    return problems


def _base_gguf_gb(base_model: str, quant: str) -> float:
    params_b = BASE_PARAMS_B.get(base_model, 4.0)
    bits = QUANTS.get(quant, QUANTS["Q4_K_M"])["bits"]
    return round(params_b * 1e9 * bits / 8 / GB, 1)


def plan_adapter_gguf(
    adapter_dir: str | os.PathLike[str],
    *,
    base_model: str,
    quant: str = "Q4_K_M",
    out_dir: str = "outputs/gguf",
    ram_gb: float | None = None,
    disk_gb: float | None = None,
    base_gguf_repo: str | None = None,
) -> dict[str, Any]:
    """Decide whether an **adapter-only** GGUF is possible, and how long it takes.

    No merge, no base weights, no bf16 copy: converts ``adapter_model.safetensors``
    to a small ``.gguf`` that llama.cpp applies at load time. The base model still
    needs an ordinary GGUF file; a ready-made one is suggested when known.
    """
    if quant not in QUANTS:
        raise AdapterGGUFError(f"bilinmeyen quant {quant!r}; desteklenen: {sorted(QUANTS)}")

    adapter = Path(adapter_dir)
    problems: list[str] = []
    if not adapter.is_dir():
        problems.append(f"adaptör dizini yok: {adapter}")
        config: dict[str, Any] = {}
        adapter_mb = 0.0
    else:
        config = read_adapter_config(adapter)
        adapter_mb = adapter_weight_mb(adapter)
        problems.extend(check_adapter_compatibility(adapter, config))

    declared_base = str(config.get("base_model_name_or_path") or "")
    if declared_base and declared_base != base_model:
        problems.append(
            f"Adaptör '{declared_base}' için eğitilmiş, plan '{base_model}' diyor. "
            "Taban GGUF ile adaptör aynı model olmalı (llama.cpp arch uyuşmazlığını reddeder)."
        )

    arch_key = detect_arch(base_model)
    arch = ARCH_SUPPORT.get(arch_key or "")
    if arch is None:
        problems.append(
            f"'{base_model}' için llama.cpp mimari eşlemesi bilinmiyor. "
            "llama.cpp'de `conversion/` ve `src/models/` altında mimariyi doğrulayın; "
            "desteklenmiyorsa bu adaptör .gguf'a çevrilemez."
        )
    elif not (arch["conversion"] and arch["runtime"] and arch["lora"]):
        problems.append(
            f"llama.cpp {arch['llama_arch']} mimarisini tam desteklemiyor "
            f"(conversion={arch['conversion']}, runtime={arch['runtime']}, lora={arch['lora']})."
        )

    ram = ram_gb if ram_gb is not None else detect_ram_gb()
    disk = disk_gb if disk_gb is not None else detect_disk_gb(Path(out_dir))

    base_repo = base_gguf_repo or KNOWN_BASE_GGUF.get(base_model)
    base_gb = _base_gguf_gb(base_model, quant)
    # f16 adapter GGUF ≈ the safetensors it reads (both are 2 bytes/parameter).
    adapter_gguf_mb = round(adapter_mb, 1)
    needed_disk = round(base_gb + adapter_gguf_mb / 1000 + 1.0, 1)

    if ram is not None and ram < BASE_RAM_GB:
        problems.append(
            f"RAM yetersiz: dönüşüm ≈{BASE_RAM_GB} GB ister, makinede ≈{ram:.1f} GB var."
        )
    if disk is not None and disk < needed_disk:
        problems.append(
            f"Disk yetersiz: taban GGUF {base_gb} GB + adaptör GGUF "
            f"{adapter_gguf_mb / 1000:.2f} GB + pay ≈{needed_disk} GB gerekir, "
            f"boş ≈{disk:.1f} GB."
        )

    out = Path(out_dir)
    adapter_out = out / f"gokturk-lora-{arch['llama_arch'].lower() if arch else 'x'}.gguf"
    base_file = out / "base" / f"*{quant}*.gguf"

    steps: list[dict[str, Any]] = []
    if base_repo:
        steps.append(
            {
                "step": 1,
                "what": f"Hazır taban GGUF'u indir ({base_repo}, {quant})",
                "command": [
                    "huggingface-cli",
                    "download",
                    base_repo,
                    "--include",
                    f"*{quant}*.gguf",
                    "--local-dir",
                    str(out / "base"),
                ],
                "minutes": f"2-6 (≈{base_gb} GB, bağlantıya göre)",
            }
        )
    else:
        steps.append(
            {
                "step": 1,
                "what": "Taban için GGUF bulun/üretin (hazır depo bilinmiyor)",
                "command": ["llama.cpp/convert_hf_to_gguf.py", "<taban-dizini>", "--outtype", "bf16"],
                "minutes": f"taban modelin tamamı (≈{base_gb} GB) indirilir",
            }
        )
    steps.extend(
        [
            {
                "step": 2,
                "what": "llama.cpp'yi al ve dönüştürücünün bağımlılıklarını kur",
                "command": [
                    "bash",
                    "-c",
                    f"git clone --depth 1 {LLAMA_CPP_REPO} llama.cpp && pip install gguf",
                ],
                "minutes": "1-3 (klon) + 1 (gguf)",
            },
            {
                "step": 3,
                "what": "ADAPTÖRÜ dönüştür (taban ağırlıkları indirilmez, birleştirme YOK)",
                "command": [
                    "python",
                    "llama.cpp/convert_lora_to_gguf.py",
                    str(adapter),
                    "--base-model-id",
                    base_model,
                    "--outfile",
                    str(adapter_out),
                    "--outtype",
                    "f16",
                ],
                "minutes": f"1-3 (≈{adapter_mb} MB adaptör)",
            },
            {
                "step": 4,
                "what": "Doğrula: taban + adaptör birlikte yükleniyor mu?",
                "command": [
                    "llama.cpp/build/bin/llama-server",
                    "-m",
                    str(base_file),
                    "--lora",
                    str(adapter_out),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8080",
                ],
                "minutes": "1-2 (yalnızca yükleme; yanıt üretimi ayrı)",
            },
        ]
    )

    return {
        "feasible": not problems,
        "problems": problems,
        "kind": "adapter-only",
        "merge": False,
        "base_model": base_model,
        "base_gguf_repo": base_repo,
        "quant": quant,
        "llama_arch": arch["llama_arch"] if arch else None,
        "arch_verified": arch is not None,
        "arch_note": arch["note"] if arch else None,
        "adapter_dir": str(adapter),
        "adapter_mb": adapter_mb,
        "adapter_rank": config.get("r"),
        "adapter_alpha": config.get("lora_alpha"),
        "adapter_targets": config.get("target_modules"),
        "adapter_gguf_mb": adapter_gguf_mb,
        "adapter_gguf_path": str(adapter_out),
        "base_gguf_gb": base_gb,
        "needed_ram_gb": BASE_RAM_GB,
        "needed_disk_gb": needed_disk,
        "available_ram_gb": ram,
        "available_disk_gb": disk,
        "estimated_total_minutes": "4-13 (taban GGUF indirmesi hariç)" if base_repo else "6-16",
        "steps": steps,
        "caveat": (
            "Süreler tahmindir; ölçülen tek şey adaptör boyutu ve boş disk. "
            "Dönüşüm bu modülde yalnızca --run ile çalışır."
        ),
    }


def render_plan(plan: dict[str, Any]) -> str:
    """Human-readable Turkish rendering of :func:`plan_adapter_gguf`."""
    lines = [
        "=" * 62,
        "Adaptör-only GGUF planı (llama.cpp LoRA, birleştirme YOK)",
        "=" * 62,
        f"Taban model      : {plan['base_model']}",
        f"llama.cpp mimari : {plan['llama_arch'] or 'BİLİNMİYOR'}",
        f"Adaptör          : {plan['adapter_mb']} MB (r={plan['adapter_rank']}, "
        f"alpha={plan['adapter_alpha']})",
        f"Adaptör GGUF     : ≈{plan['adapter_gguf_mb']} MB (f16)",
        f"Taban GGUF       : ≈{plan['base_gguf_gb']} GB ({plan['quant']})"
        + (f" -> {plan['base_gguf_repo']}" if plan.get("base_gguf_repo") else ""),
        f"Gereken RAM      : ≈{plan['needed_ram_gb']} GB (birleştirme yok)",
        f"Gereken disk     : ≈{plan['needed_disk_gb']} GB",
        f"Süre tahmini     : {plan['estimated_total_minutes']}",
        "",
    ]
    if plan.get("arch_note"):
        lines.append(f"[i] {plan['arch_note']}")
    if plan["problems"]:
        lines.append("ENGELLEYİCİ SORUN VAR:")
        lines.extend(f"  [✗] {p}" for p in plan["problems"])
    else:
        lines.append("SONUÇ: YAPILABİLİR — adaptör tek başına .gguf'a çevrilebilir.")
    lines.append("")
    lines.append("Adımlar:")
    for step in plan["steps"]:
        lines.append(f"  {step['step']}. {step['what']}  ({step['minutes']})")
        lines.append("     " + " ".join(str(part) for part in step["command"]))
    lines.append("")
    lines.append(plan["caveat"])
    return "\n".join(lines)


def run_conversion(
    plan: dict[str, Any],
    *,
    llama_cpp_dir: str | os.PathLike[str],
) -> Path:
    """Run ``convert_lora_to_gguf.py`` from an existing llama.cpp checkout.

    Refuses when the plan is infeasible, when the script is missing, or when the
    converter does not expose ``--base-model-id`` (older/newer release drift).
    """
    if not plan["feasible"]:
        raise AdapterGGUFError(
            "Plan engelleyici sorun bildiriyor; dönüşüm çalıştırılmadı: "
            + "; ".join(plan["problems"])
        )
    root = Path(llama_cpp_dir)
    script = root / "convert_lora_to_gguf.py"
    if not script.is_file():
        raise AdapterGGUFError(
            f"{script} yok. Önce: git clone --depth 1 {LLAMA_CPP_REPO} {root}"
        )
    if "--base-model-id" not in script.read_text(encoding="utf-8", errors="ignore"):
        raise AdapterGGUFError(
            "llama.cpp dönüştürücüsünde --base-model-id yok; sürüm değişmiş. "
            "Taban config'ini yerel bir dizine indirip --base <dizin> kullanın."
        )
    out = Path(plan["adapter_gguf_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        plan["adapter_dir"],
        "--base-model-id",
        plan["base_model"],
        "--outfile",
        str(out),
        "--outtype",
        "f16",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise AdapterGGUFError(
            f"convert_lora_to_gguf.py başarısız (çıkış {completed.returncode}):\n"
            f"{(completed.stderr or completed.stdout)[-2000:]}"
        )
    if not out.is_file():
        raise AdapterGGUFError(f"Dönüşüm 0 döndü ama çıktı yok: {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptörü tek başına GGUF'a çevir (llama.cpp LoRA; birleştirme yok)"
    )
    parser.add_argument("--adapter", required=True, help="Eğitim çıktısındaki adapter/ dizini")
    parser.add_argument("--base-model", required=True, help="ör. Qwen/Qwen3.5-4B")
    parser.add_argument("--quant", default="Q4_K_M", choices=sorted(QUANTS))
    parser.add_argument("--out-dir", default="outputs/gguf")
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--disk-gb", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--run",
        metavar="LLAMA_CPP_DIR",
        default=None,
        help="Dönüşümü bu llama.cpp kopyasıyla gerçekten çalıştır",
    )
    args = parser.parse_args(argv)

    try:
        plan = plan_adapter_gguf(
            args.adapter,
            base_model=args.base_model,
            quant=args.quant,
            out_dir=args.out_dir,
            ram_gb=args.ram_gb,
            disk_gb=args.disk_gb,
        )
    except AdapterGGUFError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 4

    print(json.dumps(plan, ensure_ascii=False, indent=2) if args.json else render_plan(plan))
    if args.run:
        try:
            out = run_conversion(plan, llama_cpp_dir=args.run)
        except AdapterGGUFError as exc:
            print(f"\nDönüşüm çalıştırılamadı: {exc}", file=sys.stderr)
            return 3
        print(f"\nYazıldı: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0 if plan["feasible"] else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
