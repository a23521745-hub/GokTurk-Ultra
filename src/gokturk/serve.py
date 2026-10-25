"""Serving paths without merging: dynamic LoRA.

The bottleneck people hit when publishing a fine-tune is the **merge**: folding
a LoRA into a 9B/20B base materialises the whole model (~18-42 GB) before any
GGUF conversion. On Kaggle's free tier (29 GB RAM, 20 GB disk) that is
impossible, which is why ``gokturk.publish --gguf-plan`` refuses.

Turns out the merge is not required to *use* an adapter. Three runtimes load the
adapter next to the base at inference time:

============  ================================================  ============
path          mechanism                                          merge?
============  ================================================  ============
``peft``      ``PeftModel.from_pretrained(base, adapter)``        no
``vllm-lora`` ``--enable-lora --lora-modules name=path``           no
``llamacpp``  ``convert_lora_to_gguf.py`` + ``--lora``              no
``merged``    ``merge_and_unload`` then ``convert_hf_to_gguf.py``  **yes**
============  ================================================  ============

The llama.cpp path is the interesting one for this project: the converter reads
``adapter_config.json`` + ``adapter_model.safetensors`` and needs only the base
model's *config*, not its weights. So a GGUF deliverable is now possible on
machines that cannot merge at all -- including Kaggle.

What this module does **not** claim
-----------------------------------
Dynamic loading solves *storage and memory during conversion*. It does not make
a 9B model fast on a 2-vCPU box: the base still has to be loaded and run. Speed
is modelled by :func:`gokturk.publish.plan_cpu_serving`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .publish import (
    BASE_PARAMS_B,
    QUANTS,
    PublishError,
    detect_disk_gb,
    detect_ram_gb,
    estimate_sizes,
)

#: Paths are ordered by "least heavy first" -- the first workable one wins.
SERVING_PATHS: dict[str, dict[str, Any]] = {
    "peft": {
        "label": "transformers + PEFT (Python içinde birleştirmeden)",
        "needs": "torch + transformers + peft; GPU önerilir",
        "merge": False,
        "note": (
            "Her istek için adaptör ağırlıkları attention projeksiyonlarına "
            "eklenir. Ek donanım kurulumu yok, tek dosya ile çalışır; "
            "CPU'da yavaş, GPU'da pratik."
        ),
    },
    "vllm-lora": {
        "label": "vLLM çoklu-LoRA sunucusu",
        "needs": "CUDA GPU; vLLM",
        "merge": False,
        "note": (
            "Tek taban model, istek başına adaptör seçimi; "
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 ile /v1/load_lora_adapter "
            "üzerinden çalışırken değiştirilebilir."
        ),
    },
    "llamacpp": {
        "label": "llama.cpp: GGUF taban + GGUF LoRA",
        "needs": "llama.cpp derlemesi; CPU'da da çalışır",
        "merge": False,
        "note": (
            "convert_lora_to_gguf.py yalnızca adaptörü çevirir (taban ağırlıkları "
            "gerekmez), llama-server --lora ile uygular; POST /lora-adapters ile "
            "önbelleğe alınmış adaptörler arasında geçiş yapılabilir."
        ),
    },
    "merged": {
        "label": "LoRA'yı birleştirip tek GGUF üret",
        "needs": "≈1.6× model boyutu RAM + ≈2.5× disk",
        "merge": True,
        "note": "Eski yol; yalnızca tek dosya dağıtmak zorunluysa gerekli.",
    },
}


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def _adapter_size_mb(adapter_dir: str | os.PathLike[str] | None) -> float | None:
    if not adapter_dir:
        return None
    path = Path(adapter_dir)
    if not path.is_dir():
        return None
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / 1e6, 1)


def plan_serving(
    base_model: str = "Qwen/Qwen3.5-9B",
    *,
    adapter_dir: str | os.PathLike[str] | None = None,
    quant: str = "Q4_K_M",
    ram_gb: float | None = None,
    vram_gb: float = 0.0,
    context_tokens: int = 4096,
    vcpus: int = 2,
) -> dict[str, Any]:
    """Rank the no-merge serving paths for a base + adapter pair.

    Returns every path with ``feasible``, the commands to run, and the caveats
    that actually bite in practice (quantised-base limits in vLLM, DoRA/
    embeddings limits in the GGUF converter).
    """
    if quant not in QUANTS:
        raise PublishError(f"bilinmeyen quant {quant!r}; desteklenen: {sorted(QUANTS)}")
    params_b = BASE_PARAMS_B.get(base_model)
    if params_b is None:
        raise PublishError(
            f"bilinmeyen taban model {base_model!r}; bilinenler: {sorted(BASE_PARAMS_B)}"
        )
    ram = ram_gb if ram_gb is not None else detect_ram_gb()
    disk = detect_disk_gb(Path("."))
    moe = "gpt-oss" in base_model
    sizes = estimate_sizes(
        params_b, quant=quant, moe_expert_share=0.93 if moe else 0.0
    )
    weights = sizes["final_gb"]
    adapter_mb = _adapter_size_mb(adapter_dir)

    paths: list[dict[str, Any]] = []

    # --- peft -------------------------------------------------------------
    # The adapter is tiny, but the base must be resident. bf16 for a GPU,
    # fp32/quantised for CPU.
    peft_ram = round(params_b * 2.0, 1) if vram_gb >= params_b * 2.1 else round(params_b * 2.0, 1)
    peft_feasible = (vram_gb >= params_b * 2.1) or (ram or 0) >= peft_ram * 0.95
    paths.append(
        {
            "path": "peft",
            "feasible": peft_feasible,
            "requirements_gb": {"vram_or_ram": peft_ram},
            "adapter_mb": adapter_mb,
            "commands": [
                [
                    sys.executable,
                    "-m",
                    "gokturk.serve",
                    "--emit-space",
                    "space/",
                    "--base-model",
                    base_model,
                    "--adapter-repo",
                    "<kullanici>/gokturk-ultra-lora",
                ]
            ],
            "caveats": [
                "CPU'da 9B fp32 ≈" + str(peft_ram) + " GB RAM ister; GPU'da bf16 aynı boyutta.",
                "Qwen3.5 çok kipli ise yükleyici AutoProcessor ister (bu repoda TODO).",
            ],
        }
    )

    # --- vllm -------------------------------------------------------------
    vllm_feasible = vram_gb >= weights * 1.35
    paths.append(
        {
            "path": "vllm-lora",
            "feasible": vllm_feasible,
            "requirements_gb": {"vram": round(weights * 1.35, 1)},
            "adapter_mb": adapter_mb,
            "commands": [
                [
                    "vllm",
                    "serve",
                    base_model,
                    "--enable-lora",
                    "--max-lora-rank",
                    "32",
                    "--lora-modules",
                    "gokturk=<kullanici>/gokturk-ultra-lora",
                ]
            ],
            "caveats": [
                "bitsandbytes 4-bit tabanla dinamik LoRA desteklenmez: bnb QLoRA'da "
                "yalnızca --qlora-adapter-name-or-path ile tek adaptör, başlangıçta.",
                "Dinamik değiştirme için VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 gerekir "
                "(güvenlik notu: adaptör yolunu çağıran belirler).",
                "Adaptör ranki (r) --max-lora-rank'ı aşamaz. Bizim eğitimde r=16.",
            ],
        }
    )

    # --- llama.cpp LoRA (no merge; the Kaggle-friendly one) ----------------
    lora_gguf_mb = adapter_mb
    paths.append(
        {
            "path": "llamacpp",
            "feasible": True,  # needs no big memory: adapter-only conversion
            "requirements_gb": {"disk": round(weights + 2.0, 1)},
            "adapter_mb": adapter_mb,
            "commands": [
                [
                    "python",
                    "llama.cpp/convert_lora_to_gguf.py",
                    str(adapter_dir or "<adapter>"),
                    "--base",
                    base_model,
                    "--outfile",
                    f"gokturk-{quant}-lora.gguf",
                    "--outtype",
                    "f16",
                ],
                [
                    "llama-server",
                    "-m",
                    f"{base_model.split('/')[-1]}-{quant}.gguf",
                    "--lora",
                    f"gokturk-{quant}-lora.gguf",
                    "-c",
                    str(context_tokens),
                    "--jinja",
                ],
            ],
            "caveats": [
                (
                    f"Adaptör çevirisi küçük (≈{lora_gguf_mb} MB) ve taban ağırlığı "
                    "gerektirmez; bu yüzden Kaggle'da bile yapılabilir."
                    if lora_gguf_mb is not None
                    else "Adaptör çevirisi taban ağırlığı gerektirmez; boyut için "
                    "--adapter <dizin> verin. Kaggle'da bile yapılabilir."
                ),
                "llama.cpp yalnızca lora_A/lora_B çevirir: DoRA desteklenmez, "
                "adaptörde embed_tokens/lm_head varsa dönüşüm hata verir.",
                "Quantize taban + LoRA çalışır ama f16 tabana göre küçük kalite farkı olur.",
                "uygulama sırasında taban modelin kendi GGUF'u gerekir "
                f"(≈{weights} GB indirme).",
            ],
        }
    )

    # --- merged (legacy) --------------------------------------------------
    merged_ram = round(sizes["merged_gb"] * 1.6, 1)
    merged_disk = round(
        sizes["merged_gb"] + sizes["gguf_unquantized_gb"] + sizes["final_gb"] + 5.0, 1
    )
    paths.append(
        {
            "path": "merged",
            # Both walls matter: merging dies on RAM first, then on disk.
            "feasible": (ram or 0) >= merged_ram and (disk or 0) >= merged_disk,
            "requirements_gb": {"ram": merged_ram, "disk": merged_disk},
            "adapter_mb": adapter_mb,
            "commands": [["python", "-m", "gokturk.publish", "--gguf-plan", "--adapter", str(adapter_dir or "<adapter>")]],
            "caveats": [
                f"≈{merged_ram} GB RAM ve ≈{merged_disk} GB disk ister; "
                f"bu makinede RAM {ram} GB, boş disk {disk} GB.",
                "Kaggle ücretsiz katmanında (29 GB RAM / 20 GB disk) mümkün değil.",
                "Yalnızca tek dosya dağıtmak zorunluysa tercih edin.",
            ],
        }
    )

    recommended = next((p["path"] for p in paths if p["feasible"]), None)
    return {
        "base_model": base_model,
        "params_b": params_b,
        "quant": quant,
        "weights_gb": weights,
        "ram_gb": ram,
        "vram_gb": vram_gb,
        "adapter_mb": adapter_mb,
        "paths": paths,
        "recommended": recommended,
        "note": (
            "Birleştirme (merge) yalnızca 'merged' yolunda gerekir. Diğer üç yol "
            "adaptörü taban modelin yanında yükler; 42 GB'lık birleşik checkpoint "
            "üretilmez. Bu, dönüşüm/deployment kısıtını çözer — modelin CPU'da "
            "hızlanmasını sağlamaz (hız için: gokturk.publish --serve-plan)."
        ),
    }


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        f"SERVİS PLANI — {plan['base_model']} + adaptör ({plan['quant']})",
        "=" * 72,
    ]
    for entry in plan["paths"]:
        spec = SERVING_PATHS[entry["path"]]
        mark = "✓" if entry["feasible"] else "✗"
        lines.append(f"\n[{mark}] {entry['path']} — {spec['label']}")
        lines.append(f"    merge gerekir mi: {'EVET' if spec['merge'] else 'hayır'}")
        req = ", ".join(f"{k} ≈{v} GB" for k, v in entry["requirements_gb"].items())
        lines.append(f"    gereksinim: {req}")
        for caveat in entry["caveats"]:
            lines.append(f"    ! {caveat}")
        for command in entry["commands"][:1]:
            lines.append("    $ " + " ".join(command))
    if plan["recommended"]:
        lines.append(f"\nÖNERİLEN: {plan['recommended']}")
    lines.append("\n" + plan["note"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hugging Face Space scaffold (dynamic adapter loading, no merge)
# ---------------------------------------------------------------------------
SPACE_README = """---
title: {title}
emoji: 🐺
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# {title}

{base_model} tabanı + [`{adapter_repo}`](https://huggingface.co/{adapter_repo})
adaptörü. **Birleştirme yok**: adaptör çalışma anında `PeftModel` ile yüklenir.

## Uyarılar

* Bu bir **demo**dur. Ücretsiz Spaces donanımında (2 vCPU) yanıt süreleri
  dakikalarla ölçülür; günlük GPU kotası da sınırlıdır.
* Model kalitesi ölçülmemiştir; benchmark iddiası yoktur.
* Adaptör, sentetik veriyle eğitilmiş bir LoRA'dır ve taban modelin lisansı
  geçerlidir.
"""

SPACE_APP = '''"""GökTürk-Ultra demo: taban model + dinamik LoRA (birleştirme yok).

Adaptör çalışma anında yüklenir (`PeftModel`), yani 18-42 GB'lık birleşik
checkpoint hiç üretilmez. Bu dosya `gokturk.serve --emit-space` tarafından
üretildi; elle düzenleyebilirsiniz.
"""

from __future__ import annotations

import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "{base_model}")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "{adapter_repo}")
SYSTEM = (
    "Sen GökTürk-Ultra'sın: Türkçe konuşan, talimatları izleyen ve gerektiğinde "
    "araç çağıran bir yapay zekâ asistanısın."
)

# ZeroGPU Spaces: @spaces.GPU mevcut olduğunda GPU isteği açar, yoksa no-op.
try:
    import spaces

    gpu = spaces.GPU
except Exception:  # pragma: no cover - normal Python ortamı

    def gpu(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32  # CPU: bf16 yok, fp16 yavaş


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=_pick_dtype(),
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()


@gpu(duration=60)
def chat(message: str, history: list[tuple[str, str]]):
    messages = [{{"role": "system", "content": SYSTEM}}]
    for user, assistant in history or []:
        messages.append({{"role": "user", "content": user}})
        messages.append({{"role": "assistant", "content": assistant}})
    messages.append({{"role": "user", "content": message}})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return answer.strip()


demo = gr.ChatInterface(
    chat,
    title="GökTürk-Ultra (demo)",
    description=(
        f"Taban: {{BASE_MODEL}} + adaptör: {{ADAPTER_REPO}}. "
        "Ücretsiz CPU donanımında yanıtlar yavaştır; bu bir demo, ürün değil."
    ),
)

if __name__ == "__main__":
    demo.launch()
'''

SPACE_REQUIREMENTS = """torch>=2.1
transformers>=4.55,<6
peft>=0.12
accelerate>=0.33
gradio>=4.44
spaces
"""


def emit_space(
    out_dir: str | os.PathLike[str],
    *,
    base_model: str = "Qwen/Qwen3.5-9B",
    adapter_repo: str = "<kullanici>/gokturk-ultra-lora",
    title: str = "GökTürk-Ultra",
    overwrite: bool = False,
) -> dict[str, str]:
    """Write a minimal Space (app.py, requirements.txt, README.md).

    Returns ``{filename: path}``. Existing files are kept unless ``overwrite``,
    because a scaffold must never silently destroy hand edits.
    """
    if adapter_repo.count("/") != 1:
        raise PublishError(
            f"adapter_repo 'kullanici/depo' biçiminde olmalı, alınan: {adapter_repo!r}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "app.py": SPACE_APP.format(base_model=base_model, adapter_repo=adapter_repo),
        "requirements.txt": SPACE_REQUIREMENTS,
        "README.md": SPACE_README.format(
            title=title, base_model=base_model, adapter_repo=adapter_repo
        ),
    }
    written: dict[str, str] = {}
    for name, content in files.items():
        target = out / name
        if target.exists() and not overwrite:
            written[name] = f"{target} (atlandı: zaten var)"
            continue
        target.write_text(content, encoding="utf-8")
        written[name] = str(target)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gokturk.serve",
        description="Birleştirmeden servis: dinamik LoRA yolları ve Spaces iskeleti.",
    )
    parser.add_argument("--plan", action="store_true", help="Servis yollarını karşılaştır")
    parser.add_argument("--emit-space", metavar="DIR", help="Space iskeleti yaz")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", help="Yerel adaptör dizini (boyut/komutlar için)")
    parser.add_argument("--adapter-repo", default="<kullanici>/gokturk-ultra-lora")
    parser.add_argument("--quant", default="Q4_K_M", choices=sorted(QUANTS))
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--vram-gb", type=float, default=0.0)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--vcpu", type=int, default=2)
    parser.add_argument("--title", default="GökTürk-Ultra")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.emit_space:
            written = emit_space(
                args.emit_space,
                base_model=args.base_model,
                adapter_repo=args.adapter_repo,
                title=args.title,
                overwrite=args.overwrite,
            )
            for name, path in written.items():
                print(f"{name:<18} {path}")
            print(
                "\nSonraki adım: bu klasörü bir Hugging Face Space deposu olarak "
                "yükleyin (ücretsiz hesapta Gradio Space için plan kısıtı var; "
                "bkz. docs/servis-ve-dinamik-lora.md)."
            )
            return 0

        if args.plan:
            plan = plan_serving(
                args.base_model,
                adapter_dir=args.adapter,
                quant=args.quant,
                ram_gb=args.ram_gb,
                vram_gb=args.vram_gb,
                context_tokens=args.context,
                vcpus=args.vcpu,
            )
            print(json.dumps(plan, ensure_ascii=False, indent=2) if args.json else render_plan(plan))
            return 0 if plan["recommended"] else 2

        parser.print_help()
        return 4
    except PublishError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
