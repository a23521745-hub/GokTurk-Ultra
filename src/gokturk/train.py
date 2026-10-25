"""Training entry point (LoRA / QLoRA / full fine-tuning) with an honest preflight.

What this module does
---------------------
1. Loads and validates ``configs/training.yaml``.
2. Runs :mod:`gokturk.env_check` and **refuses to start** when the requested
   method is not supported by the detected hardware/architecture. Nothing is
   reported as "trained" unless training actually ran.
3. Loads the JSONL splits produced by :mod:`gokturk.data_quality`.
4. Tokenises chat examples and masks prompt tokens (only the assistant answer
   contributes to the loss).
5. Applies LoRA/QLoRA when configured, trains, and saves adapter + tokenizer +
   a run manifest.

Backends
--------
``--backend transformers`` (default)
    Generic transformers + peft path (LoRA / QLoRA via bitsandbytes).
``--backend unsloth``
    Unsloth's fused 4-bit QLoRA path, which is what makes gpt-oss-20b fit on a
    16 GB GPU (≈14 GB VRAM by Unsloth's published figures). Unsloth is an
    optional dependency; when it is missing the run stops with an actionable
    message instead of silently falling back to a path that cannot fit.

Modes
-----
``--plan-only``   print the preflight report and the resolved plan; no model load.
``--smoke``       tiny randomly initialised model + on-the-fly word-level
                  tokenizer, a couple of optimiser steps on CPU or GPU. This
                  proves the *pipeline* works. It proves nothing about model
                  quality, and it never touches the real checkpoint.
(default)         real training run for the configured checkpoint.

Honesty rules encoded here
--------------------------
* gpt-oss checkpoints ship in MXFP4 and have no gradient-based training support
  in transformers/peft. The preflight therefore blocks QLoRA-on-2×T4 and bf16
  LoRA below ~44 GB VRAM instead of pretending it works.
* ``optimization.method="full"`` at 20B is blocked; the code path exists and is
  exercised by the smoke test only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import TrainingConfig, load_training_config
from .data_quality import read_jsonl
from .env_check import PreflightReport, check_environment, detect_hardware, package_versions

LOGGER = logging.getLogger("gokturk.train")

#: Fallback chat template used only when a tokenizer has no chat_template.
FALLBACK_SYSTEM = (
    "Sen GökTürk-Ultra'sın: Türkçe konuşan, talimatları izleyen ve gerektiğinde "
    "araç çağıran bir yapay zekâ asistanısın."
)
FALLBACK_TEMPLATE = (
    "<|system|>\n" + FALLBACK_SYSTEM + "\n<|user|>\n{prompt}\n<|assistant|>\n"
)
FALLBACK_SUFFIX = "\n<|end|>\n"


# ---------------------------------------------------------------------------
# tokenisation / dataset
# ---------------------------------------------------------------------------
def build_prompt_text(tokenizer: Any, prompt: str, system: str | None = None) -> str:
    """Text of the prompt part, ending right before the assistant answer."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return FALLBACK_TEMPLATE.format(prompt=prompt)


def encode_example(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_len: int,
    system: str | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` with prompt tokens masked as -100."""
    prompt_text = build_prompt_text(tokenizer, prompt, system)
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full_text = prompt_text + response + FALLBACK_SUFFIX

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Longest common prefix: templates can re-normalise whitespace, so trusting
    # len(prompt_ids) blindly could leak prompt tokens into the loss.
    boundary = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):
        if a != b:
            break
        boundary += 1
    if boundary == 0:
        LOGGER.warning("prompt prefix mismatch; masking only the first token")

    input_ids = full_ids[:max_seq_len]
    labels = list(input_ids)
    mask_upto = min(boundary, len(labels))
    for i in range(mask_upto):
        labels[i] = -100
    if all(lab == -100 for lab in labels):
        # Truncation cut the answer away entirely: drop this example.
        return [], []
    return input_ids, labels


class ChatSFTDataset:
    """Minimal torch Dataset over a JSONL file (no ``datasets`` dependency)."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_seq_len: int,
        *,
        prompt_field: str = "prompt",
        response_field: str = "response",
        system: str | None = FALLBACK_SYSTEM,
        limit: int | None = None,
    ) -> None:
        records, issues = read_jsonl(path)
        if issues:
            LOGGER.warning("%d unparseable line(s) in %s", len(issues), path)
        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        for rec in records:
            prompt = str(rec.get(prompt_field, ""))
            response = str(rec.get(response_field, ""))
            input_ids, labels = encode_example(
                tokenizer, prompt, response, max_seq_len, system
            )
            if not input_ids:
                skipped += 1
                continue
            self.examples.append(
                {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
            )
            if limit and len(self.examples) >= limit:
                break
        if skipped:
            LOGGER.warning("%d example(s) skipped after truncation in %s", skipped, path)
        if not self.examples:
            raise ValueError(f"{path}: no usable training example after tokenisation")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def build_collator(pad_token_id: int):
    import torch

    def collate(features: Sequence[Mapping[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, Any] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [pad_token_id] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return collate


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def resolve_torch_dtype(requested: str, hardware) -> str:
    if requested == "auto":
        if hardware.cuda_available and hardware.bf16_supported:
            return "bf16"
        if hardware.cuda_available:
            return "fp16"
        return "fp32"
    return requested


def _torch_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def build_quantization_config(cfg: TrainingConfig):
    """Build a BitsAndBytesConfig for QLoRA. Raises if bitsandbytes is missing."""
    if not cfg.model.load_in_4bit:
        return None
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "QLoRA requires bitsandbytes (CUDA-only). Install it with "
            "`pip install bitsandbytes`, or use method=lora / a CPU smoke run."
        ) from exc
    import torch

    compute_dtype = cfg.quantization.compute_dtype
    if compute_dtype == "auto":
        compute_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.method,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(compute_dtype),
    )


def resolve_target_modules(model: Any, configured: Sequence[str]) -> list[str]:
    """Map configured names to modules that actually exist in the model."""
    if list(configured) != ["auto"]:
        present = {name.split(".")[-1] for name, _ in model.named_modules()}
        missing = [m for m in configured if m not in present]
        if missing:
            raise ValueError(f"LoRA target modules not found in model: {missing}")
        return list(configured)

    import torch.nn as nn

    leaf_names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1]
    }
    leaf_names.discard("lm_head")
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    chosen = [m for m in preferred if m in leaf_names]
    if not chosen:
        # Unknown architecture: fall back to every Linear leaf except lm_head.
        chosen = sorted(leaf_names)
        LOGGER.warning(
            "No known attention/MLP projection names found; applying LoRA to all "
            "linear leaves: %s",
            chosen,
        )
    return chosen


def apply_peft(model: Any, cfg: TrainingConfig) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    target_modules = resolve_target_modules(model, cfg.lora.target_modules)
    LOGGER.info("LoRA hedef modülleri: %s", target_modules)
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.optimization.gradient_checkpointing
        )
    elif cfg.optimization.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=target_modules,
        bias=cfg.lora.bias,  # type: ignore[arg-type]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(cfg: TrainingConfig, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_config = build_quantization_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": 0}  # single GPU, no sharding
    else:
        model_kwargs["torch_dtype"] = _torch_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
    if quant_config is None and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


# ---------------------------------------------------------------------------
# smoke components (no downloads)
# ---------------------------------------------------------------------------
def build_smoke_tokenizer(texts: Sequence[str], max_vocab: int = 2000):
    """A word-level tokenizer trained on the fly: fully offline, deterministic."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # WordLevel needs the full vocabulary (including [UNK]) at construction time,
    # so the vocabulary is collected first and the tokenizer built afterwards.
    trainer_vocab: dict[str, int] = {"[UNK]": 0}
    for text in texts:
        for word in text.split():
            if word not in trainer_vocab and len(trainer_vocab) < max_vocab:
                trainer_vocab[word] = len(trainer_vocab)
    tok = Tokenizer(models.WordLevel(vocab=trainer_vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(
        single="$A [EOS]", special_tokens=[("[EOS]", 0)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="[UNK]",
        pad_token="[UNK]",
    )
    fast.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    return fast


def build_smoke_model(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=max(vocab_size, 64),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        out = os.popen("git rev-parse HEAD 2>/dev/null").read().strip()
        return out or None
    except Exception:  # pragma: no cover
        return None


def estimate_total_steps(cfg: TrainingConfig, num_examples: int, max_steps: int | None) -> int:
    """Total optimiser steps, used to convert warmup_ratio into warmup_steps."""
    opt = cfg.optimization
    if max_steps is not None and max_steps > 0:
        return max_steps
    if opt.max_steps > 0:
        return opt.max_steps
    per_step = max(1, opt.per_device_train_batch_size * opt.gradient_accumulation_steps)
    steps_per_epoch = max(1, -(-num_examples // per_step))
    return max(1, int(steps_per_epoch * opt.epochs))


def build_training_arguments(
    cfg: TrainingConfig,
    dtype: str,
    output_dir: Path,
    *,
    num_examples: int = 1,
    max_steps: int | None = None,
):
    """Build ``TrainingArguments`` across transformers 4.x and 5.x.

    transformers 5 removed ``warmup_ratio``; it is converted to ``warmup_steps``
    from the estimated step count when the installed version requires it.
    Unknown keyword arguments are dropped with a warning instead of crashing.
    """
    import inspect

    from transformers import TrainingArguments

    opt = cfg.optimization
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=opt.epochs,
        max_steps=opt.max_steps,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        learning_rate=opt.learning_rate,
        lr_scheduler_type=opt.lr_scheduler_type,
        weight_decay=opt.weight_decay,
        max_grad_norm=opt.max_grad_norm,
        gradient_checkpointing=opt.gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs,
        logging_steps=opt.logging_steps,
        save_steps=opt.save_steps,
        save_total_limit=opt.save_total_limit,
        eval_strategy="steps",
        eval_steps=opt.eval_steps,
        save_strategy="steps",
        report_to=[] if opt.report_to == "none" else [opt.report_to],
        seed=opt.seed,
        data_seed=opt.seed,
        bf16=dtype == "bf16",
        fp16=dtype == "fp16",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim=opt.optim,
    )
    if dtype == "fp32":
        kwargs["optim"] = "adamw_torch"
    # BitsAndBytes optimizers are CUDA-only; fall back for CPU smoke runs.
    if kwargs["optim"] == "paged_adamw_8bit" and not _cuda_available():
        LOGGER.warning("CUDA yok: optim=paged_adamw_8bit yerine adamw_torch kullanılacak")
        kwargs["optim"] = "adamw_torch"

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" in accepted:
        kwargs["warmup_ratio"] = opt.warmup_ratio
    else:
        total = estimate_total_steps(cfg, num_examples, max_steps)
        kwargs["warmup_steps"] = int(total * opt.warmup_ratio)
    dropped = sorted(set(kwargs) - accepted)
    for key in dropped:
        LOGGER.warning("TrainingArguments bu sürümde '%s' kabul etmiyor; atlandı", key)
        kwargs.pop(key)
    return TrainingArguments(**kwargs)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ModuleNotFoundError:  # pragma: no cover
        return False


def pin_single_gpu(device: str = "0") -> bool:
    """``CUDA_VISIBLE_DEVICES``'i tek karta sabitle; torç import'undan ÖNCE çağır.

    Bu repo hiçbir yerde model paralelliği uygulamıyor (bkz. ``docs/architecture.md``).
    Kaggle'da iki T4 görünürse ``Trainer``/``accelerate`` modeli ``nn.DataParallel``
    ile çoğaltmaya çalışır; 4-bit (bitsandbytes) bir model çoğaltılamaz ve hata
    ancak derin bir izle "replica 0" olarak görünür. Bu yüzden süreç, kullanıcı
    açıkça başka bir şey istemediyse **tek kart** görür.

    Zaten ayarlanmış bir ``CUDA_VISIBLE_DEVICES`` değeri **ezilmez**.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    return True


def run_training(
    cfg: TrainingConfig,
    *,
    smoke: bool = False,
    plan_only: bool = False,
    max_steps: int | None = None,
    output_dir: str | None = None,
    skip_preflight: bool = False,
    cpu_ok: bool = False,
    unsloth_plain_loss: bool | None = None,
    backend: str = "transformers",
) -> int:
    if backend not in ("transformers", "unsloth"):
        raise ValueError(f"unknown backend {backend!r}")
    # Unsloth'un füzyonlu CE çekirdeği, Kaggle imajındaki güncel torch/transformers
    # ile çalışmıyor (2026-10-25'te iki kez "scalar Tensor" hatası). Bu yüzden
    # unsloth yolu için varsayılan DÜZ kayıp; füzyonlu yolu istemek açık bir
    # tercih olmalı (--unsloth-fused-loss).
    if unsloth_plain_loss is None:
        unsloth_plain_loss = backend == "unsloth"
        if unsloth_plain_loss:
            LOGGER.warning(
                "Unsloth arka ucu: füzyonlu kayıp varsayılan olarak KAPALI "
                "(bilinen uyumsuzluk). Açmak için --unsloth-fused-loss."
            )
    if unsloth_plain_loss:
        # UNSLOTH_RETURN_LOGITS=1, Unsloth'un füzyonlu CE çekirdeğini kapatır ve
        # düz logits + standart CE kullanır. Import'tan ÖNCE ayarlanmalı.
        # Bedeli: tam logits bellekte tutulur (daha çok VRAM, biraz daha yavaş).
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        LOGGER.warning(
            "Unsloth füzyonlu kayıp kapatıldı (UNSLOTH_RETURN_LOGITS=1): "
            "daha çok VRAM kullanılır, uyumluluk sorunları bu yolu atlar."
        )
    pin_single_gpu()
    hardware = detect_hardware()
    dtype = resolve_torch_dtype(cfg.model.dtype, hardware)

    # Preflight is computed for the *configured* model unless smoke mode
    # replaces it with a tiny random one.
    model_name = cfg.model.name_or_path
    method = "lora" if smoke else cfg.optimization.method
    if backend == "unsloth" and not smoke:
        from .unsloth_backend import preflight_unsloth

        report: PreflightReport = preflight_unsloth(
            model_name,
            load_in_4bit=cfg.model.load_in_4bit,
            hardware=hardware,
            max_seq_length=cfg.data.max_seq_len,
        )
    else:
        report = check_environment(
            model_name="gokturk-smoke-tiny" if smoke else model_name,
            method=method,
            dtype=dtype,
            load_in_4bit=cfg.model.load_in_4bit if not smoke else False,
            params=(1e6 if smoke else None),
            seq_len=cfg.data.max_seq_len,
            batch_size=cfg.optimization.per_device_train_batch_size,
            check_hub=False,
            cpu_ok=smoke or cpu_ok,
            cpu_mini=(cpu_ok and not smoke),
        )

    if plan_only:
        print(report.render())
        print("\n=== Plan ===")
        print(json.dumps(_plan_dict(cfg, dtype, smoke, backend), ensure_ascii=False, indent=2))
        if not report.ok:
            print("\nUYARI: Bu plan mevcut donanımda çalıştırılamaz (yukarıdaki engelleyici bulgular).")
        return 0

    if not report.ok:
        print(report.render(), file=sys.stderr)
        # İki ayrı kapı: bayrak (--skip-preflight) ve yapılandırma
        # (require_preflight_ok: false). İkisi de açıkça söylenir; sessiz geçiş yok.
        override = skip_preflight or not cfg.require_preflight_ok
        if not override:
            print(
                "\nEğitim başlatılmadı: ön denetim engelleyici sorun bildirdi.\n"
                "Bu donanımda desteklenmeyen bir yöntemi çalıştırıyormuş gibi göstermiyoruz.\n"
                "Seçenekler: farklı donanım, farklı yöntem/model, ya da --plan-only ile "
                "planı inceleyip --skip-preflight ile yalnızca kendi sorumluluğunuzda deneme.",
                file=sys.stderr,
            )
            return 2
        reason = "--skip-preflight verildi" if skip_preflight else "require_preflight_ok: false"
        print(
            f"UYARI: {reason}; engelleyici bulgulara rağmen deneniyor. "
            "Manifest bu bulguları ve koşunun geçersiz kılındığını kaydeder.",
            file=sys.stderr,
        )

    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    texts: list[str] = []
    train_records, _ = read_jsonl(cfg.data.train_file)
    texts = [
        str(r.get("prompt", "")) + "\n" + str(r.get("response", ""))
        for r in train_records[:2000]
    ]

    if smoke:
        tokenizer = build_smoke_tokenizer(texts)
        model = build_smoke_model(len(tokenizer))
        method = "lora"
    elif backend == "unsloth":
        from . import unsloth_backend

        model, tokenizer = unsloth_backend.load_model_and_tokenizer(
            cfg, max_seq_length=cfg.data.max_seq_len
        )
    else:
        model, tokenizer = load_model_and_tokenizer(cfg, dtype)

    if backend == "unsloth" and not smoke:
        from . import unsloth_backend

        model = unsloth_backend.apply_lora(
            model, cfg, max_seq_length=cfg.data.max_seq_len
        )
    elif method in ("lora", "qlora"):
        model = apply_peft(model, cfg)
    else:
        if cfg.optimization.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs
            )

    pad_id = tokenizer.pad_token_id or 0
    train_ds = ChatSFTDataset(
        cfg.data.train_file,
        tokenizer,
        cfg.data.max_seq_len,
        prompt_field=cfg.data.prompt_field,
        response_field=cfg.data.response_field,
    )
    eval_ds = None
    try:
        eval_ds = ChatSFTDataset(
            cfg.data.eval_file,
            tokenizer,
            cfg.data.max_seq_len,
            prompt_field=cfg.data.prompt_field,
            response_field=cfg.data.response_field,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Değerlendirme seti kullanılamıyor (%s); eval atlanacak", exc)

    args = build_training_arguments(
        cfg, dtype, out, num_examples=len(train_ds), max_steps=max_steps
    )
    if max_steps is not None:
        args.max_steps = max_steps
        args.num_train_epochs = 1

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collator(pad_id),
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    if eval_ds is not None:
        metrics.update({f"eval_{k}": v for k, v in trainer.evaluate().items()})

    if backend == "unsloth" and not smoke:
        from . import unsloth_backend

        unsloth_backend.save_adapter(model, tokenizer, str(out / "adapter"))
    else:
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(str(out / "adapter"))

    manifest = {
        "kind": "smoke" if smoke else "training",
        "backend": backend,
        "note": (
            "Duman testi: rastgele başlatılmış küçük model, kalite iddiası yok."
            if smoke
            else "Eğitim çalıştı; kalite iddiası için evaluate.py sonuçları gerekir."
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "git_commit": _git_commit(),
        "config": _config_dict(cfg),
        "resolved_dtype": dtype,
        "packages": package_versions(),
        "preflight": report.to_dict(),
        "data": {"train_examples": len(train_ds), "eval_examples": len(eval_ds) if eval_ds else 0},
        "metrics": metrics,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Bitti. Çıktılar: %s", out)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return 0


def _config_dict(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "model": asdict(cfg.model),
        "quantization": asdict(cfg.quantization),
        "lora": asdict(cfg.lora),
        "data": asdict(cfg.data),
        "optimization": asdict(cfg.optimization),
        "output_dir": cfg.output_dir,
    }


def _plan_dict(
    cfg: TrainingConfig, dtype: str, smoke: bool, backend: str = "transformers"
) -> dict[str, Any]:
    return {
        "backend": backend,
        "base_model": "gokturk-smoke-tiny (rastgele)" if smoke else cfg.model.name_or_path,
        "method": "lora(smoke)" if smoke else cfg.optimization.method,
        "dtype": dtype,
        "load_in_4bit": cfg.model.load_in_4bit,
        "max_seq_len": cfg.data.max_seq_len,
        "batch": cfg.optimization.per_device_train_batch_size,
        "grad_accum": cfg.optimization.gradient_accumulation_steps,
        "epochs": cfg.optimization.epochs,
        "max_steps": cfg.optimization.max_steps,
        "lr": cfg.optimization.learning_rate,
        "lora": asdict(cfg.lora),
        "train_file": cfg.data.train_file,
        "eval_file": cfg.data.eval_file,
        "output_dir": cfg.output_dir,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GökTürk-Ultra eğitim giriş noktası")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument(
        "--backend",
        choices=["transformers", "unsloth"],
        default="transformers",
        help="Eğitim arka ucu: generic transformers/peft veya Unsloth 4-bit QLoRA",
    )
    parser.add_argument("--plan-only", action="store_true", help="Yalnızca denetim ve plan")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Küçük rastgele modelle boru hattı testi (model indirmez, kalite ölçmez)",
    )
    parser.add_argument("--model", default=None, help="model.name_or_path üzerine yaz")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-file", default=None, help="data.train_file üzerine yaz")
    parser.add_argument("--eval-file", default=None, help="data.eval_file üzerine yaz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--unsloth-plain-loss",
        action="store_true",
        help="Unsloth füzyonlu CE kaybını kapat (UNSLOTH_RETURN_LOGITS=1; daha çok VRAM). "
        "Unsloth arka ucunda bu zaten varsayılandır.",
    )
    parser.add_argument(
        "--unsloth-fused-loss",
        action="store_true",
        help="Unsloth füzyonlu kaybı ZORLA aç (bilinen uyumsuzluk; yalnızca doğrulanmış "
        "sürümlerle deneyin)",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="CUDA_VISIBLE_DEVICES değeri (varsayılan '0': tek kart; "
        "'0,1' vermek DataParallel'a yol açar ve 4-bit modellerde çalışmaz)",
    )
    parser.add_argument(
        "--cpu-ok",
        action="store_true",
        help="CPU'da bilinçli mini koşu (≤1B model, kısa LoRA); ön denetim engellemez",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Engelleyici bulguları yok say")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = load_training_config(args.config)
    except Exception as exc:
        print(f"Yapılandırma hatası: {exc}", file=sys.stderr)
        return 4
    if args.gpus:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            LOGGER.warning(
                "Birden fazla GPU görünür (CUDA_VISIBLE_DEVICES=%s): bu repo model "
                "paralelliği uygulamaz; 4-bit modeller DataParallel ile çoğaltılamaz.",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
    if args.unsloth_fused_loss and args.unsloth_plain_loss:
        print("Hata: --unsloth-fused-loss ve --unsloth-plain-loss birlikte verilemez.", file=sys.stderr)
        return 4
    plain_loss = False if args.unsloth_fused_loss else (True if args.unsloth_plain_loss else None)
    if args.model:
        cfg.model.name_or_path = args.model
    if args.train_file:
        cfg.data.train_file = args.train_file
    if args.eval_file:
        cfg.data.eval_file = args.eval_file

    return run_training(
        cfg,
        smoke=args.smoke,
        plan_only=args.plan_only,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        skip_preflight=args.skip_preflight,
        cpu_ok=args.cpu_ok,
        unsloth_plain_loss=plain_loss,
        backend=args.backend,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
