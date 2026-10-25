"""Typed configuration loading and validation.

All configuration lives in YAML files under ``configs/``. Loading is strict:
unknown keys raise :class:`ConfigError` so that a typo in a YAML file can never
silently change training behaviour.

API credentials are *never* part of configuration. Configuration only stores
the **name** of an environment variable (``api_key_env``) that is read at
runtime; see :mod:`gokturk.teachers`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

try:  # pragma: no cover - import guard is covered by the fallback branch
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


T = TypeVar("T")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'config'} must be a mapping, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown key(s) {unknown}. "
            f"Known keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(spec.type, str) and spec.type.startswith(("List[", "list[")):
            kwargs[name] = list(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], path: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ConfigError(f"{path}: missing required key(s) {missing}")


ENV_REF_PREFIX = "env:"


def resolve_env_ref(value: str) -> tuple[str, str | None]:
    """Resolve ``env:VAR`` references. Returns (value, missing_env_name)."""
    if not isinstance(value, str) or not value.startswith(ENV_REF_PREFIX):
        return value, None
    var = value[len(ENV_REF_PREFIX) :].strip()
    if not var:
        raise ConfigError("empty 'env:' reference")
    resolved = os.environ.get(var, "").strip()
    if not resolved:
        return "", var
    return resolved, None


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ConfigError(
            "PyYAML is required to load configuration files. "
            "Install it with: pip install pyyaml"
        )
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on file content
        raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{p}: file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return raw


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------
#: Semantic teacher roles used for task routing (see configs/teachers.yaml).
KNOWN_TEACHER_ROLES = ("general", "reasoning", "code")


@dataclass
class TeacherModelConfig:
    """One teacher model reachable through an OpenAI-compatible endpoint."""

    name: str
    model: str
    #: One role or several: "general", "reasoning", "code" (comma separated or list).
    role: Any = "general"
    weight: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    #: Optional per-model override of the provider block.
    api_style: str | None = None

    def validate(self, path: str = "teacher") -> None:
        if not self.name.strip():
            raise ConfigError(f"{path}: 'name' must not be empty")
        # An empty model id is allowed at load time only so that an
        # unconfigured template can be inspected with --dry-run; a real
        # generation run calls TeachersConfig.require_ready() first.
        normalize_roles(self.role)  # validates role names
        if self.weight <= 0:
            raise ConfigError(f"{path} ({self.name}): weight must be > 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"{path} ({self.name}): temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError(f"{path} ({self.name}): top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ConfigError(f"{path} ({self.name}): max_tokens must be >= 1")

    def roles(self) -> list[str]:
        return normalize_roles(self.role)

    def to_public_dict(self) -> dict[str, Any]:
        """Metadata safe to write into a dataset record (no secrets)."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.roles(),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


def normalize_roles(role: Any) -> list[str]:
    """Parse a teacher role declaration into a normalised list."""
    if isinstance(role, str):
        parts = [p.strip() for p in role.replace(";", ",").split(",")]
    elif isinstance(role, (list, tuple)):
        parts = [str(p).strip() for p in role]
    else:
        parts = []
    cleaned = [p for p in parts if p]
    unknown = sorted(set(cleaned) - set(KNOWN_TEACHER_ROLES))
    if unknown:
        raise ConfigError(
            f"unknown teacher role(s) {unknown}; known: {list(KNOWN_TEACHER_ROLES)}"
        )
    return cleaned or ["general"]


@dataclass
class ProviderConfig:
    """Connection settings shared by all teachers of one provider."""

    name: str = "ssyz-evren"
    #: Base URL of the OpenAI-compatible API, e.g. ``https://host/v1``.
    #: Deliberately empty by default: it must be verified against the
    #: provider's own documentation and copied into a local config file.
    #: Supports ``env:VAR_NAME`` so the URL can live outside the repository.
    base_url: str = ""
    api_style: str = "openai-chat"
    api_key_env: str = "EVREN_API_KEY"
    timeout_s: float = 120.0
    max_retries: int = 4
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    requests_per_minute: float = 20.0
    #: Attempts per sample before the batch aborts.
    max_attempts_per_sample: int = 3
    #: Fail the whole run after this many consecutive sample failures.
    abort_after_consecutive_failures: int = 8
    extra_headers: dict[str, str] = field(default_factory=dict)

    def validate(self, path: str = "provider") -> None:
        if self.api_style != "openai-chat":
            raise ConfigError(
                f"{path}.api_style={self.api_style!r} is not implemented. "
                "Only 'openai-chat' (POST {base_url}/chat/completions) is "
                "supported. If the provider uses a different wire format, "
                "implement an adapter in gokturk/teachers.py and document it."
            )
        if not self.api_key_env.isupper():
            raise ConfigError(
                f"{path}.api_key_env must be an upper-case environment variable "
                f"name, got {self.api_key_env!r}. Never put the key itself here."
            )
        if self.timeout_s <= 0:
            raise ConfigError(f"{path}.timeout_s must be > 0")
        if self.max_retries < 0:
            raise ConfigError(f"{path}.max_retries must be >= 0")
        if self.requests_per_minute <= 0:
            raise ConfigError(f"{path}.requests_per_minute must be > 0")
        lowered = {k.lower() for k in self.extra_headers}
        if lowered & {"authorization", "api-key", "x-api-key"}:
            raise ConfigError(
                f"{path}.extra_headers must not contain credentials; "
                "use api_key_env instead."
            )

    def api_key(self) -> str | None:
        """Read the key from the environment. Never logs or returns it to disk."""
        key = os.environ.get(self.api_key_env, "").strip()
        return key or None

    def require_api_key(self) -> str:
        key = self.api_key()
        if key is None:
            raise ConfigError(
                f"Environment variable {self.api_key_env} is not set. "
                "Teacher generation cannot run without credentials. "
                "Set it locally (export) or via Kaggle Secrets; never commit it."
            )
        return key


@dataclass
class TeachersConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    teachers: list[dict[str, Any]] = field(default_factory=list)
    generation_prompt_version: str = "v1"
    #: Environment variables referenced by config but not set (e.g. model ids).
    unresolved_refs: list[str] = field(default_factory=list)

    def validate(self) -> None:
        self.provider.validate()
        if not self.teachers:
            raise ConfigError("teachers: at least one teacher must be configured")
        names = set()
        for i, raw in enumerate(self.teachers):
            t = TeacherModelConfig(**_filter_teacher(raw, i))
            t.validate(f"teachers[{i}]")
            if t.name in names:
                raise ConfigError(f"teachers[{i}]: duplicate name {t.name!r}")
            names.add(t.name)
        if abs(sum(float(t.get("weight", 1.0)) for t in self.teachers) - 0.0) == 0.0:
            raise ConfigError("teachers: weights must not all be zero")

    def models(self) -> list[TeacherModelConfig]:
        return [
            TeacherModelConfig(**_filter_teacher(raw, i))
            for i, raw in enumerate(self.teachers)
        ]

    def require_ready(self) -> None:
        """Raise unless the provider is fully configured for real requests."""
        problems: list[str] = []
        if self.unresolved_refs:
            problems.append(
                "unresolved environment reference(s): "
                + ", ".join(sorted(set(self.unresolved_refs)))
            )
        if not self.provider.base_url.strip():
            problems.append("provider.base_url is empty")
        for m in self.models():
            if not m.model.strip():
                problems.append(f"teacher {m.name!r} has no resolved model id")
        if problems:
            raise ConfigError(
                "Teacher configuration is not ready: " + "; ".join(problems)
                + ". Model ids and the base URL must come from the provider's own "
                "documentation (or an env: reference); this project never guesses them."
            )


def _filter_teacher(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    known = {f.name for f in fields(TeacherModelConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"teachers[{index}]: unknown key(s) {unknown}")
    return dict(raw)


def load_teachers_config(path: str | os.PathLike[str]) -> TeachersConfig:
    raw = load_yaml(path)
    _require_keys(raw, ["provider", "teachers"], str(path))
    provider = _build(ProviderConfig, raw["provider"], "provider")
    unresolved: list[str] = []
    provider.base_url, missing = resolve_env_ref(provider.base_url)
    if missing:
        unresolved.append(missing)
    resolved_teachers = []
    for raw_teacher in raw["teachers"]:
        teacher_raw = dict(raw_teacher)
        if "model" in teacher_raw:
            teacher_raw["model"], missing = resolve_env_ref(str(teacher_raw["model"]))
            if missing:
                unresolved.append(missing)
        resolved_teachers.append(teacher_raw)
    cfg = TeachersConfig(
        provider=provider,
        teachers=resolved_teachers,
        generation_prompt_version=str(raw.get("generation_prompt_version", "v1")),
        unresolved_refs=unresolved,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    #: Base checkpoint. Must be verified against the model card before use.
    name_or_path: str = "openai/gpt-oss-20b"
    #: ``auto`` resolves to bf16 when the GPU supports it, else fp16.
    dtype: str = "auto"
    #: Load the base weights in 4-bit NF4 (requires bitsandbytes + CUDA).
    load_in_4bit: bool = False
    #: Unsloth-style dynamic 4-bit path is *not* implemented here; keep False.
    unsloth_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"

    def validate(self, path: str = "model") -> None:
        if not self.name_or_path.strip():
            raise ConfigError(f"{path}.name_or_path must not be empty")
        if self.dtype not in ("auto", "fp16", "bf16", "fp32"):
            raise ConfigError(
                f"{path}.dtype must be one of auto|fp16|bf16|fp32, got {self.dtype!r}"
            )
        if self.unsloth_4bit:
            raise ConfigError(
                f"{path}.unsloth_4bit is not implemented in gokturk.train. "
                "Use the separate Unsloth path documented in "
                "docs/architecture.md, or set it to false."
            )
        if self.attn_implementation not in ("sdpa", "eager", "flash_attention_2"):
            raise ConfigError(f"{path}.attn_implementation is invalid")


@dataclass
class QuantizationConfig:
    method: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "auto"

    def validate(self, path: str = "quantization") -> None:
        if self.method not in ("nf4", "fp4"):
            raise ConfigError(f"{path}.method must be nf4|fp4")
        if self.compute_dtype not in ("auto", "fp16", "bf16", "fp32"):
            raise ConfigError(f"{path}.compute_dtype is invalid")


@dataclass
class LoraConfigYaml:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: Module name suffixes. ``auto`` resolves from the loaded architecture.
    target_modules: list[str] = field(default_factory=lambda: ["auto"])
    bias: str = "none"

    def validate(self, path: str = "lora") -> None:
        if self.r < 1:
            raise ConfigError(f"{path}.r must be >= 1")
        if self.alpha < 1:
            raise ConfigError(f"{path}.alpha must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"{path}.dropout must be in [0, 1)")
        if not self.target_modules:
            raise ConfigError(f"{path}.target_modules must not be empty")
        if self.bias not in ("none", "all", "lora_only"):
            raise ConfigError(f"{path}.bias must be none|all|lora_only")


@dataclass
class DataConfig:
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/valid.jsonl"
    max_seq_len: int = 1024
    #: Field names in the JSONL file.
    prompt_field: str = "prompt"
    response_field: str = "response"

    def validate(self, path: str = "data") -> None:
        if self.max_seq_len < 64:
            raise ConfigError(f"{path}.max_seq_len must be >= 64")
        for key in ("train_file", "eval_file"):
            if not getattr(self, key):
                raise ConfigError(f"{path}.{key} must not be empty")


@dataclass
class OptimizationConfig:
    method: str = "qlora"  # full | lora | qlora
    epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    eval_steps: int = 100
    report_to: str = "none"

    def validate(self, path: str = "optimization") -> None:
        if self.method not in ("full", "lora", "qlora"):
            raise ConfigError(f"{path}.method must be full|lora|qlora")
        if self.per_device_train_batch_size < 1:
            raise ConfigError(f"{path}.per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError(f"{path}.gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError(f"{path}.learning_rate must be > 0")
        if self.epochs <= 0:
            raise ConfigError(f"{path}.epochs must be > 0")
        if self.max_steps == 0:
            raise ConfigError(f"{path}.max_steps must be -1 (unlimited) or > 0")
        if self.save_total_limit < 0:
            raise ConfigError(f"{path}.save_total_limit must be >= 0")


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfigYaml = field(default_factory=LoraConfigYaml)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output_dir: str = "outputs/run-001"
    #: Do not start training if the preflight finds a blocking problem.
    require_preflight_ok: bool = True
    #: Set true only to inspect the plan/logs on hardware the plan does not
    #: support. Training still refuses to start; nothing is silently faked.
    allow_unverified_hardware: bool = False
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    peft: bool = True

    def validate(self) -> None:
        self.model.validate()
        self.quantization.validate()
        self.lora.validate()
        self.data.validate()
        self.optimization.validate()
        if self.optimization.method == "qlora" and not self.model.load_in_4bit:
            raise ConfigError(
                "optimization.method=qlora requires model.load_in_4bit=true"
            )
        if self.optimization.method != "qlora" and self.model.load_in_4bit:
            raise ConfigError(
                "model.load_in_4bit=true is only valid with optimization.method=qlora"
            )
        if self.optimization.method == "full" and self.peft:
            raise ConfigError("optimization.method=full requires peft=false")
        if not self.output_dir:
            raise ConfigError("output_dir must not be empty")


def load_training_config(path: str | os.PathLike[str]) -> TrainingConfig:
    raw = load_yaml(path)
    allowed = {f.name for f in fields(TrainingConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {unknown}")
    cfg = TrainingConfig(
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        quantization=_build(
            QuantizationConfig, raw.get("quantization", {}), "quantization"
        ),
        lora=_build(LoraConfigYaml, raw.get("lora", {}), "lora"),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        optimization=_build(
            OptimizationConfig, raw.get("optimization", {}), "optimization"
        ),
        output_dir=str(raw.get("output_dir", "outputs/run-001")),
        require_preflight_ok=bool(raw.get("require_preflight_ok", True)),
        allow_unverified_hardware=bool(raw.get("allow_unverified_hardware", False)),
        gradient_checkpointing_kwargs=dict(
            raw.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
        ),
        peft=bool(raw.get("peft", True)),
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------
@dataclass
class QualityConfig:
    min_prompt_chars: int = 12
    max_prompt_chars: int = 8000
    min_response_chars: int = 20
    max_response_chars: int = 16000
    #: Jaccard similarity (word 3-gram shingles) above which two records are
    #: considered near duplicates.
    near_duplicate_threshold: float = 0.9
    #: Reject responses that are obviously truncated or empty of content.
    min_response_words: int = 4
    max_repeated_line_ratio: float = 0.5
    valid_ratio: float = 0.02
    test_ratio: float = 0.02
    seed: int = 42
    #: Drop examples whose response contains these markers (teacher refusals).
    refusal_markers: list[str] = field(
        default_factory=lambda: [
            "olarak dil modeli",
            "as an ai language model",
            "üzgünüm, bu isteği",
            "I'm sorry, but I cannot",
        ]
    )

    def validate(self, path: str = "quality") -> None:
        if self.min_response_chars >= self.max_response_chars:
            raise ConfigError(f"{path}: min_response_chars must be < max_response_chars")
        if self.min_prompt_chars >= self.max_prompt_chars:
            raise ConfigError(f"{path}: min_prompt_chars must be < max_prompt_chars")
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ConfigError(f"{path}.near_duplicate_threshold must be in (0, 1]")
        total = self.valid_ratio + self.test_ratio
        if not 0.0 < total < 0.9:
            raise ConfigError(
                f"{path}: valid_ratio + test_ratio must be in (0, 0.9), got {total}"
            )


def load_quality_config(path: str | os.PathLike[str] | None = None) -> QualityConfig:
    if path is None:
        cfg = QualityConfig()
    else:
        cfg = _build(QualityConfig, load_yaml(path), "quality")
    cfg.validate()
    return cfg
