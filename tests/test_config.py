"""Configuration loading, strictness and secret handling."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gokturk.config import (
    ConfigError,
    ProviderConfig,
    TeacherModelConfig,
    TeachersConfig,
    load_quality_config,
    load_teachers_config,
    load_training_config,
    resolve_env_ref,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_repo_configs_are_valid():
    training = load_training_config(ROOT / "configs" / "training.yaml")
    assert training.optimization.method == "qlora"
    assert training.model.load_in_4bit is True
    assert training.data.max_seq_len >= 64

    teachers = load_teachers_config(ROOT / "configs" / "teachers.yaml")
    assert len(teachers.models()) == 2
    # The shipped template must not contain a hard-coded endpoint or key.
    assert teachers.unresolved_refs, "template should require env-provided values"
    with pytest.raises(ConfigError):
        teachers.require_ready()


def test_teacher_config_rejects_guessed_model_id(tmp_path: Path):
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: https://api.invalid/v1
          api_key_env: TEST_KEY
        teachers:
          - name: t1
            model: ""
        """,
    )
    cfg = load_teachers_config(path)
    with pytest.raises(ConfigError, match="no resolved model id"):
        cfg.require_ready()


def test_teacher_config_rejects_lowercase_key_env(tmp_path: Path):
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: https://api.invalid/v1
          api_key_env: my_key
        teachers:
          - name: t1
            model: model-x
        """,
    )
    with pytest.raises(ConfigError, match="upper-case"):
        load_teachers_config(path)


def test_teacher_config_rejects_credentials_in_headers(tmp_path: Path):
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: https://api.invalid/v1
          api_key_env: TEST_KEY
          extra_headers:
            Authorization: Bearer demo-value
        teachers:
          - name: t1
            model: model-x
        """,
    )
    with pytest.raises(ConfigError, match="must not contain credentials"):
        load_teachers_config(path)


def test_unknown_keys_are_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: https://api.invalid/v1
          api_key_env: TEST_KEY
          typo_field: 1
        teachers:
          - name: t1
            model: model-x
        """,
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_teachers_config(path)


def test_unsupported_api_style_is_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: https://api.invalid/v1
          api_key_env: TEST_KEY
          api_style: anthropic-messages
        teachers:
          - name: t1
            model: model-x
        """,
    )
    with pytest.raises(ConfigError, match="openai-chat"):
        load_teachers_config(path)


def test_env_reference_resolution(monkeypatch):
    monkeypatch.setenv("GTU_TEST_MODEL", "verified-model-id")
    resolved, missing = resolve_env_ref("env:GTU_TEST_MODEL")
    assert resolved == "verified-model-id"
    assert missing is None

    resolved, missing = resolve_env_ref("env:GTU_UNSET_MODEL")
    assert resolved == ""
    assert missing == "GTU_UNSET_MODEL"

    resolved, missing = resolve_env_ref("literal-model")
    assert (resolved, missing) == ("literal-model", None)


def test_teacher_config_resolves_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GTU_BASE", "https://api.invalid/v1")
    monkeypatch.setenv("GTU_M", "model-abc")
    path = _write(
        tmp_path,
        "teachers.yaml",
        """
        provider:
          base_url: "env:GTU_BASE"
          api_key_env: TEST_KEY
        teachers:
          - name: t1
            model: "env:GTU_M"
        """,
    )
    cfg = load_teachers_config(path)
    assert cfg.unresolved_refs == []
    cfg.require_ready()  # does not raise
    assert cfg.models()[0].model == "model-abc"


def test_api_key_is_only_read_from_environment(monkeypatch, provider: ProviderConfig):
    monkeypatch.delenv(provider.api_key_env, raising=False)
    assert provider.api_key() is None
    with pytest.raises(ConfigError):
        provider.require_api_key()

    monkeypatch.setenv(provider.api_key_env, "  secret-value  ")
    assert provider.api_key() == "secret-value"


def test_teacher_weight_validation():
    with pytest.raises(ConfigError):
        TeacherModelConfig(name="x", model="m", weight=0).validate()


def test_training_config_rejects_qlora_without_4bit(tmp_path: Path):
    path = _write(
        tmp_path,
        "training.yaml",
        """
        model:
          name_or_path: openai/gpt-oss-20b
          load_in_4bit: false
        optimization:
          method: qlora
        """,
    )
    with pytest.raises(ConfigError, match="load_in_4bit"):
        load_training_config(path)


def test_training_config_rejects_unimplemented_unsloth_flag(tmp_path: Path):
    path = _write(
        tmp_path,
        "training.yaml",
        """
        model:
          name_or_path: openai/gpt-oss-20b
          load_in_4bit: true
          unsloth_4bit: true
        optimization:
          method: qlora
        """,
    )
    with pytest.raises(ConfigError, match="unsloth_4bit"):
        load_training_config(path)


def test_training_config_rejects_unknown_top_level_key(tmp_path: Path):
    path = _write(
        tmp_path,
        "training.yaml",
        """
        model:
          name_or_path: x
        optimizasyon:
          method: lora
        """,
    )
    with pytest.raises(ConfigError, match="unknown top-level key"):
        load_training_config(path)


def test_quality_config_validation(tmp_path: Path):
    cfg = load_quality_config()
    assert 0 < cfg.valid_ratio < 1

    path = _write(
        tmp_path,
        "quality.yaml",
        """
        valid_ratio: 0.5
        test_ratio: 0.5
        """,
    )
    with pytest.raises(ConfigError, match="valid_ratio"):
        load_quality_config(path)


def test_provider_default_has_no_hardcoded_endpoint():
    assert ProviderConfig().base_url == ""
    cfg = TeachersConfig(provider=ProviderConfig(), teachers=[])
    with pytest.raises(ConfigError):
        cfg.validate()
