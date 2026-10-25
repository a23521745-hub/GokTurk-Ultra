"""Repository hygiene: required files, ignore rules, no committed secrets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    "configs/training.yaml",
    "configs/teachers.yaml",
    "docs/architecture.md",
    "notebooks/kaggle_training.ipynb",
    "src/gokturk/teachers.py",
    "src/gokturk/data_generation.py",
    "src/gokturk/data_quality.py",
    "src/gokturk/train.py",
    "src/gokturk/evaluate.py",
    "data/seeds/turkish_seed_tasks.jsonl",
    "data/eval/turkish_eval.jsonl",
]

# High-signal patterns only: generic "key = value" matches produce false
# positives on Python identifiers such as ``pad_token = tokenizer.eos_token``.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"'][A-Za-z0-9._-]{24,}[\"']"),
]

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".ipynb", ".toml", ".example"}


@pytest.mark.parametrize("relpath", REQUIRED_FILES)
def test_required_file_exists(relpath: str):
    assert (ROOT / relpath).is_file(), f"missing required file: {relpath}"


def test_env_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in ignore}
    assert ".env" in entries
    assert ".env.*" in entries
    assert "!.env.example" in entries
    # heavy artefacts must never be committed
    for pattern in ("outputs/", "data/raw/", "data/processed/", "*.safetensors"):
        assert pattern in entries, f"{pattern} should be ignored"


def test_env_example_has_no_values():
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        key, _, value = line.partition("=")
        assert value.strip() == "", f".env.example must not contain a value for {key}"


def test_no_secrets_committed():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "outputs", "node_modules"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} ({pattern.pattern})")
    assert not offenders, f"possible secrets: {offenders}"


def test_teacher_config_endpoint_is_the_verified_public_one():
    """The endpoint is pinned on purpose: it is EVREN's public gateway.

    Guessing is still forbidden -- the check below makes sure the only URLs in
    the file are the verified gateway and the documentation links cited next
    to it, and that no credential ever appears there.
    """
    text = (ROOT / "configs" / "teachers.yaml").read_text(encoding="utf-8")
    assert 'base_url: "https://evren-llmapi.ssyz.org.tr/v1"' in text
    urls = set(re.findall(r"https?://[^\s\"'<)]+", text))
    allowed = {
        "https://evren-llmapi.ssyz.org.tr/v1",
        "https://evren.ssyz.org.tr/llm-inference",
    }
    assert urls <= allowed, f"unexpected URL(s) in config: {sorted(urls - allowed)}"
    assert "api_key:" not in text and "sk-" not in text


def test_readme_documents_install_and_known_limitations():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install -e ." in readme
    assert "pytest" in readme
    assert "docs/architecture.md" in readme
    # honesty markers
    assert "eğitilmiş model ağırlığı" in readme.lower()


def test_sample_datasets_are_schema_valid():
    """Every committed sample file must pass the same gate as real data."""
    from gokturk.data_quality import assess_record, read_jsonl

    files = sorted((ROOT / "data" / "samples").glob("*.jsonl"))
    assert files, "no sample data files found"
    for path in files:
        records, issues = read_jsonl(path)
        assert not issues, f"{path.name}: parse errors {issues}"
        assert records, f"{path.name}: empty"
        for record in records:
            assert assess_record(record) == [], (path.name, record.get("id"))


def test_new_task_families_are_documented_in_configs():
    from gokturk.task_specs import family_ids

    training = (ROOT / "configs" / "training_unsloth_t4.yaml").read_text(encoding="utf-8")
    assert "load_in_4bit: true" in training
    assert "unsloth" in training.lower()
    # The Unsloth config must not silently claim a full run by default.
    assert "max_steps: 60" in training
    assert set(family_ids()) >= {"coding", "system_architecture", "turkish_advanced"}


def test_license_is_apache_2():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_pyproject_declares_core_without_heavy_deps():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "pyyaml" in core
    for heavy in ("torch", "transformers", "bitsandbytes"):
        assert heavy not in core, f"{heavy} must stay optional"
