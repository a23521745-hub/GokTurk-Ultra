"""The Kaggle notebook must stay valid and must not leak secrets."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_training.ipynb"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)api[_-]?key\s*=\s*[\"'][^\"'\s]{8,}[\"']"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
]


@pytest.fixture(scope="module")
def notebook():
    if not NOTEBOOK.is_file():
        pytest.fail(f"notebook missing: {NOTEBOOK}")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_v4(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "notebook has no cells"


def test_code_cells_are_syntactically_valid_python(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            # IPython magics/shell escapes are not valid Python; neutralise them.
            lines.append(" " * (len(line) - len(stripped)) + "pass" if stripped.startswith(("%", "!")) else line)
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"cell {index} has a syntax error: {exc}")


def test_notebook_contains_no_credentials(notebook):
    raw = NOTEBOOK.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(raw), f"possible credential in notebook: {pattern.pattern}"


def test_notebook_covers_required_steps(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    required = [
        "check_environment",      # ortam denetimi
        "pip",                    # paket kurulumu
        "build_datasets",         # veri doğrulama
        "run_training",           # eğitim denemesi
        "smoke",                  # duman testi
        "run_manifest.json",      # çıktı kaydı
        "unsloth",                # 4-bit QLoRA yolu
        "preflight_unsloth",      # Unsloth ön denetimi
        "plan_only",              # sentetik veri planı
        "blocking",               # hata durumunda tanılama
        "list_models",            # sağlayıcının kendi model listesiyle doğrulama
        "push_adapter",           # isteğe bağlı HF yüklemesi (kapalı varsayılan)
        "plan_gguf",              # GGUF fizibilitesi (dönüşüm değil)
    ]
    for token in required:
        assert token in text, f"notebook is missing step: {token}"


def test_notebook_uses_verified_teacher_ids():
    """Teacher ids live in config; the notebook must not carry a dead env name."""
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert "EVREN_MODEL_KIMI" not in raw
    text = "".join("".join(c["source"]) for c in json.loads(raw)["cells"])
    # The ids are verified in configs/teachers.yaml, so only the key is needed.
    assert "yalnızca `EVREN_API_KEY`" in text
    assert "mimo-v2.6-pro" in text and "glm-5.3" in text


def test_notebook_publishing_is_off_by_default(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "RUN_PUBLISH = False" in text, "publishing must be opt-in"
    assert "HF_REPO_ID" in text
    assert "HF_TOKEN" in text


def test_notebook_does_not_claim_a_trained_model(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"]).lower()
    assert "kalite iddiası" in text or "kalite sonucu" in text
    for forbidden in ("doğruluk oranı %", "accuracy %", "benchmark skoru:"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# gerçek üretim hücresi (2.000 slot, süre bütçeli, devam ettirilebilir)
# ---------------------------------------------------------------------------
def test_notebook_has_real_generation_step(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])

    assert "generate_pro" in text, "plan-only ile kalırsa hiç veri üretilmez"
    assert "GENERATION_TARGET = 2000" in text
    assert "time_budget_s=TIME_BUDGET_MIN * 60" in text, "süre bütçesi geçirilmeli"
    assert "GenerationAborted" in text, "kota/oturum kesintisi yakalanmalı"
    # Resume: önceki oturumun çıktısı /kaggle/input'tan devralınır.
    assert 'rglob("pro.jsonl")' in text and "shutil.copy2" in text
    assert "resume=True" in text


def test_notebook_generation_never_fakes_data_without_a_key(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "EVREN_API_KEY" in text
    assert "sahte veri ÜRETMEZ" in text


def test_notebook_computes_training_steps_from_data(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "max_steps = 60" not in text, "sabit adım sayısı ezberlemeye yol açar"
    assert "_raw_steps" in text and "min(1500" in text


def test_production_runbook_is_linked(notebook):
    doc = ROOT / "docs" / "veri-uretimi-kaggle.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    for needle in ("2.000", "≈6,7 saat", "--time-budget-min", "stop_reason", "resume"):
        assert needle in body, f"runbook eksik: {needle}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "veri-uretimi-kaggle.md" in readme
