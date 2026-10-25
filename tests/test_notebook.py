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
    ]
    for token in required:
        assert token in text, f"notebook is missing step: {token}"


def test_notebook_does_not_claim_a_trained_model(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"]).lower()
    assert "kalite iddiası" in text or "kalite sonucu" in text
    for forbidden in ("doğruluk oranı %", "accuracy %", "benchmark skoru:"):
        assert forbidden not in text
