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
    assert "GENERATION_TARGET = 3" in text
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
    # Tek kaynak: gokturk.cot_dataset.training_steps_for (60-1500 arasına sıkıştırır).
    assert "training_steps_for" in text
    assert "run_cfg.optimization.max_steps = _steps" in text


def test_production_runbook_is_linked(notebook):
    doc = ROOT / "docs" / "veri-uretimi-kaggle.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    for needle in ("2.000", "≈6,7 saat", "--time-budget-min", "stop_reason", "resume"):
        assert needle in body, f"runbook eksik: {needle}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "veri-uretimi-kaggle.md" in readme


# ---------------------------------------------------------------------------
# 2026-10-26 olayı: ağ tanısı, sıfır-verim sigortası, ilerleme günlüğü
# ---------------------------------------------------------------------------
def test_notebook_diagnoses_network_before_generating(notebook):
    """Tanı mantığı artık paylaşılan `diagnose_endpoint` içinde (test: test_teachers)."""
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "diagnose_endpoint" in text, "DNS tanısı olmadan 6 saat boşa gidebilir"
    assert "NET_OK" in text
    library = (ROOT / "src" / "gokturk" / "teachers.py").read_text(encoding="utf-8")
    assert "getaddrinfo" in library and "create_connection" in library


def test_notebook_stops_on_zero_yield(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "ZERO_YIELD_ABORT_AFTER" in text
    assert "zero_yield_abort_after" in text
    assert "abort_reason" in text, "sessiz döngü yerine neden yazdırılmalı"
    assert "HİÇ KAYIT ÜRETİLMEDİ" in text


def test_notebook_writes_a_progress_log(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "progress.log" in text
    assert "FileHandler" in text


def test_notebook_has_single_call_probe(notebook):
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "6a) Öğretmen ölçümü" in text
    assert "for budget in PING_BUDGETS" in text
    assert "PING_BUDGETS = (128, 512, 2048, 4096, 8192)" in text


def test_incident_report_is_documented_and_linked():
    doc = ROOT / "docs" / "uretim-olay-2026-10-26.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    for needle in ("5,7 saat", "0 kayıt", "TeacherUnavailableError", "zero_yield_abort_after",
                   "MAX_ESCALATED_TOKENS", "health-check-failed"):
        assert needle in body, f"olay raporunda eksik: {needle}"
    assert "uretim-olay-2026-10-26.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uretim-olay-2026-10-26.md" in (
        ROOT / "docs" / "veri-uretimi-kaggle.md"
    ).read_text(encoding="utf-8")


def test_incident_report_states_the_remaining_limits():
    body = (ROOT / "docs" / "uretim-olay-2026-10-26.md").read_text(encoding="utf-8")
    # Düzeltmelerin neyi ÇÖZMEDİĞİ de yazılı olmalı.
    assert "DNS sorunu çözülmedi" in body
    assert "hâlâ tahmin" in body
    assert "gerçek uç nokta doğrulaması yapılmadı" in body


# ---------------------------------------------------------------------------
# 3b olayı: yanlış teşhis ("anahtar yok") bir daha basılmamalı
# ---------------------------------------------------------------------------
def _cell_with(needle: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and needle in "".join(cell["source"]):
            return "".join(cell["source"])
    raise AssertionError(f"hücre bulunamadı: {needle}")


def test_cell_3b_no_longer_blames_the_api_key_unconditionally():
    source = _cell_with("list_models")
    assert "describe_error" in source, "teşhis hatanın türüne göre üretilmeli"
    # Koşulsuz suçlama cümlesi kalkmalı: 2026-10-26'da anahtar VARDI, sorun DNS'ti.
    # (Metin yalnızca olayı anlatan YORUMDA kalabilir; bir print() içinde olamaz.)
    printed = [line for line in source.splitlines() if line.lstrip().startswith("print(")]
    assert not any("En sık sebep: EVREN_API_KEY" in line for line in printed)
    # "Anahtar yok" yalnızca gerçekten yoksa (diagnose_endpoint.key_present) yazılır.
    assert "elif not _diag.key_present:" in source
    assert source.index("elif not _diag.key_present:") < source.index("Anahtar yok")


def test_cell_3b_diagnoses_network_before_requesting_models():
    source = _cell_with("list_models")
    assert "diagnose_endpoint" in source
    assert source.index("diagnose_endpoint") < source.index("list_models()")


def test_cell_3b_is_declared_optional():
    source = _cell_with("list_models")
    assert "OPSİYONEL" in source


def test_cell_6b_uses_shared_diagnosis_instead_of_hand_rolled_sockets():
    source = _cell_with("RUN_GENERATION")
    assert "diagnose_endpoint" in source, "tanı mantığı tek yerde olmalı"
    assert "gethostbyname" not in source
    assert "create_connection" not in source


def test_incident_report_covers_the_3b_misdiagnosis():
    body = (ROOT / "docs" / "uretim-olay-2026-10-26.md").read_text(encoding="utf-8")
    assert "3b hücresi yanlış yeri suçladı" in body
    for needle in ("TeacherDnsError", "list_models", "diagnose_endpoint", "describe_error"):
        assert needle in body, f"3b bölümünde eksik: {needle}"


def test_teacher_probe_does_not_depend_on_a_later_cell():
    """6a, 6b'de tanımlanan NET_OK'a bakmamalı (2026-10-26: her zaman atlandı)."""
    source = _cell_with("PING_BUDGETS")
    code_lines = [ln for ln in source.splitlines() if not ln.strip().startswith("#")]
    assert not any("NET_OK" in ln for ln in code_lines), "NET_OK bir sonraki hücrede tanımlı"
    assert "diagnose_endpoint" in source
    assert "escalate_truncation=False" in source, "ölçüm yükseltmeyi kapatmalı"
    assert "128" in source and "512" in source


def test_probe_warns_when_any_teacher_has_no_content():
    import json
    from pathlib import Path

    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_training.ipynb").read_text())
    probe = next("".join(c["source"]) for c in notebook["cells"]
                 if c["cell_type"] == "code" and "PING_BUDGETS =" in "".join(c["source"]))
    assert "if len(verdict) != len(teachers.teachers):" in probe


def test_production_rejects_intermittent_dns_and_defaults_to_pilot(notebook):
    code = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "RUN_GENERATION =" in "".join(c["source"]))
    assert "not _diag.dns_intermittent" in code
    assert "GENERATION_TARGET = 3" in code
    assert "TIME_BUDGET_MIN = 15" in code
