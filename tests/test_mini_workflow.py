"""GitHub Actions mini-adaptör iş akışı ve CPU mini profili.

Bu testlerin koruduğu sözleşmeler:

* iş akışı **yalnızca elle** tetiklenir (her push'ta eğitim başlamaz),
* GPU olmadığını gizlemez: `--cpu-ok` kullanır, `--skip-preflight` ile
  engelleyici bulguları susturmaz,
* secret'lar yalnızca `secrets.*` üzerinden geçer, YAML içine gömülü anahtar yoktur,
* mini profil gerçekten CPU'da çalışabilir (fp32, 4-bit yok, bnb yok),
* `Qwen/Qwen3-0.6B` bilinen model tablosunda olduğu için ön denetim boyut
  tahminini yapabilir.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from gokturk import env_check
from gokturk.config import load_training_config
from gokturk.env_check import HardwareInfo, PackageInfo, check_environment

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "mini-adapter.yml"
CONFIG = ROOT / "configs" / "training_mini_cpu.yaml"

_SECRET_REF = re.compile(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _on_block(workflow: dict) -> dict:
    # YAML 1.1 `on` anahtarını boolean True yapar.
    return workflow.get("on", workflow.get(True))


def _steps(workflow: dict) -> list[dict]:
    return workflow["jobs"]["mini"]["steps"]


def _step_text(workflow: dict) -> str:
    return yaml.safe_dump(_steps(workflow), allow_unicode=True)


# ---------------------------------------------------------------------------
# iş akışı iskeleti
# ---------------------------------------------------------------------------
def test_workflow_is_manual_only(workflow):
    on = _on_block(workflow)
    assert list(on) == ["workflow_dispatch"], "her push'ta eğitim başlamamalı"


def test_workflow_inputs_are_documented(workflow):
    inputs = _on_block(workflow)["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "base_model",
        "max_steps",
        "train_file",
        "eval_file",
        "build_gguf_lora",
        "push_to_hub",
        "adapter_repo",
        "check_evren",
    }
    assert inputs["base_model"]["default"] == "Qwen/Qwen3-0.6B"
    # Yanlışlıkla büyük bir modele geçilirse varsayılan adım sayısı CPU'ya uygun.
    assert int(inputs["max_steps"]["default"]) <= 100
    assert inputs["push_to_hub"]["default"] == "false"
    assert inputs["build_gguf_lora"]["default"] == "true"


def test_workflow_runs_on_standard_cpu_runner(workflow):
    job = workflow["jobs"]["mini"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] <= 120
    assert workflow["permissions"]["contents"] == "read"


def test_workflow_claims_no_gpu_and_checks_it(workflow):
    text = _step_text(workflow)
    assert "nvidia-smi" in text, "donanım adımı GPU olup olmadığını yazmalı"
    assert "free -h" in text and "df -h" in text
    assert "GPU YOK" in text.upper()


def test_workflow_installs_cpu_torch_before_extras(workflow):
    """`.[train]` CUDA tekerleğini çekmesin: önce CPU indeksinden torch kurulur."""
    run_steps = [s for s in _steps(workflow) if "run" in s]
    install = next(s for s in run_steps if "pip install" in s["run"])
    assert install["run"].index("download.pytorch.org/whl/cpu") < install["run"].index('".[train')


# ---------------------------------------------------------------------------
# dürüstlük: GPU yokluğu gizlenmiyor
# ---------------------------------------------------------------------------
def test_workflow_uses_cpu_ok_not_skip_preflight(raw_workflow):
    assert "--cpu-ok" in raw_workflow
    assert "--skip-preflight" not in raw_workflow, (
        "CPU koşusu engelleyici bulguları sessizce yut mamalı; --cpu-ok kullanın"
    )
    assert "--plan-only" in raw_workflow, "eğitimden önce plan/ön denetim görünmeli"


def test_workflow_runs_logcheck_and_artifact_upload(workflow):
    text = _step_text(workflow)
    assert "gokturk.logcheck" in text
    assert "actions/upload-artifact" in text
    assert "outputs/mini/run_manifest.json" in text


def test_workflow_summary_states_this_is_not_gokturk_ultra(workflow):
    summary = next(s for s in _steps(workflow) if s.get("name") == "Özet")
    assert "GökTürk-Ultra değildir" in summary["run"]
    assert "GPU yok" in summary["run"]


def test_workflow_gguf_step_does_not_merge(workflow):
    step = next(s for s in _steps(workflow) if "GGUF" in s.get("name", ""))
    assert "convert_lora_to_gguf.py" in step["run"]
    assert "--merge" not in step["run"], "adaptör-only yol varsayılan olmalı"
    assert step["if"].startswith("${{ inputs.build_gguf_lora")


def test_workflow_gguf_step_uses_hub_id_not_local_dir(workflow):
    """`--base` bir dizin bekler; uzak taban için `--base-model-id` şarttır."""
    step = next(s for s in _steps(workflow) if "GGUF" in s.get("name", ""))
    run = step["run"]
    assert "--base-model-id" in run
    assert '--base "${{' not in run
    # torch==2.11.0'ı PyPI'dan isteyen kök requirements kurulmamalı (disk).
    assert "-r llama.cpp/requirements.txt" not in run
    assert "pip install gguf" in run


# ---------------------------------------------------------------------------
# secret hijyeni
# ---------------------------------------------------------------------------
def test_workflow_secrets_are_referenced_not_hardcoded(raw_workflow, workflow):
    referenced = set(_SECRET_REF.findall(raw_workflow))
    assert referenced == {"HF_TOKEN", "EVREN_API_KEY"}, f"beklenmeyen secret referansı: {referenced}"
    # Token görünümlü hiçbir sabit değer YAML'a gömülü olmamalı.
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", raw_workflow)
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", raw_workflow)
    for step in _steps(workflow):
        env = step.get("env") or {}
        for value in env.values():
            if isinstance(value, str):
                assert value.startswith("${{") or value.isupper() or "=" in value or value.isdigit()


def test_workflow_hub_push_fails_loudly_without_token(workflow):
    step = next(s for s in _steps(workflow) if "Hugging Face" in s.get("name", ""))
    assert "HF_TOKEN" in step["env"]
    assert "::error::" in step["run"]
    assert "exit 1" in step["run"]


def test_evren_health_job_is_opt_in_and_cheap(workflow):
    job = workflow["jobs"]["evren"]
    assert job["if"].startswith("${{ inputs.check_evren")
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] <= 15
    text = yaml.safe_dump(job["steps"], allow_unicode=True)
    # Yalnızca bağlantı doğrulaması: veri üretimi (--mode/--per-seed) yok.
    assert "--health-check" in text
    assert "--per-seed" not in text and "--mode" not in text
    assert "pip install -e ." in text, "torch/GPU bağımlılığı kurulmamalı"


# ---------------------------------------------------------------------------
# mini profil gerçekten CPU profili mi?
# ---------------------------------------------------------------------------
def test_mini_config_parses_and_is_cpu_shaped():
    cfg = load_training_config(CONFIG)
    assert cfg.model.name_or_path == "Qwen/Qwen3-0.6B"
    assert cfg.model.dtype == "fp32"
    assert cfg.model.load_in_4bit is False
    assert cfg.model.unsloth_4bit is False
    assert cfg.quantization.compute_dtype == "fp32"
    assert cfg.optimization.optim == "adamw_torch"
    assert cfg.optimization.gradient_checkpointing is False
    assert cfg.optimization.method == "lora"
    assert cfg.optimization.max_steps <= 100
    assert cfg.data.max_seq_len <= 512
    assert cfg.require_preflight_ok is False, "profil CPU'da bilinçli çalışır"


def test_mini_config_uses_tracked_sample_data():
    cfg = load_training_config(CONFIG)
    for rel in (cfg.data.train_file, cfg.data.eval_file):
        path = ROOT / rel
        assert path.is_file(), f"izlenmeyen veri yolu: {rel}"
        assert path.read_text(encoding="utf-8").strip(), f"boş veri dosyası: {rel}"


def test_mini_config_targets_lora_only_modules():
    """llama.cpp dönüştürücüsü embed_tokens/lm_head içeren adaptörü reddeder."""
    cfg = load_training_config(CONFIG)
    forbidden = {"embed_tokens", "lm_head"}
    assert not forbidden & set(cfg.lora.target_modules)


def test_qwen3_06b_is_a_known_model():
    facts = env_check.resolve_known_model("Qwen/Qwen3-0.6B")
    assert facts is not None
    assert facts["params"] == pytest.approx(0.6e9)
    assert facts["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# ön denetim: CPU mini kapısı
# ---------------------------------------------------------------------------
def _cpu_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        env_check,
        "detect_hardware",
        lambda *a, **k: HardwareInfo(
            python_version="3.11",
            platform="Linux",
            cpu_count=4,
            ram_gb=16.0,
            disk_free_gb=60.0,
            torch_available=True,
            torch_version="2.6.0",
            cuda_available=False,
        ),
    )
    monkeypatch.setattr(
        env_check,
        "detect_packages",
        lambda: [PackageInfo(n, True, "1.0", "") for n in ("torch", "transformers", "peft", "datasets")],
    )


def test_cpu_without_flag_blocks_with_no_gpu(monkeypatch):
    _cpu_runner(monkeypatch)
    report = check_environment(model_name="Qwen/Qwen3-0.6B", method="lora",
        dtype="fp32",
        load_in_4bit=False,
    )
    assert report.ok is False
    assert "no-gpu" in {f.code for f in report.blocking}


def test_cpu_mini_flag_downgrades_blocker_to_info(monkeypatch):
    _cpu_runner(monkeypatch)
    report = check_environment(
        model_name="Qwen/Qwen3-0.6B",
        method="lora",
        dtype="fp32",
        load_in_4bit=False,
        seq_len=512,
        cpu_ok=True,
        cpu_mini=True,
    )
    assert report.ok is True
    assert "no-gpu" not in {f.code for f in report.blocking}
    info = {f.code for f in report.findings if f.severity == env_check.SEVERITY_INFO}
    assert "cpu-mini" in info
    # Bilgi bulgusu, işin sınırını açıkça söyler.
    mini = next(f for f in report.findings if f.code == "cpu-mini")
    assert "GPU yok" in mini.message
    assert "kalite" in mini.message


def test_cpu_mini_warns_when_model_is_too_big(monkeypatch):
    _cpu_runner(monkeypatch)
    report = check_environment(
        model_name="Qwen/Qwen3.5-9B",
        method="lora",
        dtype="fp32",
        load_in_4bit=False,
        seq_len=512,
        cpu_ok=True,
        cpu_mini=True,
    )
    assert "cpu-mini-too-big" in {f.code for f in report.findings}


def test_cpu_mini_flag_does_not_mask_real_blockers(monkeypatch):
    """--cpu-ok yalnızca GPU yokluğunu bilgiye indirir; başka engeli silmez."""
    _cpu_runner(monkeypatch)
    report = check_environment(
        model_name="openai/gpt-oss-20b",
        method="full",
        dtype="bf16",
        load_in_4bit=False,
        cpu_ok=True,
        cpu_mini=True,
    )
    assert report.ok is False
    assert "gptoss-full-unsupported" in {f.code for f in report.blocking}


# ---------------------------------------------------------------------------
# CLI sözleşmesi
# ---------------------------------------------------------------------------
def test_cli_accepts_cpu_ok_and_model_override(capsys):
    from gokturk.train import main

    code = main(
        [
            "--config",
            str(CONFIG),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--cpu-ok",
            "--plan-only",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    plan_json = out.split("=== Plan ===")[1].split("\nUYARI:")[0]
    plan = json.loads(plan_json)
    assert plan["base_model"] == "Qwen/Qwen3-0.6B"
    assert plan["dtype"] == "fp32"
