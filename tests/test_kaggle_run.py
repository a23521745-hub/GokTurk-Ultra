"""Kaggle gerçek koşusu: disk kapısı, 4B profili ve düzeltilmiş yükleyici iddiası.

Korunan sözleşmeler:

* **Disk, VRAM'den önce gelir.** Kaggle'ın 19,2 GB boş alanında 14B'nin bf16
  indirmesi (≈29,6 GB) engelleyici, 9B (≈18 GB) uyarı, 4B (≈9,3 GB) temiz
  olmalı. Bu denetim olmadan "plan-only çalıştı" demek yanıltıcıydı.
* **"AutoProcessor gerekir" iddiası yanlıştı** ve bir daha kaynaklara,
  dokümanlara veya config başlıklarına sızmamalı: transformers
  `Qwen3_5ForCausalLM` görüntü kulesini `^model.visual.*` ile yok sayıyor.
* İlk gerçek koşu 4B ile yapılır ve `docs/kaggle-gercek-kosus.md` bunu adım adım
  anlatır; belge **birleştirme** önermez.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gokturk import env_check
from gokturk.config import load_training_config
from gokturk.env_check import GpuInfo, HardwareInfo, PackageInfo, check_environment
from gokturk.publish import BASE_PARAMS_B

ROOT = Path(__file__).resolve().parents[1]
CONFIG_4B = ROOT / "configs" / "training_qwen3_5_4b_t4.yaml"
RUN_DOC = ROOT / "docs" / "kaggle-gercek-kosus.md"

#: Kaggle'da ölçülen boş disk (docs/kaggle-rehberi.md §1).
KAGGLE_FREE_GB = 19.2


def _kaggle_t4(monkeypatch, disk_gb: float = KAGGLE_FREE_GB) -> None:
    monkeypatch.setattr(
        env_check,
        "detect_hardware",
        lambda *a, **k: HardwareInfo(
            python_version="3.11",
            platform="Linux",
            cpu_count=4,
            ram_gb=29.0,
            disk_free_gb=disk_gb,
            torch_available=True,
            torch_version="2.6.0",
            cuda_available=True,
            cuda_version="12.4",
            gpu_count=1,
            gpus=[GpuInfo(0, "Tesla T4", 15.6, 15.0, "7.5")],
            fp16_supported=True,
        ),
    )
    monkeypatch.setattr(
        env_check,
        "detect_packages",
        lambda: [
            PackageInfo(n, True, "1.0", "")
            for n in ("torch", "transformers", "peft", "bitsandbytes", "datasets")
        ],
    )


def _qlora(model: str):
    return check_environment(
        model_name=model, method="qlora", dtype="auto", load_in_4bit=True, seq_len=1024
    )


# ---------------------------------------------------------------------------
# disk kapısı
# ---------------------------------------------------------------------------
def test_14b_download_blocks_on_kaggle_disk(monkeypatch):
    _kaggle_t4(monkeypatch)
    report = _qlora("Qwen/Qwen3-14B")
    blocking = {f.code for f in report.blocking}
    assert "weights-disk-insufficient" in blocking
    assert report.ok is False, "29,6 GB indirme 19,2 GB diske sığmaz"
    finding = next(f for f in report.findings if f.code == "weights-disk-insufficient")
    assert "29" in finding.message and "19" in finding.message


def test_9b_download_is_tight_not_blocking(monkeypatch):
    _kaggle_t4(monkeypatch)
    report = _qlora("Qwen/Qwen3.5-9B")
    codes = {f.code for f in report.findings}
    assert "weights-disk-tight" in codes
    assert "weights-disk-insufficient" not in codes
    assert report.ok is True


def test_4b_download_fits_comfortably(monkeypatch):
    _kaggle_t4(monkeypatch)
    report = _qlora("Qwen/Qwen3.5-4B")
    codes = {f.code for f in report.findings}
    assert "weights-disk-tight" not in codes
    assert "weights-disk-insufficient" not in codes
    assert report.ok is True


def test_disk_gate_needs_a_known_disk_size(monkeypatch):
    """disk_free_gb bilinmiyorsa (0) yanlış engel üretilmemeli."""
    _kaggle_t4(monkeypatch, disk_gb=0.0)
    report = _qlora("Qwen/Qwen3-14B")
    assert "weights-disk-insufficient" not in {f.code for f in report.findings}


# ---------------------------------------------------------------------------
# 4B profili
# ---------------------------------------------------------------------------
def test_qwen3_5_4b_params_come_from_hf_api():
    facts = env_check.resolve_known_model("Qwen/Qwen3.5-4B")
    assert facts is not None
    # safetensors toplamı: 4_659_865_088
    assert facts["params"] == pytest.approx(4.66e9)
    assert facts["license"] == "Apache-2.0"
    assert facts["gated"] is False
    assert BASE_PARAMS_B["Qwen/Qwen3.5-4B"] == pytest.approx(4.66)


def test_4b_config_is_a_kaggle_qlora_profile():
    cfg = load_training_config(CONFIG_4B)
    assert cfg.model.name_or_path == "Qwen/Qwen3.5-4B"
    assert cfg.model.load_in_4bit is True
    assert cfg.optimization.method == "qlora"
    assert cfg.optimization.gradient_checkpointing is True
    assert cfg.data.max_seq_len == 1024, "disk payı için 1024'ten başlanır"
    assert cfg.optimization.max_steps <= 100
    assert cfg.require_preflight_ok is True, "gerçek koşuda ön denetim zorunlu"


def test_4b_config_adapter_is_llamacpp_convertible():
    """llama.cpp dönüştürücüsü embed_tokens/lm_head içeren adaptörü reddeder."""
    cfg = load_training_config(CONFIG_4B)
    assert not {"embed_tokens", "lm_head"} & set(cfg.lora.target_modules)
    assert cfg.optimization.method != "dora"


# ---------------------------------------------------------------------------
# düzeltilen iddia: AutoProcessor gerekmez
# ---------------------------------------------------------------------------
def test_autoprocessor_claim_is_gone_from_shipped_texts():
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "taban-model-secimi.md",
        ROOT / "src" / "gokturk" / "env_check.py",
        ROOT / "src" / "gokturk" / "serve.py",
        ROOT / "configs" / "training_qwen3_5_9b_t4.yaml",
        ROOT / "configs" / "training_qwen3_5_4b_t4.yaml",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "AutoProcessor ister" not in text, f"{path.name}: eski yanlış iddia duruyor"
        assert "AutoProcessor" + "` yolunun eklenmesi" not in text


def test_loader_evidence_is_recorded():
    """Düzeltme bir kaynağa dayanmalı: transformers'taki yok sayma kalıbı."""
    text = (ROOT / "src" / "gokturk" / "env_check.py").read_text(encoding="utf-8")
    assert "Qwen3_5ForCausalLM" in text
    assert "^model.visual.*" in text
    assert "modeling_qwen3_5.py" in text


# ---------------------------------------------------------------------------
# Unsloth füzyonlu kayıp hatası (Kaggle'da gerçekten alındı, 2026-10-25)
# ---------------------------------------------------------------------------
FUSED_LOSS_TRACEBACK = """
117.7s	  File "/usr/local/lib/python3.13/dist-packages/unsloth_zoo/cross_entropy_loss.py", line 506, in accumulate_chunk
117.7s	    (chunk_loss, (unscaled_loss,)) = torch.func.grad_and_value(
117.7s	RuntimeError: grad_and_value(f)(*args): Expected f(*args) to return a scalar Tensor, got tensor with 2 dims. Maybe you wanted to use the vjp or jacrev APIs instead?
"""


def test_fused_loss_traceback_is_recognised():
    from gokturk.logcheck import FAILED, analyze_log

    report = analyze_log(FUSED_LOSS_TRACEBACK)
    codes = {f["code"] for f in report["findings"]}
    assert "unsloth-fused-loss" in codes
    assert report["verdict"] == FAILED
    finding = next(f for f in report["findings"] if f["code"] == "unsloth-fused-loss")
    assert "--backend transformers" in finding["action"]
    assert "UNSLOTH_RETURN_LOGITS" in finding["action"]
    assert "average_tokens_across_devices" in finding["action"]


DATAPARALLEL_TRACEBACK = """
160.2s File "/usr/local/lib/python3.13/dist-packages/torch/nn/parallel/data_parallel.py", line 197, in forward
160.2s RuntimeError: Caught RuntimeError in replica 0 on device 0.
160.2s Original Traceback (most recent call last):
"""


def test_dataparallel_replica_error_is_recognised():
    from gokturk.logcheck import FAILED, analyze_log

    report = analyze_log(DATAPARALLEL_TRACEBACK)
    codes = {f["code"] for f in report["findings"]}
    assert "dataparallel-replica" in codes
    assert report["verdict"] == FAILED
    finding = next(f for f in report["findings"] if f["code"] == "dataparallel-replica")
    assert "CUDA_VISIBLE_DEVICES=0" in finding["action"]
    assert "4-bit" in finding["message"]


def test_notebook_clones_an_explicit_branch():
    """Dal açıkça yazılmalı ve klonlama --branch kullanmalı (varsayılana güvenmeyin)."""
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(cell["source"]) for cell in nb["cells"])
    assert "REPO_BRANCH" in text
    assert '"--branch", REPO_BRANCH' in text
    # dal bulunamazsa main'e düşerken uyarmalı
    assert "main çekiliyor" in text


def test_notebook_explains_missing_config_instead_of_crashing():
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "config_path.is_file()" in text
    assert "config yok:" in text
    assert "yanlış dalda" in text


def test_notebook_checks_internet_before_downloading():
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "huggingface.co" in text
    assert "Internet" in text


def test_run_doc_mentions_branch_and_internet():
    text = RUN_DOC.read_text(encoding="utf-8")
    assert "REPO_BRANCH" in text
    assert "Internet" in text


def test_notebook_pins_a_single_gpu_before_torch_import():
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in text
    # ilk kod hücresinde, torch import'undan önce olmalı
    import json

    nb = json.loads(text)
    first_code = next(c for c in nb["cells"] if c["cell_type"] == "code")
    source = "".join(first_code["source"])
    assert "CUDA_VISIBLE_DEVICES" in source
    # GPU'ları sorgulayan/import eden her satırdan önce olmalı
    assert source.index("CUDA_VISIBLE_DEVICES") < source.index("nvidia-smi")


def test_fused_loss_rule_does_not_fire_on_normal_logs():
    from gokturk.logcheck import analyze_log

    report = analyze_log("{'loss': 1.83, 'grad_norm': 0.6, 'learning_rate': 1e-4, 'epoch': 0.1}\n")
    assert "unsloth-fused-loss" not in {f["code"] for f in report["findings"]}


def test_unsloth_plain_loss_flag_sets_the_env_var(monkeypatch):
    """Bayrak, `import unsloth`'tan önce UNSLOTH_RETURN_LOGITS=1 ayarlamalı."""
    from gokturk.train import main

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    code = main(
        [
            "--config",
            str(CONFIG_4B),
            "--unsloth-plain-loss",
            "--plan-only",
        ]
    )
    assert code == 0
    import os

    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"


def test_unsloth_plain_loss_is_the_default_and_can_be_overridden():
    import inspect

    from gokturk.train import main, run_training

    signature = inspect.signature(run_training).parameters["unsloth_plain_loss"]
    assert signature.default is None, "None = arka uca göre karar ver"
    source = inspect.getsource(main)
    assert "--unsloth-plain-loss" in source
    assert "--unsloth-fused-loss" in source, "füzyonlu yolu istemek açık bir tercih olmalı"


def test_plain_loss_defaults_on_for_unsloth_backend(monkeypatch):
    """Füzyonlu kayıp iki kez çöktü: unsloth yolunda varsayılan düz kayıp."""
    import gokturk.train as train_mod

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    # Preflight'ta durdurmak için plan_only; env ayarı plan_only'den ÖNCE yapılır.
    code = train_mod.main(
        ["--config", str(CONFIG_4B), "--backend", "unsloth", "--plan-only"]
    )
    assert code in (0, 2)
    import os

    assert os.environ.get("UNSLOTH_RETURN_LOGITS") == "1"


def test_fused_loss_and_plain_loss_flags_conflict():
    from gokturk.train import main

    code = main(
        [
            "--config",
            str(CONFIG_4B),
            "--unsloth-plain-loss",
            "--unsloth-fused-loss",
            "--plan-only",
        ]
    )
    assert code == 4


def test_single_gpu_pin_is_respected_and_never_overridden(monkeypatch):
    import os

    from gokturk.train import pin_single_gpu

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_single_gpu() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert pin_single_gpu() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "açık tercih ezilmemeli"


def test_notebook_selects_profile_explicitly():
    """Notebook artık Unsloth'u otomatik seçmemeli: 4B profili varsayılan."""
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "RUN_PROFILE" in text
    assert "training_qwen3_5_4b_t4.yaml" in text
    assert 'if unsloth_report is not None and unsloth_report.ok' not in text, (
        "eski otomatik Unsloth seçimi geri gelmemeli"
    )


def test_run_doc_documents_the_fused_loss_failure():
    text = RUN_DOC.read_text(encoding="utf-8")
    assert "unsloth-fused-loss" in text
    assert "UNSLOTH_RETURN_LOGITS" in text
    assert "average_tokens_across_devices" in text
    assert "scalar Tensor" in text


# ---------------------------------------------------------------------------
# Kaggle çıktısından gelen düzeltmeler (2026-10-25, başarılı koşu sonrası)
# ---------------------------------------------------------------------------
def test_notebook_collects_hf_token_from_secrets():
    """Kullanıcı HF_TOKEN'ı girse de notebook onu hiç okumuyordu."""
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(cell["source"]) for cell in nb["cells"])
    assert 'SECRET_KEYS = ["EVREN_API_KEY", "HF_TOKEN"]' in text
    assert "HF_TOKEN:" in text, "token durumu ekrana yazılmalı"


def test_notebook_refuses_publish_without_repo_id():
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "repo_ok" in text
    assert "HF_REPO_ID boş" in text


def test_notebook_warns_about_tiny_training_set():
    """4 örnek × 60 epoch sessizce 'başarı' gibi görünüyordu."""
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "UYARI: eğitim seti yalnızca" in text
    assert "_train_rows < 100" in text
    assert "EZBERLER" in text


def test_notebook_uses_adapter_only_gguf_plan():
    """Eski hücre birleştirme planı basıyordu; doğru yol adaptör-only."""
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "plan_adapter_gguf" in text
    assert "render_plan" in text
    assert "birleştirme YOK" in text


def test_notebook_renders_logcheck():
    text = (ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8")
    assert "analyze_manifest" in text
    assert "render_logcheck" in text


# ---------------------------------------------------------------------------
# Bayat kod koruması (2026-10-25: lora_gguf import edilemedi, koşu çöpe gitti)
# ---------------------------------------------------------------------------
def test_notebook_defaults_to_main_and_refreshes_hard():
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'REPO_BRANCH = "main"' in text
    # `git pull --ff-only` sessizce eski kodda kalabiliyordu.
    assert '"reset", "-q", "--hard", "FETCH_HEAD"' in text
    assert "en son commit'e getiriliyor" in text


def test_notebook_checks_required_capabilities():
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "_CAPABILITIES" in text
    assert "src/gokturk/lora_gguf.py" in text
    assert "bu kopyada eksik yetenekler" in text
    # git pull sonrası eski modüller bellekte kalmasın
    assert "del sys.modules[_name]" in text


def test_optional_publish_cell_survives_missing_module():
    """İsteğe bağlı hücre, import hatası yüzünden koşuyu bitirmemeli."""
    import json

    nb = json.loads((ROOT / "notebooks" / "kaggle_training.ipynb").read_text(encoding="utf-8"))
    cell = next(
        c for c in nb["cells"] if c["cell_type"] == "code" and "RUN_PUBLISH" in "".join(c["source"])
    )
    source = "".join(cell["source"])
    assert "except ImportError" in source
    assert "Adaptör-only GGUF planı bu kod sürümünde yok" in source
    assert "başarısız olsa da eğitim çıktıları korunur" in source
    # İçe aktarmalar try bloklarının içinde olmalı: satır başında çıplak import yok.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from gokturk") or stripped.startswith("import gokturk"):
            assert line.startswith("    "), f"korumasız import: {line!r}"


def test_epoch_explosion_is_flagged_in_manifest():
    from gokturk.logcheck import analyze_manifest

    manifest = {
        "kind": "training",
        "backend": "transformers",
        "resolved_dtype": "fp16",
        "data": {"train_examples": 4, "eval_examples": 1},
        "metrics": {"train_loss": 0.0777, "eval_eval_loss": 2.261, "epoch": 60.0},
    }
    report = analyze_manifest(manifest)
    codes = {f["code"] for f in report["findings"]}
    assert "epoch-explosion" in codes
    finding = next(f for f in report["findings"] if f["code"] == "epoch-explosion")
    assert "ezberleme" in finding["message"]
    assert "600–1200" in finding["action"]


def test_normal_epoch_count_is_not_flagged():
    from gokturk.logcheck import analyze_manifest

    manifest = {
        "kind": "training",
        "backend": "transformers",
        "data": {"train_examples": 9000, "eval_examples": 500},
        "metrics": {"train_loss": 1.2, "eval_eval_loss": 1.25, "epoch": 1.4},
    }
    assert "epoch-explosion" not in {f["code"] for f in analyze_manifest(manifest)["findings"]}


# ---------------------------------------------------------------------------
# koşu belgesi
# ---------------------------------------------------------------------------
def test_run_checklist_doc_covers_the_required_steps():
    text = RUN_DOC.read_text(encoding="utf-8")
    for needle in (
        "df -h",
        "--plan-only",
        "--max-steps 60",
        "gokturk.logcheck",
        "Qwen3_5ForCausalLM",
        "model.visual.*",
        "gokturk.publish --adapter",
        "--serve-plan",
    ):
        assert needle in text, f"eksik adım: {needle}"
    assert "--merge" not in text.split("**Yapılmayacak:**")[0]
    assert "Yapılmayacak" in text, "birleştirmenin yapılmayacağı açıkça yazmalı"


def test_run_doc_is_linked_from_readme_and_model_selection():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    selection = (ROOT / "docs" / "taban-model-secimi.md").read_text(encoding="utf-8")
    assert "kaggle-gercek-kosus.md" in readme
    assert "kaggle-gercek-kosus.md" in selection
