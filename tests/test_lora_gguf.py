"""Adaptör-only GGUF planı (llama.cpp LoRA) — birleştirmesiz yol.

Korunan sözleşmeler:

* **Birleştirme yok.** Adaptör tek başına çevrilir; bu yol ≈2 GB RAM ister,
  ``publish.plan_gguf`` ise ≈67 GB ister. İkisi karıştırılmamalı.
* llama.cpp mimari eşlemesi (qwen3_5 → QWEN35) ve ``--base-model-id`` kullanımı
  doğru; ``--base`` bir dizin beklediği için kullanılmaz.
* llama.cpp'nin reddettiği adaptörler (DoRA, embed_tokens/lm_head,
  modules_to_save) erkenden engellenir — ağa çıkmadan.
* ``--run`` yalnızca llama.cpp kopyası verildiğinde ve plan uygunsa çalışır.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gokturk import lora_gguf
from gokturk.lora_gguf import (
    ARCH_SUPPORT,
    AdapterGGUFError,
    check_adapter_compatibility,
    detect_arch,
    plan_adapter_gguf,
    read_adapter_config,
    run_conversion,
)

ROOT = Path(__file__).resolve().parents[1]

GOOD_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def make_adapter(tmp_path: Path, **overrides) -> Path:
    """Create a minimal but realistic PEFT adapter directory."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "r": 16,
        "lora_alpha": 32,
        "use_dora": False,
        "target_modules": GOOD_TARGETS,
    }
    config.update(overrides)
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    # 2 MB "weights": enough for size maths, no real tensors needed.
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 2_000_000)
    return adapter


# ---------------------------------------------------------------------------
# mimari eşlemesi
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("Qwen/Qwen3.5-4B", "qwen3_5"),
        ("Qwen/Qwen3.5-9B", "qwen3_5"),
        ("Qwen/Qwen3.5-35B-A3B", "qwen3_5_moe"),
        ("Qwen/Qwen3-14B", "qwen3"),
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("openai/gpt-oss-20b", "gpt_oss"),
        ("some/unknown-model", None),
    ],
)
def test_detect_arch(model, expected):
    assert detect_arch(model) == expected


def test_qwen3_5_is_fully_supported_by_llamacpp():
    entry = ARCH_SUPPORT["qwen3_5"]
    assert entry["llama_arch"] == "QWEN35"
    assert entry["conversion"] and entry["runtime"] and entry["lora"]
    assert "Qwen3_5ForCausalLM" in entry["hf_classes"]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def test_plan_is_feasible_without_merging(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path),
        base_model="Qwen/Qwen3.5-4B",
        out_dir=str(tmp_path / "gguf"),
        ram_gb=29.0,
        disk_gb=20.0,
    )
    assert plan["feasible"] is True
    assert plan["merge"] is False and plan["kind"] == "adapter-only"
    assert plan["llama_arch"] == "QWEN35"
    assert plan["base_gguf_repo"] == "unsloth/Qwen3.5-4B-GGUF"
    # Birleştirme yolunun ≈67 GB RAM istediği yerde bu yol ≈2 GB ister.
    assert plan["needed_ram_gb"] <= 2.5
    assert plan["needed_disk_gb"] < 10.0


def test_plan_convert_step_uses_base_model_id_not_a_directory(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path),
        base_model="Qwen/Qwen3.5-4B",
        out_dir=str(tmp_path / "gguf"),
        ram_gb=16.0,
        disk_gb=50.0,
    )
    convert = next(s for s in plan["steps"] if "ADAPTÖRÜ" in s["what"])
    command = [str(part) for part in convert["command"]]
    assert "--base-model-id" in command
    assert "--base" not in command, "--base dizin bekler; HF kimliğiyle çalışmaz"
    assert command[command.index("--outtype") + 1] == "f16"
    assert plan["adapter_gguf_path"].endswith(".gguf")


def test_plan_refuses_tiny_disk(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path),
        base_model="Qwen/Qwen3.5-4B",
        out_dir=str(tmp_path / "gguf"),
        ram_gb=29.0,
        disk_gb=1.5,
    )
    assert plan["feasible"] is False
    assert any("Disk yetersiz" in p for p in plan["problems"])


def test_plan_refuses_base_mismatch(tmp_path):
    adapter = make_adapter(tmp_path, base_model_name_or_path="Qwen/Qwen3.5-9B")
    plan = plan_adapter_gguf(
        adapter, base_model="Qwen/Qwen3.5-4B", ram_gb=32.0, disk_gb=50.0
    )
    assert plan["feasible"] is False
    assert any("için eğitilmiş" in p for p in plan["problems"])


def test_plan_refuses_unknown_architecture(tmp_path):
    adapter = make_adapter(tmp_path, base_model_name_or_path="acme/unknown-7b")
    plan = plan_adapter_gguf(
        adapter, base_model="acme/unknown-7b", ram_gb=32.0, disk_gb=50.0
    )
    assert plan["feasible"] is False
    assert any("mimari eşlemesi bilinmiyor" in p for p in plan["problems"])


def test_plan_reports_measured_adapter_size(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path), base_model="Qwen/Qwen3.5-4B", ram_gb=32.0, disk_gb=50.0
    )
    assert plan["adapter_mb"] == pytest.approx(2.0, abs=0.2)
    assert plan["adapter_rank"] == 16


# ---------------------------------------------------------------------------
# llama.cpp'nin reddettikleri
# ---------------------------------------------------------------------------
def test_dora_is_refused(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path, use_dora=True),
        base_model="Qwen/Qwen3.5-4B",
        ram_gb=32.0,
        disk_gb=50.0,
    )
    assert plan["feasible"] is False
    assert any("DoRA" in p for p in plan["problems"])


def test_embedding_targets_are_refused(tmp_path):
    problems = check_adapter_compatibility(
        tmp_path, {"target_modules": ["q_proj", "embed_tokens", "lm_head"]}
    )
    assert any("gömme/başlık" in p for p in problems)


def test_modules_to_save_is_refused(tmp_path):
    problems = check_adapter_compatibility(
        tmp_path, {"target_modules": GOOD_TARGETS, "modules_to_save": ["lm_head"]}
    )
    assert any("modules_to_save" in p for p in problems)


def test_our_trained_targets_pass(tmp_path):
    assert check_adapter_compatibility(tmp_path, {"target_modules": GOOD_TARGETS}) == []


# ---------------------------------------------------------------------------
# hata yolları ve --run
# ---------------------------------------------------------------------------
def test_missing_adapter_config_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AdapterGGUFError, match="adapter_config.json yok"):
        read_adapter_config(empty)


def test_missing_weights_raises(tmp_path):
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AdapterGGUFError, match="adapter_model"):
        plan_adapter_gguf(adapter, base_model="Qwen/Qwen3.5-4B", ram_gb=32.0, disk_gb=50.0)


def test_run_conversion_refuses_infeasible_plan(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path),
        base_model="Qwen/Qwen3.5-4B",
        ram_gb=32.0,
        disk_gb=0.5,
    )
    with pytest.raises(AdapterGGUFError, match="engelleyici"):
        run_conversion(plan, llama_cpp_dir=tmp_path)


def test_run_conversion_requires_the_script(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path), base_model="Qwen/Qwen3.5-4B", ram_gb=32.0, disk_gb=50.0
    )
    with pytest.raises(AdapterGGUFError, match="convert_lora_to_gguf.py yok"):
        run_conversion(plan, llama_cpp_dir=tmp_path / "llama.cpp")


def test_run_conversion_executes_and_verifies_output(tmp_path):
    """Sahte bir dönüştürücüyle uçtan uca: komut satırı ve çıktı denetimi."""
    adapter = make_adapter(tmp_path)
    out_dir = tmp_path / "gguf"
    plan = plan_adapter_gguf(
        adapter,
        base_model="Qwen/Qwen3.5-4B",
        out_dir=str(out_dir),
        ram_gb=32.0,
        disk_gb=50.0,
    )
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_lora_to_gguf.py").write_text(
        "import sys\n"
        "args = sys.argv[1:]\n"
        "assert '--base-model-id' in args, args\n"
        "assert '--base' not in args, args\n"
        "out = args[args.index('--outfile') + 1]\n"
        "open(out, 'wb').write(b'GGUF' + b'\\0' * 64)\n",
        encoding="utf-8",
    )
    written = run_conversion(plan, llama_cpp_dir=llama)
    assert written.is_file()
    assert written.read_bytes().startswith(b"GGUF")


def test_run_conversion_reports_converter_failure(tmp_path):
    plan = plan_adapter_gguf(
        make_adapter(tmp_path), base_model="Qwen/Qwen3.5-4B", ram_gb=32.0, disk_gb=50.0
    )
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_lora_to_gguf.py").write_text(
        "# --base-model-id destekleniyor\n"
        "import sys\nsys.stderr.write('KeyError: unexpected tensor\\n')\nsys.exit(1)\n",
        encoding="utf-8",
    )
    with pytest.raises(AdapterGGUFError, match="başarısız"):
        run_conversion(plan, llama_cpp_dir=llama)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_prints_plan_and_returns_zero(tmp_path, capsys):
    code = lora_gguf.main(
        [
            "--adapter",
            str(make_adapter(tmp_path)),
            "--base-model",
            "Qwen/Qwen3.5-4B",
            "--ram-gb",
            "32",
            "--disk-gb",
            "50",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "YAPILABİLİR" in out
    assert "birleştirme YOK" in out


def test_cli_returns_two_when_infeasible(tmp_path, capsys):
    code = lora_gguf.main(
        [
            "--adapter",
            str(make_adapter(tmp_path, use_dora=True)),
            "--base-model",
            "Qwen/Qwen3.5-4B",
            "--ram-gb",
            "32",
            "--disk-gb",
            "50",
        ]
    )
    assert code == 2


def test_cli_returns_four_on_bad_adapter(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert lora_gguf.main(["--adapter", str(empty), "--base-model", "Qwen/Qwen3.5-4B"]) == 4


# ---------------------------------------------------------------------------
# belge ve README bağlantısı
# ---------------------------------------------------------------------------
def test_gguf_doc_exists_and_states_no_merge():
    doc = ROOT / "docs" / "gguf-donusumu.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    for needle in (
        "convert_lora_to_gguf.py",
        "--base-model-id",
        "QWEN35",
        "birleştirme YOK",
        "gokturk.lora_gguf",
    ):
        assert needle in text, f"eksik: {needle}"


def test_readme_links_the_gguf_doc():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "gguf-donusumu.md" in readme
