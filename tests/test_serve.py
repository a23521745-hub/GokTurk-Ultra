"""No-merge serving: path ranking, Space scaffold, honest caveats."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gokturk.publish import PublishError
from gokturk.serve import (
    SERVING_PATHS,
    emit_space,
    main,
    plan_serving,
    render_plan,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# path table
# ---------------------------------------------------------------------------
def test_only_the_legacy_path_needs_a_merge():
    assert SERVING_PATHS["merged"]["merge"] is True
    for name in ("peft", "vllm-lora", "llamacpp"):
        assert SERVING_PATHS[name]["merge"] is False, name


def test_every_path_is_documented():
    for name, spec in SERVING_PATHS.items():
        assert spec["label"], name
        assert spec["needs"], name
        assert spec["note"], name


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def test_plan_includes_all_four_paths():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=48.0, ram_gb=64.0)
    assert [p["path"] for p in plan["paths"]] == ["peft", "vllm-lora", "llamacpp", "merged"]


def test_llamacpp_path_is_always_feasible_because_it_needs_no_merge():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=0.0, ram_gb=8.0)
    llama = next(p for p in plan["paths"] if p["path"] == "llamacpp")
    assert llama["feasible"] is True
    command = " ".join(" ".join(c) for c in llama["commands"])
    assert "convert_lora_to_gguf.py" in command
    assert "--lora" in command


def test_merged_path_is_refused_on_kaggle_sized_machine():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=0.0, ram_gb=29.0)
    merged = next(p for p in plan["paths"] if p["path"] == "merged")
    # 28.8 GB RAM would fit, but the ~46 GB disk does not; the plan must look
    # at the real machine and report both walls.
    assert merged["requirements_gb"]["disk"] > 20.0
    assert merged["requirements_gb"]["ram"] > 20.0


def test_vllm_caveats_mention_the_bnb_limitation():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=48.0, ram_gb=64.0)
    vllm = next(p for p in plan["paths"] if p["path"] == "vllm-lora")
    joined = " ".join(vllm["caveats"])
    assert "bitsandbytes" in joined
    assert "--qlora-adapter-name-or-path" in joined
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in joined


def test_llamacpp_caveats_mention_dora_and_embeddings():
    plan = plan_serving("Qwen/Qwen3.5-9B", ram_gb=32.0)
    llama = next(p for p in plan["paths"] if p["path"] == "llamacpp")
    joined = " ".join(llama["caveats"])
    assert "DoRA" in joined
    assert "embed_tokens" in joined


def test_plan_recommends_a_no_merge_path_when_gpu_is_present():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=48.0, ram_gb=64.0)
    assert plan["recommended"] in {"peft", "vllm-lora", "llamacpp"}
    assert plan["recommended"] != "merged"


def test_plan_rejects_unknown_base_model():
    with pytest.raises(PublishError, match="bilinmeyen taban model"):
        plan_serving("nobody/unknown-7b")


def test_plan_reports_adapter_size_when_given(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 100_000)
    plan = plan_serving("Qwen/Qwen3.5-9B", adapter_dir=adapter, ram_gb=32.0)
    assert plan["adapter_mb"] == pytest.approx(0.1, abs=0.01)


def test_render_plan_marks_feasibility_and_says_no_merge():
    plan = plan_serving("Qwen/Qwen3.5-9B", vram_gb=48.0, ram_gb=64.0)
    text = render_plan(plan)
    assert "merge gerekir mi: hayır" in text
    assert "merge gerekir mi: EVET" in text
    assert "ÖNERİLEN" in text
    assert "42 GB" in text  # the merge size it avoids


# ---------------------------------------------------------------------------
# Space scaffold
# ---------------------------------------------------------------------------
def test_emit_space_writes_three_files(tmp_path):
    written = emit_space(tmp_path / "space", adapter_repo="user/gokturk-lora")
    assert set(written) == {"app.py", "requirements.txt", "README.md"}
    for name in written:
        assert (tmp_path / "space" / name).is_file()


def test_generated_app_is_valid_python_and_loads_adapter_dynamically(tmp_path):
    emit_space(tmp_path, adapter_repo="user/gokturk-lora")
    source = (tmp_path / "app.py").read_text(encoding="utf-8")
    ast.parse(source)  # must be syntactically valid
    assert "PeftModel.from_pretrained" in source
    assert "merge_and_unload" not in source, "the scaffold must not merge"
    assert "user/gokturk-lora" in source
    assert "spaces.GPU" in source  # ZeroGPU support
    # The trained system message must match train.py.
    from gokturk.train import FALLBACK_SYSTEM

    assert FALLBACK_SYSTEM.split(":")[0] in source


def test_generated_readme_has_space_metadata(tmp_path):
    emit_space(tmp_path, adapter_repo="user/gokturk-lora", title="GökTürk Test")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "title: GökTürk Test" in readme
    assert "demo" in readme.lower()
    assert "benchmark iddiası yok" in readme


def test_emit_space_does_not_overwrite_by_default(tmp_path):
    emit_space(tmp_path, adapter_repo="user/a")
    app = tmp_path / "app.py"
    app.write_text("# elle düzenlendi\n", encoding="utf-8")
    emit_space(tmp_path, adapter_repo="user/a")
    assert app.read_text(encoding="utf-8") == "# elle düzenlendi\n"
    emit_space(tmp_path, adapter_repo="user/a", overwrite=True)
    assert "PeftModel" in app.read_text(encoding="utf-8")


def test_emit_space_rejects_bad_repo_id(tmp_path):
    with pytest.raises(PublishError, match="adapter_repo"):
        emit_space(tmp_path, adapter_repo="nodash")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_plan_exit_code_and_json(capsys):
    code = main(["--plan", "--base-model", "Qwen/Qwen3.5-9B", "--ram-gb", "64", "--vram-gb", "48", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["recommended"] != "merged"
    assert len(payload["paths"]) == 4


def test_cli_emit_space(tmp_path, capsys):
    code = main(["--emit-space", str(tmp_path), "--adapter-repo", "user/x"])
    out = capsys.readouterr().out
    assert code == 0
    assert "app.py" in out
    assert (tmp_path / "app.py").is_file()


def test_cli_without_action_prints_help(capsys):
    assert main([]) == 4
    assert "dinamik" in capsys.readouterr().out.lower()


def test_cli_rejects_unknown_base_model(capsys):
    assert main(["--plan", "--base-model", "nobody/x"]) == 4
    assert "bilinmeyen" in capsys.readouterr().err


def test_docs_explain_the_correction():
    """The old doc claimed GGUF always needs a merge; the correction must exist."""
    doc = (ROOT / "docs" / "servis-ve-dinamik-lora.md").read_text(encoding="utf-8")
    assert "convert_lora_to_gguf.py" in doc
    assert "429" not in doc  # no invented error codes
    assert "--lora" in doc
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in doc
    assert "bitsandbytes" in doc
