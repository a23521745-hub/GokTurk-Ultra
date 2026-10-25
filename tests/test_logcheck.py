"""Log/manifest reader: must be useful, and must not cry wolf on healthy runs."""

from __future__ import annotations

import json

import pytest

from gokturk.logcheck import (
    BROKEN,
    FAILED,
    OK,
    WARN,
    analyze_log,
    analyze_manifest,
    main,
    parse_log,
    render,
)

HEALTHY_LOG = """
trainable params: 12,582,912 || all params: 21,000,000,000 || trainable%: 0.06
  0%|          | 0/60 [00:00<?, ?it/s]
{'loss': 2.4132, 'grad_norm': 0.812, 'learning_rate': 1.1e-06, 'epoch': 0.01}
 10%|█         | 6/60 [01:04<09:30,  9.46s/it]{'loss': 1.8342, 'grad_norm': 0.612, 'learning_rate': 4.8e-05, 'epoch': 0.05}
 20%|██        | 12/60 [02:08<08:32, 10.68s/it]{'loss': 1.5011, 'grad_norm': 0.701, 'learning_rate': 4.6e-05, 'epoch': 0.10}
 30%|███       | 18/60 [03:12<07:28, 10.65s/it]{'loss': 1.2980, 'grad_norm': 0.580, 'learning_rate': 4.0e-05, 'epoch': 0.15}
 40%|████      | 24/60 [04:16<06:24, 10.66s/it]{'loss': 1.1121, 'grad_norm': 0.512, 'learning_rate': 3.2e-05, 'epoch': 0.20}
 50%|█████     | 30/60 [05:20<05:20, 10.66s/it]{'loss': 1.0034, 'grad_norm': 0.498, 'learning_rate': 2.5e-05, 'epoch': 0.25}
 60%|██████    | 36/60 [06:24<04:16, 10.66s/it]{'loss': 0.9120, 'grad_norm': 0.455, 'learning_rate': 1.8e-05, 'epoch': 0.30}
 70%|███████   | 42/60 [07:28<03:12, 10.66s/it]{'eval_loss': 1.088, 'eval_runtime': 12.4}
 80%|████████  | 48/60 [08:32<02:08, 10.66s/it]{'loss': 0.8511, 'grad_norm': 0.401, 'learning_rate': 8.0e-06, 'epoch': 0.40}
100%|██████████| 60/60 [10:40<00:00, 10.66s/it]
{'train_runtime': 640.2, 'train_loss': 0.9143, 'epoch': 0.5}
"""

DIVERGING_LOG = """
 10%|█         | 6/60 [01:04<09:30, 9.46s/it]{'loss': 1.5011, 'grad_norm': 4.2, 'learning_rate': 5e-05, 'epoch': 0.05}
 20%|██        | 12/60 [02:08<08:32, 10.68s/it]{'loss': 3.2980, 'grad_norm': 18.3, 'learning_rate': 5e-05, 'epoch': 0.10}
 30%|███       | 18/60 [03:12<07:28, 10.65s/it]{'loss': 7.1121, 'grad_norm': 42.9, 'learning_rate': 5e-05, 'epoch': 0.15}
"""

NaN_LOG = """
{'loss': 1.5011, 'grad_norm': 0.5, 'learning_rate': 5e-05, 'epoch': 0.1}
{'loss': nan, 'grad_norm': nan, 'learning_rate': 5e-05, 'epoch': 0.2}
"""

OOM_LOG = """Traceback (most recent call last):
  File "train.py", line 812, in run_training
    trainer.train()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

AUTH_LOG = """requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://api.invalid/v1/chat/completions
Traceback (most recent call last):
"""

MISSING_FILE_LOG = """Traceback (most recent call last):
  File "train.py", line 140, in <module>
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/train.jsonl'
"""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_extracts_steps_losses_and_progress():
    stats = parse_log(HEALTHY_LOG)
    assert stats.steps == 8
    assert stats.losses[0] == pytest.approx(2.4132)
    assert stats.losses[-1] == pytest.approx(0.8511)
    assert stats.progress == (60, 60)
    assert stats.eval_losses == [pytest.approx(1.088)]
    assert stats.final_train_loss == pytest.approx(0.9143)
    assert stats.grad_norms and stats.learning_rates


def test_parse_handles_nan_literals():
    stats = parse_log(NaN_LOG)
    assert stats.steps == 2
    assert stats.losses[-1] != stats.losses[-1]  # nan


def test_parse_on_garbage_returns_empty_stats():
    stats = parse_log("hiç sayı yok burada")
    assert stats.steps == 0
    assert stats.losses == []


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def test_healthy_log_is_reported_healthy():
    report = analyze_log(HEALTHY_LOG)
    assert report["verdict"] == OK
    codes = {f["code"] for f in report["findings"]}
    assert "loss-decreasing" in codes
    assert "auth" not in codes  # 0.401 must not be read as HTTP 401
    assert "divergence" not in codes


def test_diverging_loss_is_broken():
    report = analyze_log(DIVERGING_LOG)
    assert report["verdict"] == BROKEN
    findings = {f["code"]: f for f in report["findings"]}
    assert "divergence" in findings
    assert "erken durdurun" in findings["divergence"]["action"].lower()


def test_nan_loss_is_broken_and_not_duplicated():
    report = analyze_log(NaN_LOG)
    assert report["verdict"] == BROKEN
    nan_findings = [f for f in report["findings"] if f["code"] == "nan-loss"]
    assert len(nan_findings) == 1


def test_cuda_oom_is_failed_with_memory_advice():
    report = analyze_log(OOM_LOG)
    assert report["verdict"] == FAILED
    codes = {f["code"] for f in report["findings"]}
    assert "cuda-oom" in codes
    # A CUDA OOM must never be misreported as host-RAM exhaustion.
    assert "ram-oom" not in codes
    action = next(f["action"] for f in report["findings"] if f["code"] == "cuda-oom")
    assert "max_seq_len" in action


def test_auth_error_is_failed_and_mentions_secrets():
    report = analyze_log(AUTH_LOG)
    assert report["verdict"] == FAILED
    findings = {f["code"]: f for f in report["findings"]}
    assert "auth" in findings
    assert "Secrets" in findings["auth"]["action"]


def test_missing_data_file_is_failed_with_path_hint():
    report = analyze_log(MISSING_FILE_LOG)
    assert report["verdict"] == FAILED
    findings = {f["code"]: f for f in report["findings"]}
    assert "missing-file" in findings
    assert "/kaggle/input" in findings["missing-file"]["action"]


def test_short_but_increasing_curve_is_caught():
    report = analyze_log(DIVERGING_LOG)
    assert any(f["code"] == "divergence" for f in report["findings"])


def test_unfinished_progress_is_flagged_as_warning():
    log = DIVERGING_LOG  # 60 adımlık koşunun 18'i
    report = analyze_log(log)
    assert report["verdict"] in (BROKEN, FAILED)  # ıraksama daha ağır basar
    text = render(report)
    assert "KARAR" in text


def test_report_is_json_serialisable_and_states_its_limits():
    report = analyze_log(HEALTHY_LOG)
    payload = json.dumps(report, ensure_ascii=False)
    assert "kalite" in payload.lower()
    assert "MEKANİĞİNİ" in report["note"]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _manifest(**overrides):
    base = {
        "kind": "training",
        "backend": "unsloth",
        "resolved_dtype": "fp16",
        "duration_s": 640.0,
        "data": {"train_examples": 9600, "eval_examples": 200},
        "metrics": {"train_loss": 0.91, "eval_loss": 0.95},
        "preflight": {"ok": True},
    }
    base.update(overrides)
    return base


def test_manifest_healthy():
    report = analyze_manifest(_manifest())
    assert report["verdict"] == OK
    assert report["run"]["backend"] == "unsloth"


def test_manifest_flags_smoke_runs():
    report = analyze_manifest(_manifest(kind="smoke", note="Duman testi: kalite iddiası yok."))
    assert report["verdict"] == WARN
    assert any(f["code"] == "smoke-run" for f in report["findings"])


def test_manifest_flags_eval_gap():
    report = analyze_manifest(_manifest(metrics={"train_loss": 0.5, "eval_loss": 1.4}))
    assert any(f["code"] == "eval-gap" for f in report["findings"])


def test_manifest_flags_tiny_dataset():
    report = analyze_manifest(_manifest(data={"train_examples": 12}))
    assert any(f["code"] == "tiny-dataset" for f in report["findings"])


def test_manifest_flags_failed_preflight():
    report = analyze_manifest(_manifest(preflight={"ok": False}))
    assert report["verdict"] == FAILED


def test_manifest_without_metrics_warns():
    report = analyze_manifest(_manifest(metrics={}))
    assert any(f["code"] == "no-loss" for f in report["findings"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_log_mode_exit_codes(tmp_path, capsys):
    healthy = tmp_path / "healthy.log"
    healthy.write_text(HEALTHY_LOG, encoding="utf-8")
    assert main(["--log", str(healthy)]) == 0
    assert "SAĞLIKLI" in capsys.readouterr().out

    failed = tmp_path / "failed.log"
    failed.write_text(OOM_LOG, encoding="utf-8")
    assert main(["--log", str(failed)]) == 1
    assert "BAŞARISIZ" in capsys.readouterr().out


def test_cli_fail_on_warning(tmp_path, capsys):
    log = tmp_path / "flat.log"
    log.write_text(
        "\n".join(
            f"{{'loss': 2.{i:04d}, 'grad_norm': 0.5, 'learning_rate': 5e-05, 'epoch': 0.1}}"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    assert main(["--log", str(log)]) == 0  # uyarı varsayılan olarak çıkış kodunu değiştirmez
    assert main(["--log", str(log), "--fail-on", "warning"]) == 1


def test_cli_manifest_mode(tmp_path, capsys):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert main(["--manifest", str(path)]) == 0
    assert "unsloth" in capsys.readouterr().out


def test_cli_missing_file_and_bad_json(tmp_path, capsys):
    assert main(["--log", str(tmp_path / "yok.log")]) == 4
    bad = tmp_path / "bad.json"
    bad.write_text("{bozuk", encoding="utf-8")
    assert main(["--manifest", str(bad)]) == 4


def test_cli_requires_an_input(capsys):
    with pytest.raises(SystemExit):
        main([])
