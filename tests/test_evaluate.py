"""Evaluation harness: checks, scoring, honesty of unmeasured categories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_jsonl
from gokturk.config import ConfigError
from gokturk.evaluate import (
    CATEGORIES,
    EvalItem,
    format_check,
    load_eval_set,
    main,
    run_check,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_eval_set_is_valid_and_covers_all_categories():
    items = load_eval_set(ROOT / "data" / "eval" / "turkish_eval.jsonl")
    assert len(items) >= 12
    assert {i.category for i in items} == set(CATEGORIES)
    counts = {c: sum(1 for i in items if i.category == c) for c in CATEGORIES}
    assert all(n >= 3 for n in counts.values()), counts


def test_eval_item_rejects_unknown_check(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {
                "id": "x",
                "category": "coding",
                "prompt": "test",
                "checks": [{"type": "vibes_check"}],
            }
        ],
    )
    with pytest.raises(ConfigError, match="unknown check type"):
        load_eval_set(path)


def test_eval_item_rejects_bad_category(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "eval.jsonl",
        [{"id": "x", "category": "astrology", "prompt": "p", "checks": [{"type": "json_valid"}]}],
    )
    with pytest.raises(ConfigError, match="category"):
        load_eval_set(path)


@pytest.mark.parametrize(
    "check,response,expected",
    [
        ({"type": "contains_all", "value": ["a", "b"]}, "a ve b", True),
        ({"type": "contains_all", "value": ["a", "c"]}, "a ve b", False),
        ({"type": "contains_any", "value": ["x", "y"]}, "y burada", True),
        ({"type": "not_contains", "value": ["maliyet"]}, "gider düştü", True),
        ({"type": "not_contains", "value": ["maliyet"]}, "maliyet düştü", False),
        ({"type": "regex", "value": r"^\d{2}\.\d{2}$"}, "15.30", True),
        ({"type": "json_valid"}, '{"tool": "x", "arguments": {}}', True),
        ({"type": "json_valid"}, "hiç json yok", False),
        ({"type": "json_keys", "value": ["tool", "arguments"]}, '{"tool": "x", "arguments": {}}', True),
        ({"type": "json_keys", "value": ["tool", "arguments"]}, '{"tool": "x"}', False),
        ({"type": "min_words", "value": 3}, "bir iki üç", True),
        ({"type": "max_words", "value": 2}, "bir iki üç", False),
        ({"type": "min_lines", "value": 2}, "satır1\nsatır2", True),
        ({"type": "code_fence", "language": "python"}, "```python\nprint(1)\n```", True),
        ({"type": "code_fence", "language": "python"}, "```sql\nselect 1\n```", False),
        ({"type": "starts_with", "value": "Merhaba"}, "Merhaba dünya", True),
        ({"type": "ends_with", "value": "dünya"}, "Merhaba dünya", True),
    ],
)
def test_checks(check, response, expected):
    ok, detail = run_check(check, response)
    assert ok is expected, detail


def _items():
    return [
        EvalItem(id="a", category="coding", prompt="p", checks=[{"type": "contains_all", "value": ["def"]}]),
        EvalItem(id="b", category="coding", prompt="p", checks=[{"type": "contains_all", "value": ["zzz"]}]),
        EvalItem(id="c", category="tool_calling", prompt="p", checks=[{"type": "json_valid"}]),
    ]


def test_scoring_separates_pass_fail_and_missing():
    report = score_predictions(_items(), {"a": "def f():", "b": "başka metin"})
    assert report.n_items == 3
    assert report.n_scored == 2  # 'c' had no prediction
    assert report.overall_pass_rate == pytest.approx(0.5)

    by_cat = {c.category: c for c in report.categories}
    assert by_cat["coding"].n_items == 2
    assert by_cat["coding"].pass_rate == pytest.approx(0.5)
    # Unmeasured categories are None, never silently zero.
    assert by_cat["turkish_language"].pass_rate is None
    assert by_cat["turkish_language"].n_items == 0
    assert by_cat["tool_calling"].pass_rate == pytest.approx(0.0)

    missing = [i for i in report.items if i.missing_prediction]
    assert [i.id for i in missing] == ["c"]


def test_report_markdown_states_limitations():
    report = score_predictions(_items(), {"a": "def f():", "b": "x", "c": "{}"})
    md = report.render_markdown()
    assert "ölçülmedi" in md or "Sınırlamalar" in md
    assert "benchmark" in md.lower()
    assert "Sınırlamalar" in md


def test_empty_predictions_never_produce_a_fake_score():
    report = score_predictions(_items(), {})
    assert report.overall_pass_rate is None
    assert report.n_scored == 0


def test_format_check_reports_data_problems_not_quality(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "ok",
                "prompt": "Geçerli bir soru metni burada yer alıyor mu?",
                "response": "Evet, bu yanıt yeterince uzun ve anlamlı görünüyor.",
                "source": {"kind": "teacher", "teacher": "t"},
            },
            ensure_ascii=False,
        )
        + "\n"
        + '{"id": "bad"}\n'
        + "{bozuk\n",
        encoding="utf-8",
    )
    report = format_check(path)
    assert report["mode"] == "format"
    assert report["records"] == 2
    assert len(report["parse_errors"]) == 1
    assert report["records_with_problems"] == 1
    assert "model kalitesi" in report["note"]


def test_cli_score_writes_reports(tmp_path: Path, capsys):
    preds = write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {"id": "eval-tr-001", "response": "bağlaç ayrı yazılır, ek bitişik yazılır " + "kelime " * 50},
        ],
    )
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    code = main(
        [
            "--mode",
            "score",
            "--eval-set",
            str(ROOT / "data" / "eval" / "turkish_eval.jsonl"),
            "--predictions",
            str(preds),
            "--out",
            str(out),
            "--markdown",
            str(md),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_items"] >= 12
    assert payload["n_scored"] == 1
    assert md.read_text(encoding="utf-8").startswith("# Değerlendirme raporu")


def test_cli_requires_predictions(capsys):
    code = main(["--mode", "score"])
    assert code == 4
    assert "predictions" in capsys.readouterr().err
