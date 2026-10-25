"""Data generation: prompt construction, safety aborts, resume, CLI guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FakePool, write_jsonl
from gokturk.config import ConfigError
from gokturk.data_generation import (
    GenerationAborted,
    SeedTask,
    build_answer_request,
    build_draft_request,
    generate,
    load_seeds,
    main,
    make_record_id,
    parse_draft,
)
from gokturk.teachers import TeacherAuthError, TeacherServerError

ROOT = Path(__file__).resolve().parents[1]
GOOD_DRAFT = json.dumps(
    {"prompt": "Türkiye'de yenilenebilir enerji yatırımlarının ekonomik etkilerini kısaca açıklar mısın?"},
    ensure_ascii=False,
)
GOOD_ANSWER = (
    "Yenilenebilir enerji yatırımları istihdam yaratır, enerji ithalatını azaltır ve "
    "uzun vadede maliyetleri düşürür. Bu etkiler bölgesel kalkınmayı da destekler."
)


def test_shipped_seeds_are_valid():
    seeds = load_seeds(ROOT / "data" / "seeds" / "turkish_seed_tasks.jsonl")
    assert len(seeds) >= 10
    assert {s.task_type for s in seeds} >= {"turkish_qa", "coding", "tool_calling"}


def test_seed_validation_rejects_bad_task_type(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "seeds.jsonl",
        [{"id": "s1", "task_type": "astrology", "topic": "x"}],
    )
    with pytest.raises(ConfigError, match="task_type"):
        load_seeds(path)


def test_seed_validation_rejects_duplicate_ids(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "seeds.jsonl",
        [
            {"id": "s1", "task_type": "coding", "topic": "x"},
            {"id": "s1", "task_type": "coding", "topic": "y"},
        ],
    )
    with pytest.raises(ConfigError, match="duplicate seed id"):
        load_seeds(path)


def test_draft_prompt_contains_no_secrets_and_requests_json():
    seed = SeedTask(id="s1", task_type="coding", topic="Python listeleri")
    system, user = build_draft_request(seed, variant=1)
    assert "JSON" in user
    assert "api_key" not in (system + user).lower()
    assert "Python listeleri" in user


def test_tool_calling_answers_request_strict_json():
    seed = SeedTask(id="s1", task_type="tool_calling", topic="hava durumu")
    _, user = build_answer_request(seed, "İstanbul için hava durumu nedir?")
    assert '"tool"' in user
    assert "JSON" in user


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"prompt": "Bu geçerli bir Türkçe istektir."}', "Bu geçerli bir Türkçe istektir."),
        ('```json\n{"prompt": "Kod bloğu içinde gelen istek."}\n```', "Kod bloğu içinde gelen istek."),
        ("Elbette: {\"prompt\": \"Açıklama ile gelen istek metni.\"}", "Açıklama ile gelen istek metni."),
        ('{"prompt": "kısa"}', None),
        ('{"baska": "alan"}', None),
        ("JSON yok burada", None),
        ("{bozuk json}", None),
    ],
)
def test_parse_draft(text, expected):
    assert parse_draft(text) == expected


def test_record_id_is_deterministic():
    a = make_record_id("seed-1", "mimo", 0, "coding")
    b = make_record_id("seed-1", "mimo", 0, "coding")
    c = make_record_id("seed-1", "glm", 0, "coding")
    assert a == b != c
    assert a.startswith("gtu-cod-")


def test_generate_happy_path(teachers_config, tmp_path: Path):
    seeds = [SeedTask(id="s1", task_type="turkish_qa", topic="enerji")]
    pool = FakePool(drafts=[GOOD_DRAFT], answers=[GOOD_ANSWER])
    out = tmp_path / "gen.jsonl"
    stats = generate(teachers_config, seeds, out_path=out, pool=pool)
    assert stats.accepted == 1 and stats.rejected == 0
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["prompt"].startswith("Türkiye'de yenilenebilir")
    assert record["source"]["teacher"] == "mimo"
    assert record["source"]["seed_id"] == "s1"
    assert "api_key" not in json.dumps(record).lower()


def test_bad_draft_is_rejected_not_repaired(teachers_config, tmp_path: Path):
    seeds = [SeedTask(id="s1", task_type="turkish_qa", topic="enerji")]
    pool = FakePool(drafts=["Bu JSON değil"], answers=[GOOD_ANSWER])
    stats = generate(teachers_config, seeds, out_path=tmp_path / "gen.jsonl", pool=pool)
    assert stats.accepted == 0
    assert stats.rejected == 1
    assert any("drafter" in r for r in stats.reject_reasons)


def test_low_quality_answer_is_rejected(teachers_config, tmp_path: Path):
    seeds = [SeedTask(id="s1", task_type="turkish_qa", topic="enerji")]
    pool = FakePool(drafts=[GOOD_DRAFT], answers=["Kısa."])
    stats = generate(teachers_config, seeds, out_path=tmp_path / "gen.jsonl", pool=pool)
    assert stats.accepted == 0
    assert stats.rejected == 1


def test_auth_error_aborts_generation(teachers_config, tmp_path: Path):
    class FailingPool(FakePool):
        def ask(self, system_prompt, user_prompt):
            raise TeacherAuthError("HTTP 401: invalid key")

    seeds = [SeedTask(id="s1", task_type="turkish_qa", topic="enerji")]
    with pytest.raises(GenerationAborted, match="TeacherAuthError"):
        generate(
            teachers_config,
            seeds,
            out_path=tmp_path / "gen.jsonl",
            pool=FailingPool([], []),
        )
    # nothing was written: no fabricated placeholder data
    assert not (tmp_path / "gen.jsonl").exists() or not (tmp_path / "gen.jsonl").read_text().strip()


def test_consecutive_failures_abort(teachers_config, tmp_path: Path):
    teachers_config.provider.abort_after_consecutive_failures = 2

    class FlakyPool(FakePool):
        def register_failure(self):
            self.failures = getattr(self, "failures", 0) + 1
            return self.failures

        def ask(self, system_prompt, user_prompt):
            raise TeacherServerError("HTTP 503")

    seeds = [
        SeedTask(id=f"s{i}", task_type="turkish_qa", topic="enerji") for i in range(5)
    ]
    with pytest.raises(GenerationAborted, match="ardışık"):
        generate(
            teachers_config,
            seeds,
            out_path=tmp_path / "gen.jsonl",
            pool=FlakyPool([], []),
        )


def test_resume_skips_existing_ids(teachers_config, tmp_path: Path):
    seeds = [SeedTask(id="s1", task_type="turkish_qa", topic="enerji", prompt="Sabit istek metni burada.")]
    out = tmp_path / "gen.jsonl"
    record = {
        "id": make_record_id("s1", "mimo", 0, "turkish_qa"),
        "prompt": "Sabit istek metni burada.",
        "response": GOOD_ANSWER,
        "source": {"kind": "teacher", "teacher": "mimo", "task_type": "turkish_qa", "seed_id": "s1"},
    }
    write_jsonl(out, [record])
    pool = FakePool(drafts=[], answers=[GOOD_ANSWER])
    stats = generate(teachers_config, seeds, out_path=out, pool=pool, resume=True)
    assert stats.skipped_existing == 1
    assert len(out.read_text().strip().splitlines()) == 1


def test_max_records_limit(teachers_config, tmp_path: Path):
    seeds = [
        SeedTask(id=f"s{i}", task_type="turkish_qa", topic="enerji", prompt="Sabit istek metni burada.")
        for i in range(5)
    ]
    pool = FakePool(drafts=[], answers=[GOOD_ANSWER] * 5)
    stats = generate(teachers_config, seeds, out_path=tmp_path / "gen.jsonl", pool=pool, max_records=2)
    assert stats.accepted == 2


# ---------------------------------------------------------------------------
# CLI guards
# ---------------------------------------------------------------------------
def test_cli_dry_run_works_without_credentials(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("EVREN_API_KEY", raising=False)
    monkeypatch.delenv("EVREN_BASE_URL", raising=False)
    out = tmp_path / "gen.jsonl"
    code = main(
        [
            "--teachers-config",
            str(ROOT / "configs" / "teachers.yaml"),
            "--seeds",
            str(ROOT / "data" / "seeds" / "turkish_seed_tasks.jsonl"),
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "planned_records" in captured.out
    assert not out.exists()
    assert "EVREN_API_KEY" in captured.err


def test_cli_list_models_requires_key(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("EVREN_API_KEY", raising=False)
    code = main(
        [
            "--teachers-config",
            str(ROOT / "configs" / "teachers.yaml"),
            "--seeds",
            str(ROOT / "data" / "seeds" / "turkish_seed_tasks.jsonl"),
            "--list-models",
        ]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "EVREN_API_KEY" in captured.err


def test_cli_plan_only_requires_pro_mode(monkeypatch, capsys):
    """--plan-only in basic mode used to fall through to the API-key gate."""
    monkeypatch.delenv("EVREN_API_KEY", raising=False)
    code = main(["--plan-only", "--target", "50"])
    captured = capsys.readouterr()
    assert code == 4
    assert "--mode pro" in captured.err
    assert "EVREN_API_KEY" not in captured.err


def test_cli_refuses_to_generate_without_key(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("EVREN_API_KEY", raising=False)
    code = main(
        [
            "--teachers-config",
            str(ROOT / "configs" / "teachers.yaml"),
            "--seeds",
            str(ROOT / "data" / "seeds" / "turkish_seed_tasks.jsonl"),
            "--out",
            str(tmp_path / "gen.jsonl"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "Durduruldu" in captured.err
    assert not (tmp_path / "gen.jsonl").exists()
