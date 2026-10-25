"""Professional pipeline: candidate selection, critique/repair, provenance, safety."""

from __future__ import annotations

import json

import pytest

from gokturk.config import QualityConfig
from gokturk.curriculum import build_plan
from gokturk.data_generation import (
    GenerationAborted,
    ProSettings,
    build_record_pro,
    generate_pro,
    parse_critique,
    plan_only,
)
from gokturk.task_specs import get_family
from gokturk.teachers import TeacherAuthError, TeacherRequestError, TeacherServerError

GOOD_DRAFT = json.dumps(
    {"prompt": "Python'da bir metindeki kelime sıklığını hesaplayan fonksiyon yaz ve örnekle göster."},
    ensure_ascii=False,
)

GOOD_ANSWER = (
    "Aşağıdaki fonksiyon `Counter` ile kelime sıklığını hesaplar.\n\n"
    "```python\nfrom collections import Counter\n\n\n"
    "def kelime_sikligi(metin: str) -> dict[str, int]:\n"
    "    return dict(Counter(metin.split()))\n```\n\n"
    "Örnek: `kelime_sikligi('a b a')` sonucu `{'a': 2, 'b': 1}` olur. "
    "Boş metin için boş sözlük döner; çok büyük dosyalarda satır satır okuma "
    "tercih edilmelidir çünkü tüm içeriği belleğe almak maliyetlidir."
)

WEAK_ANSWER = "Kod yazabilirim ama şu an yanıt veremiyorum."

CLEAN_CRITIQUE = '{"sorunlar": [], "duzeltilmis_yanit": ""}'
REPAIR_CRITIQUE = json.dumps(
    {"sorunlar": ["Kenar durumlar eksik"], "duzeltilmis_yanit": GOOD_ANSWER},
    ensure_ascii=False,
)


class ScriptedPool:
    """Fake pool that answers by prompt kind, with per-call overrides."""

    def __init__(
        self,
        *,
        drafts: list[str] | None = None,
        answers: list[str] | None = None,
        critiques: list[str] | None = None,
        teacher: str = "mimo",
        raise_first: Exception | None = None,
    ) -> None:
        self.drafts = list(drafts or [GOOD_DRAFT])
        self.answers = list(answers or [GOOD_ANSWER])
        self.critiques = list(critiques or [CLEAN_CRITIQUE])
        # When a script runs out, the last entry is recycled (queues express
        # per-call overrides, not exact call counts).
        self._last = {"draft": self.drafts[-1], "answer": self.answers[-1],
                      "critique": self.critiques[-1]}
        self.teacher_name = teacher
        self.calls: list[dict] = []
        self.failures = 0
        self.raise_first = raise_first
        from gokturk.config import TeacherModelConfig

        self._model = TeacherModelConfig(name=teacher, model="model-x", role="code,general")

    def _respond(self, text: str):
        from gokturk.teachers import TeacherResponse

        return (
            TeacherResponse(
                text=text,
                model="model-x",
                teacher_name=self.teacher_name,
                latency_s=0.01,
                attempts=1,
                usage={"total_tokens": 42},
                finish_reason="stop",
            ),
            self._model,
        )

    @staticmethod
    def _kind(system_prompt: str, user_prompt: str) -> str:
        if "denetçisisin" in system_prompt:
            return "critique"
        if "veri seti tasarımcısı" in system_prompt:
            return "draft"
        return "answer"

    def ask_with_fallback(self, system_prompt, user_prompt, *, role=None, temperature=None, max_tokens=None):
        kind = self._kind(system_prompt, user_prompt)
        self.calls.append({"kind": kind, "role": role, "temperature": temperature})
        if self.raise_first is not None and kind == "answer":
            exc, self.raise_first = self.raise_first, None
            raise exc
        pool = {"draft": self.drafts, "answer": self.answers, "critique": self.critiques}[kind]
        text = pool.pop(0) if len(pool) > 1 else pool[0]
        return self._respond(text)

    def register_failure(self) -> int:
        self.failures += 1
        return self.failures

    def reset_failures(self) -> None:
        self.failures = 0

    # helpers used by legacy code paths
    def ask(self, system_prompt, user_prompt, **kwargs):
        return self.ask_with_fallback(system_prompt, user_prompt, **kwargs)


def _settings(**overrides) -> ProSettings:
    base = dict(candidates=2, critique=True, max_repair_attempts=1)
    base.update(overrides)
    return ProSettings(**base)


def _run(tmp_path, pool, *, target=2, families=("coding",), settings=None, **kwargs):
    from gokturk.config import ProviderConfig, TeachersConfig

    teachers = TeachersConfig(
        provider=ProviderConfig(
            name="test",
            base_url="https://example.invalid/v1",
            api_key_env="GTU_PRO_KEY",
            requests_per_minute=6000,
            abort_after_consecutive_failures=3,
        ),
        teachers=[{"name": "mimo", "model": "model-x", "role": "code,general"}],
    )
    teachers.validate()
    out = tmp_path / "pro.jsonl"
    stats = generate_pro(
        teachers,
        target=target,
        out_path=out,
        families=list(families),
        settings=settings or _settings(),
        quality=QualityConfig(min_response_chars=20),
        pool=pool,
        **kwargs,
    )
    return stats, out


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_critique_variants():
    assert parse_critique(CLEAN_CRITIQUE) == {"problems": [], "repaired": ""}
    parsed = parse_critique(f"İşte: {REPAIR_CRITIQUE}")
    assert parsed["problems"] == ["Kenar durumlar eksik"]
    assert parsed["repaired"].startswith("Aşağıdaki")
    assert parse_critique("JSON yok") is None
    assert parse_critique('{"sorunlar": "tek"}')["problems"] == ["tek"]


def test_pro_settings_validation():
    from gokturk.config import ConfigError

    with pytest.raises(ConfigError, match="candidates"):
        ProSettings(candidates=0).validate()
    with pytest.raises(ConfigError, match="max_repair_attempts"):
        ProSettings(max_repair_attempts=9).validate()


# ---------------------------------------------------------------------------
# record shape
# ---------------------------------------------------------------------------
def test_record_has_full_provenance_and_no_secrets():
    slot = build_plan(1, seed=0, families=["coding"]).slots[0]
    family = get_family("coding")
    from gokturk.config import ProviderConfig, TeacherModelConfig

    teacher = TeacherModelConfig(name="mimo", model="m", role="code")
    record = build_record_pro(
        slot=slot,
        family=family,
        prompt="istek",
        response="yanıt",
        teacher=teacher,
        provider=ProviderConfig(base_url="https://example.invalid/v1"),
        candidate_index=1,
        candidates_tried=2,
        critique={"problems": ["x"], "repaired": ""},
        draft_teacher="mimo",
        response_meta={"finish_reason": "stop", "usage": {"total_tokens": 5}},
        soft_notes={"no_meta_preamble": ["dolgu"]},
    )
    assert record["source"]["family_label"] == family.label
    assert record["source"]["teacher_roles"] == ["code"]
    assert record["meta"]["candidate_index"] == 1
    assert record["meta"]["critique_problems"] == ["x"]
    assert record["meta"]["review"]["status"] == "unreviewed"
    dumped = json.dumps(record).lower()
    for token in ("api_key", "authorization", "bearer", "sk-"):
        assert token not in dumped


# ---------------------------------------------------------------------------
# generation behaviour
# ---------------------------------------------------------------------------
def test_pipeline_accepts_clean_answer(tmp_path):
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER])
    stats, out = _run(tmp_path, pool)
    assert stats.accepted == 2
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all(rec["source"]["task_type"] == "coding" for rec in lines)
    assert lines[0]["meta"]["candidates_tried"] == 2
    assert (tmp_path / "pro.critique.jsonl").is_file()


def test_candidate_selection_prefers_clean_answer(tmp_path):
    # First candidate fails validators (no code block), second is clean.
    pool = ScriptedPool(answers=[WEAK_ANSWER, GOOD_ANSWER], critiques=[CLEAN_CRITIQUE])
    stats, out = _run(tmp_path, pool, target=1)
    assert stats.accepted == 1
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert "```python" in record["response"]
    assert record["meta"]["candidate_index"] == 1


def test_temperature_jitter_passed_only_where_appropriate(tmp_path):
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER])
    _run(tmp_path, pool, target=1)
    answer_calls = [c for c in pool.calls if c["kind"] == "answer"]
    assert len(answer_calls) == 2
    assert all(c["temperature"] is not None for c in answer_calls)
    assert answer_calls[0]["temperature"] <= answer_calls[1]["temperature"]


def test_role_routing_is_requested(tmp_path):
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER])
    _run(tmp_path, pool, target=1, families=["coding"])
    answer_roles = [c["role"] for c in pool.calls if c["kind"] == "answer"]
    assert answer_roles and all(role == "code" for role in answer_roles)
    # The critique step deliberately uses a different (reasoning) reviewer role.
    critique_roles = [c["role"] for c in pool.calls if c["kind"] == "critique"]
    assert critique_roles and all(role == "reasoning" for role in critique_roles)


def test_critique_repair_is_applied_when_it_passes(tmp_path):
    pool = ScriptedPool(
        answers=[WEAK_ANSWER, GOOD_ANSWER],
        critiques=[REPAIR_CRITIQUE],
    )
    stats, out = _run(tmp_path, pool, target=1)
    assert stats.accepted == 1
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["meta"]["repaired"] is True
    assert "```python" in record["response"]


def test_repair_is_rejected_when_it_fails_validators(tmp_path):
    bad_repair = json.dumps(
        {"sorunlar": ["eksik"], "duzeltilmis_yanit": "kod bloğu olmayan kısa yanıt"},
        ensure_ascii=False,
    )
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER], critiques=[bad_repair])
    stats, out = _run(tmp_path, pool, target=1)
    assert stats.accepted == 1
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert "```python" in record["response"]
    assert record["meta"]["repaired"] is False
    assert stats.rejected_repairs >= 1


def test_unusable_draft_is_rejected(tmp_path):
    pool = ScriptedPool(drafts=["JSON değil"], answers=[GOOD_ANSWER])
    stats, out = _run(tmp_path, pool, target=1)
    assert stats.accepted == 0
    assert stats.rejected == 1
    assert any("drafter" in reason for reason in stats.reject_reasons)


def test_last_resort_rejection_writes_rejects_file(tmp_path):
    pool = ScriptedPool(answers=[WEAK_ANSWER, WEAK_ANSWER], critiques=[CLEAN_CRITIQUE])
    stats, _ = _run(tmp_path, pool, target=1)
    assert stats.accepted == 0
    rejects = (tmp_path / "pro.rejects.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert rejects and "reason" in json.loads(rejects[0])


def test_critique_disabled_skips_extra_calls(tmp_path):
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER])
    stats, _ = _run(tmp_path, pool, target=1, settings=_settings(critique=False))
    assert stats.accepted == 1
    assert all(c["kind"] != "critique" for c in pool.calls)


def test_resume_skips_completed_slots(tmp_path):
    pool = ScriptedPool(answers=[GOOD_ANSWER] * 6, drafts=[GOOD_DRAFT] * 4)
    _run(tmp_path, pool, target=2)
    before = (tmp_path / "pro.jsonl").read_text(encoding="utf-8")
    pool2 = ScriptedPool(answers=[GOOD_ANSWER] * 6, drafts=[GOOD_DRAFT] * 4)
    stats, out = _run(tmp_path, pool2, target=2)
    assert stats.skipped_existing == 2
    assert out.read_text(encoding="utf-8") == before


def test_auth_error_aborts_without_fabricating_data(tmp_path):
    pool = ScriptedPool(raise_first=TeacherAuthError("HTTP 401"))
    with pytest.raises(GenerationAborted, match="TeacherAuthError"):
        _run(tmp_path, pool, target=3)
    out = tmp_path / "pro.jsonl"
    assert not out.exists() or not out.read_text(encoding="utf-8").strip()


def test_request_error_is_fatal(tmp_path):
    pool = ScriptedPool(raise_first=TeacherRequestError("HTTP 400: bad model"))
    with pytest.raises(GenerationAborted, match="TeacherRequestError"):
        _run(tmp_path, pool, target=2)


def test_consecutive_transport_failures_abort(tmp_path):
    class AlwaysFailing(ScriptedPool):
        def ask_with_fallback(self, system_prompt, user_prompt, **kwargs):
            if self._kind(system_prompt, user_prompt) == "answer":
                raise TeacherServerError("503")
            return super().ask_with_fallback(system_prompt, user_prompt, **kwargs)

    pool = AlwaysFailing()
    with pytest.raises(GenerationAborted, match="ardışık"):
        _run(tmp_path, pool, target=10)


def test_plan_only_needs_no_credentials(tmp_path):
    summary = plan_only(target=50, seed=1, families=["coding", "reasoning"], plan_path=tmp_path / "p.json")
    assert summary["cost"]["slots"] == 50
    assert set(summary["counts"]["family"]) == {"coding", "reasoning"}
    assert (tmp_path / "p.json").is_file()
    assert "tahmin" in summary["cost"]["note"].lower()


def test_sleep_between_is_honoured(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    pool = ScriptedPool(answers=[GOOD_ANSWER, GOOD_ANSWER])
    _run(tmp_path, pool, target=1, sleep_between=0.01)
    assert slept and all(s == 0.01 for s in slept)
