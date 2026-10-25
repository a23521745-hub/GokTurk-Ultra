"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gokturk.config import (  # noqa: E402
    ProviderConfig,
    TeacherModelConfig,
    TeachersConfig,
)
from gokturk.teachers import TeacherResponse  # noqa: E402


def make_record(
    rec_id: str = "gtu-001",
    prompt: str = "Türkiye'nin başkenti neresidir ve neden önemlidir?",
    response: str = "Türkiye'nin başkenti Ankara'dır. 1923'ten beri başkenttir.",
    *,
    teacher: str = "mimo-v26-pro",
    task_type: str = "turkish_qa",
    seed_id: str = "seed-1",
    kind: str = "teacher",
) -> dict[str, Any]:
    return {
        "id": rec_id,
        "prompt": prompt,
        "response": response,
        "source": {
            "kind": kind,
            "teacher": teacher,
            "endpoint": "https://example.invalid/v1",
            "task_type": task_type,
            "template_version": "v1",
            "generated_at": "2026-10-25T10:00:00Z",
            "seed_id": seed_id,
        },
    }


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


class FakeTransport:
    """Callable transport that replays queued responses/exceptions."""

    def __init__(self, responses: Sequence[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, payload: dict[str, Any], api_key: str, timeout: float):
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeTransport exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def chat_payload(text: str, model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def provider() -> ProviderConfig:
    return ProviderConfig(
        name="test",
        base_url="https://example.invalid/v1",
        api_key_env="GTU_TEST_API_KEY",
        requests_per_minute=6000.0,
        max_retries=2,
        max_attempts_per_sample=3,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )


@pytest.fixture
def teachers_config(provider: ProviderConfig) -> TeachersConfig:
    cfg = TeachersConfig(
        provider=provider,
        teachers=[
            {"name": "mimo", "model": "mimo-test", "weight": 1.0, "temperature": 0.1},
            {"name": "kimi", "model": "kimi-test", "weight": 1.0, "temperature": 0.2},
        ],
    )
    cfg.validate()
    return cfg


class FakePool:
    """Stand-in for TeacherPool: deterministic draft/answer responses."""

    def __init__(self, drafts: Sequence[str], answers: Sequence[str], teacher_name: str = "mimo"):
        self.drafts = list(drafts)
        self.answers = list(answers)
        self.teacher_name = teacher_name
        self.calls = 0
        self._model = TeacherModelConfig(name=teacher_name, model="mimo-test")

    def ask(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        is_draft = "veri seti tasarımcısı" in system_prompt or "JSON döndür" in system_prompt
        text: str | None = None
        if is_draft and self.drafts:
            text = self.drafts.pop(0)
        elif self.answers:
            text = self.answers.pop(0)
        if text is None:
            raise AssertionError("FakePool ran out of canned responses")
        return (
            TeacherResponse(
                text=text,
                model="mimo-test",
                teacher_name=self.teacher_name,
                latency_s=0.01,
                attempts=1,
                usage={"total_tokens": 30},
                finish_reason="stop",
            ),
            self._model,
        )

    def register_failure(self) -> int:
        return 0

    def reset_failures(self) -> None:
        return None
