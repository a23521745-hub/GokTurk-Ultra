"""Canonical JSONL record schema for GökTürk-Ultra training data.

One line of JSON per example::

    {
      "id": "gtu-000001",                       # required, unique, stable
      "prompt": "…",                            # required, user turn
      "response": "…",                          # required, target answer
      "source": {                               # required, provenance
        "kind": "teacher",                      # teacher | human | seed | synthetic
        "teacher": "mimo-v2.6-pro",             # model id used to generate
        "endpoint": "https://…/v1",             # base URL (no credentials)
        "task_type": "turkish_qa",              # task family
        "template_version": "v1",
        "generated_at": "2026-10-25T10:00:00Z",
        "seed_id": "seed-012"                   # optional provenance link
      },
      "messages": [...],                        # optional full chat, if multi-turn
      "meta": {"lang": "tr", "tokens_response": 120}
    }

Design notes
------------
* ``source`` is a nested object; a bare string is also accepted by the
  validator and normalised to ``{"kind": <string>}`` for backwards
  compatibility with simple hand-written files.
* Nothing in the schema stores API keys, prompts to the teacher, or raw
  provider payloads. Provider metadata is limited to model id and base URL,
  which are not secrets.
* The schema is validated by hand-written code in :mod:`gokturk.data_quality`
  (no ``jsonschema`` runtime dependency).
"""

from __future__ import annotations

from typing import Any, Final

SCHEMA_VERSION: Final[str] = "1.0"

REQUIRED_FIELDS: Final[tuple[str, ...]] = ("id", "prompt", "response", "source")

OPTIONAL_FIELDS: Final[tuple[str, ...]] = ("messages", "meta", "schema_version")

ALLOWED_FIELDS: Final[tuple[str, ...]] = REQUIRED_FIELDS + OPTIONAL_FIELDS

SOURCE_KINDS: Final[tuple[str, ...]] = ("teacher", "human", "seed", "synthetic")

#: Task families used both by the generator and the evaluator.
TASK_TYPES: Final[tuple[str, ...]] = (
    "turkish_qa",
    "instruction_following",
    "summarization",
    "coding",
    "tool_calling",
    "reasoning",
    "rewrite",
    "translation",
)


def empty_record() -> dict[str, Any]:
    """Return a record skeleton (useful in tests and documentation)."""
    return {
        "id": "",
        "prompt": "",
        "response": "",
        "source": {
            "kind": "teacher",
            "teacher": "",
            "endpoint": "",
            "task_type": "turkish_qa",
            "template_version": "",
            "generated_at": "",
        },
        "meta": {"lang": "tr"},
        "schema_version": SCHEMA_VERSION,
    }
