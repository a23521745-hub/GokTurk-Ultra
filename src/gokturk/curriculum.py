"""Curriculum planner for professional synthetic data runs.

Turns a *target size* (e.g. the user's 10,000 examples) plus family weights into
a concrete, deterministic work plan: which family, which topic, which level and
which teacher role produce each slot. The plan is reproducible for a given seed
and can be dumped to JSON for auditing before any API call is made.

Teacher *model ids* are intentionally absent here: the plan speaks in terms of
semantic roles (``reasoning``, ``code``, ``general``). Runtime binding happens in
:mod:`gokturk.data_generation`, using only model ids that the operator has
verified and placed in the environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .task_specs import TaskFamily, resolve_families

LOGGER = logging.getLogger("gokturk.curriculum")


@dataclass
class Slot:
    """One planned example."""

    index: int
    family: str
    topic: str
    level: str
    role: str
    variant: int
    seed_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    target: int
    seed: int
    slots: list[Slot] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_family(self) -> dict[str, int]:
        return dict(Counter(s.family for s in self.slots).most_common())

    def by_level(self) -> dict[str, int]:
        return dict(Counter(s.level for s in self.slots).most_common())

    def by_role(self) -> dict[str, int]:
        return dict(Counter(s.role for s in self.slots).most_common())

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "seed": self.seed,
            "families": self.families,
            "counts": {
                "family": self.by_family(),
                "level": self.by_level(),
                "role": self.by_role(),
            },
            "notes": self.notes,
            "slots": [s.to_dict() for s in self.slots],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


def _largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Allocate ``total`` slots proportionally, using the largest-remainder method."""
    if total <= 0:
        return {key: 0 for key in weights}
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    weight_sum = sum(positive.values())
    if weight_sum <= 0:
        raise ValueError("weights must not all be zero")
    raw = {k: total * v / weight_sum for k, v in positive.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(raw, key=lambda k: (raw[k] - floors[k], k), reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def build_plan(
    target: int,
    *,
    seed: int = 0,
    families: Iterable[str] | None = None,
    levels: Iterable[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> Plan:
    """Create a deterministic, diversity-aware plan of ``target`` examples."""
    if target < 1:
        raise ValueError("target must be >= 1")
    selected: list[TaskFamily] = resolve_families(families)
    if levels:
        allowed = {level.casefold() for level in levels}
        compatible = [f for f in selected if set(f.levels) & allowed]
        if not compatible:
            raise ValueError(
                "no selected family offers the requested level(s) "
                f"{sorted(allowed)}; available: "
                + ", ".join(f"{f.id}={list(f.levels)}" for f in selected)
            )
        skipped = [f.id for f in selected if f not in compatible]
        selected = compatible
        if skipped:
            LOGGER.info(
                "seviye filtresi nedeniyle atlanan aileler: %s", ", ".join(skipped)
            )

    weights_by_id = {f.id: float((weights or {}).get(f.id, f.weight)) for f in selected}
    allocation = _largest_remainder(weights_by_id, target)

    rng = random.Random(seed)
    # A seeded shuffle avoids a fixed family order correlating with slot index,
    # while keeping repeated plans identical for the same seed.
    order = [f for f in selected if allocation[f.id] > 0]
    rng.shuffle(order)
    slots: list[Slot] = []
    remaining = dict(allocation)
    round_robin: list[TaskFamily] = []
    # Interleave families so a truncated run still contains every family.
    while any(remaining[f.id] > 0 for f in order):
        for family in order:
            if remaining[family.id] > 0:
                round_robin.append(family)
                remaining[family.id] -= 1

    for index, family in enumerate(round_robin):
        usable_levels = [
            level
            for level in family.levels
            if not levels or level in {lv.casefold() for lv in levels}
        ]
        seed_id = f"{family.id}-{index // max(1, len(family.topics))}"
        topic = family.pick_topic(seed_id, index)
        level = usable_levels[
            int(hashlib.sha1(f"lvl|{family.id}|{index}|{seed}".encode()).hexdigest()[:8], 16)
            % len(usable_levels)
        ]
        role = family.preferred_roles[index % len(family.preferred_roles)]
        slots.append(
            Slot(
                index=index,
                family=family.id,
                topic=topic,
                level=level,
                role=role,
                variant=index,
                seed_id=seed_id,
            )
        )

    notes = [
        "Plan yalnızca görev dağılımını belirler; içerik öğretmen modeller tarafından üretilir.",
        "Öğretmen model kimlikleri planda yer almaz: çalışma anında doğrulanmış "
        "ortam değişkenlerinden bağlanır.",
    ]
    return Plan(
        target=target,
        seed=seed,
        slots=slots,
        families=[f.id for f in selected],
        notes=notes,
    )


def estimate_cost(
    plan: Plan, *, prompt_tokens: int = 700, response_tokens: int = 900, candidates: int = 1
) -> dict[str, Any]:
    """Very rough token estimate so a run can be budgeted before it starts.

    Deliberately simple: ~4 characters per token for Turkish is a planning
    approximation, not a measurement. Real usage must be read from provider
    billing/usage counters after the run.
    """
    calls_per_slot = 2 + max(0, candidates - 1)  # drafter + N answers
    total_calls = len(plan.slots) * calls_per_slot
    input_tokens = total_calls * prompt_tokens
    output_tokens = total_calls * response_tokens
    return {
        "slots": len(plan.slots),
        "calls_per_slot": calls_per_slot,
        "total_calls": total_calls,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_tokens + output_tokens,
        "note": (
            "Kaba planlama tahminidir (≈4 karakter/token varsayımı); gerçek kullanım "
            "sağlayıcının usage sayaçlarından okunmalıdır."
        ),
    }
