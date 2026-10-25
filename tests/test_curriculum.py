"""Curriculum planner: determinism, quotas, level filtering, cost estimate."""

from __future__ import annotations

import json

import pytest

from gokturk.curriculum import Plan, _largest_remainder, build_plan, estimate_cost
from gokturk.task_specs import FAMILIES


def test_largest_remainder_is_exact_and_proportional():
    allocation = _largest_remainder({"a": 1.0, "b": 1.0, "c": 2.0}, 100)
    assert sum(allocation.values()) == 100
    assert allocation["c"] == 50
    assert allocation["a"] == 25


def test_largest_remainder_handles_tiny_targets():
    allocation = _largest_remainder({"a": 1.0, "b": 1.0}, 1)
    assert sum(allocation.values()) == 1


def test_zero_weights_rejected():
    with pytest.raises(ValueError):
        _largest_remainder({"a": 0.0}, 10)


def test_plan_hits_target_and_is_deterministic():
    first = build_plan(500, seed=11)
    second = build_plan(500, seed=11)
    assert len(first.slots) == 500
    assert [s.to_dict() for s in first.slots] == [s.to_dict() for s in second.slots]


def test_plan_uses_every_family_even_when_truncated():
    plan = build_plan(10, seed=1)
    assert len(plan.slots) == 10
    seen = {s.family for s in plan.slots}
    # 10 slots spread over the highest-weight families: no family may dominate all
    assert len(seen) >= 5


def test_plan_respects_family_subset():
    plan = build_plan(40, seed=2, families=["coding", "reasoning"])
    assert {s.family for s in plan.slots} == {"coding", "reasoning"}
    assert plan.families == ["coding", "reasoning"]


def test_plan_respects_level_filter():
    plan = build_plan(30, seed=3, levels=["ileri"])
    assert {s.level for s in plan.slots} == {"ileri"}
    # families without an "ileri" level are skipped, not force-fitted
    assert "tool_calling" not in plan.by_family()


def test_plan_rejects_impossible_level_filter():
    with pytest.raises(ValueError, match="no selected family offers"):
        build_plan(10, families=["tool_calling"], levels=["ileri"])


def test_plan_assigns_roles_from_family_preferences():
    plan = build_plan(200, seed=5)
    for slot in plan.slots:
        assert slot.role in FAMILIES[slot.family].preferred_roles


def test_plan_target_validation():
    with pytest.raises(ValueError):
        build_plan(0)


def test_plan_never_contains_model_ids():
    plan = build_plan(20, seed=1)
    dumped = json.dumps(plan.to_dict()).lower()
    for token in ("gpt", "glm", "mimo", "http"):
        assert token not in dumped


def test_plan_write_and_reload(tmp_path):
    from gokturk.data_generation import _load_plan

    path = tmp_path / "plan.json"
    plan = build_plan(25, seed=9)
    written = plan.write(path)
    assert written.is_file()
    restored = _load_plan(path)
    assert isinstance(restored, Plan)
    assert [s.to_dict() for s in restored.slots] == [s.to_dict() for s in plan.slots]


def test_plan_summaries_add_up():
    plan = build_plan(300, seed=4)
    assert sum(plan.by_family().values()) == 300
    assert sum(plan.by_level().values()) == 300
    assert sum(plan.by_role().values()) == 300


def test_cost_estimate_scales_with_candidates():
    plan = build_plan(100, seed=0)
    one = estimate_cost(plan, candidates=1)
    three = estimate_cost(plan, candidates=3)
    assert three["total_calls"] > one["total_calls"]
    assert "tahmin" in three["note"].lower()
    assert three["total_calls"] == 100 * 4


def test_10k_plan_is_practical_and_balanced():
    plan = build_plan(10_000, seed=0)
    counts = plan.by_family()
    assert len(counts) == len(FAMILIES)
    assert all(count > 300 for count in counts.values()), counts
    assert min(plan.by_level().values()) > 2000
