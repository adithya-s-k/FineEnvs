"""tasks/marine/references.jsonl replayed without ortools: the stored rolling plans and the naive online policy
reproduce their stored grades through the marine week, the hindsight optimum is proven and never below the live week's,
and a week qualifies by the 3-of-4 rule."""

import berth_core
import pytest
from berth_core import plan_from_list
from berth_core.check import unavoidable_cost

from port_week_core import marine, packs, schedule, world
from port_week_core.marine import MarineWeek

REFERENCES = marine.references()
PACK = {t.task_id: t for t in marine.pack().tasks}
V2 = {t.task_id: t for t in packs.live_pack().tasks if t.split == "eval"}
CONFIGS = [(1, True), (1, False), (8, True), (8, False)]
QUALIFYING = [
    "dock-24B-w06x1-busy-0", "dock-24B-w07x1-busy-0", "dock-24B-w16x1-busy-0", "dock-24B-w35x1-busy-0",
    "dock-36A-w05x1-busy-0", "dock-36A-w06x1-busy-0", "dock-36A-w17x1-busy-0", "dock-36A-w37x1-busy-0",
    "dock-24B-w06x1-standard-0", "dock-24B-w37x1-standard-0", "dock-36A-w06x1-standard-0", "dock-36A-w10x1-standard-0",
    "dock-36A-w15x1-standard-0", "dock-36A-w35x1-standard-0", "dock-36A-w37x1-standard-0",
]


def graded(week: world.Week) -> dict:
    return {key: week.grade[key] for key in ("cost", "reward", "feasible", "excused_cost")}


def test_the_references_are_the_marine_packs_weeks_in_pack_order():
    assert [r["task_id"] for r in REFERENCES] == list(PACK)
    assert len(REFERENCES) == 18


@pytest.mark.parametrize("ref", REFERENCES, ids=lambda r: r["task_id"])
def test_the_hindsight_optimum_is_proven_feasible_and_never_below_v2s(ref):
    task = PACK[ref["task_id"]]
    res = marine.evaluate(task, plan_from_list(task.reference["optimal_plan"]))
    assert res.feasible and res.cost == ref["optimal_cost"] == task.reference["optimal_cost"]
    assert task.reference["proven_optimal"] and task.reference["lower_bound"] == ref["optimal_cost"]
    assert ref["optimal_cost"] >= ref["v2_optimal_cost"] == V2[ref["task_id"]].reference["optimal_cost"]
    assert ref["unavoidable_cost"] == unavoidable_cost(task)


@pytest.mark.parametrize("ref", REFERENCES, ids=lambda r: r["task_id"])
def test_the_stored_rolling_plans_replay_to_the_stored_grade_with_nothing_refused(ref):
    task = PACK[ref["task_id"]]
    plans = ref["rolling"]["plans"]
    week = world.play(task, lambda w: plan_from_list(plans[w.watch]), week=MarineWeek)
    assert week.refusals == []
    assert len(plans) == len(week.watches)
    assert graded(week) == {key: value for key, value in ref["rolling"].items() if key != "plans"}
    assert ref["watch_hours"] == [w.hour for w in schedule.schedule(task)]


@pytest.mark.parametrize("ref", REFERENCES, ids=lambda r: r["task_id"])
def test_the_naive_online_policy_reproduces_its_stored_grade_every_time(ref):
    first, second = (world.play(PACK[ref["task_id"]], world.naive, week=MarineWeek) for _ in range(2))
    assert first.refusals == []
    assert graded(first) == graded(second) == ref["naive"]
    assert first.plan == second.plan


def test_every_stored_plan_moves_before_the_traffic_horizon():
    for ref in REFERENCES:
        task = PACK[ref["task_id"]]
        for plan in [task.reference["optimal_plan"], *ref["rolling"]["plans"]]:
            res = berth_core.evaluate(task, plan_from_list(plan))
            assert all(r.departure < task.rules["marine"]["hours"] for r in res.ships if r.berth_hour is not None)


def test_a_week_qualifies_when_three_of_the_four_solver_configurations_reach_the_optimum():
    for ref in REFERENCES:
        assert [(c["workers"], c["tiebreak"]) for c in ref["configs"]] == CONFIGS
        assert ref["configs"][0]["cost"] == ref["rolling"]["cost"]
        reached = sum(c["cost"] is not None and c["cost"] <= ref["optimal_cost"] for c in ref["configs"])
        assert ref["qualifies"] is (reached >= 3)
        assert ref["solver"] == {"ortools": "9.15.6755", "max_deterministic_time": 60.0}
    assert marine.task_ids() == QUALIFYING
    assert {r["rolling"]["reward"] for r in REFERENCES if r["qualifies"]} == {1.0}


def test_the_example_week_reaches_its_optimum_live():
    ref = next(r for r in REFERENCES if r["task_id"] == "dock-24B-w07x1-busy-0")
    assert ref["watch_hours"] == [0, 30, 42, 54, 84, 90, 120]
    assert (ref["optimal_cost"], ref["v2_optimal_cost"], ref["unavoidable_cost"]) == (226, 226, 177)
    assert {key: value for key, value in ref["rolling"].items() if key != "plans"} == {
        "cost": 226, "reward": 1.0, "feasible": True, "excused_cost": 0}
