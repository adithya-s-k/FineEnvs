import json

import pytest

from berth_core import TaskPack, evaluate, grade, parse_plan, plan_from_list, score
from berth_core.pack import LEGACY_PACK, TASKS_ROOT
from berth_core.data import load_calls
from berth_core.generate import build_task
from berth_core.model import PlanError
from berth_core.reward import INFEASIBLE_CAP, NAIVE_LEVEL
from berth_core.solve import naive_replan

PACK = TaskPack(TASKS_ROOT / LEGACY_PACK)


@pytest.fixture(scope="module")
def task():
    return PACK.tasks[0]


def test_pack_shape():
    assert len(PACK.tasks) == 100
    assert {t.difficulty for t in PACK.tasks} == {"easy", "medium", "hard", "expert"}
    assert set(PACK.splits()) == {"train", "test"}
    assert len({t.task_id for t in PACK.tasks}) == 100
    assert all(t.week % 5 == 0 for t in PACK.tasks if t.split == "test")


@pytest.mark.parametrize("t", PACK.tasks, ids=lambda t: t.task_id)
def test_references_reproduce(t):
    ref = t.reference
    opt, naive = plan_from_list(ref["optimal_plan"]), plan_from_list(ref["naive_plan"])
    assert evaluate(t, opt).cost == ref["optimal_cost"]
    assert evaluate(t, naive).cost == ref["naive_cost"]
    assert naive_replan(t) == naive
    assert ref["optimal_cost"] < ref["naive_cost"]
    assert grade(t, opt).reward == 1.0
    assert grade(t, naive).reward == NAIVE_LEVEL


def test_parse_formats(task):
    ref = task.reference["optimal_plan"]
    a, pa = parse_plan(task, ref)
    b, pb = parse_plan(task, json.dumps(ref))
    c, pc = parse_plan(task, {str(r["ship"]): [r["berth_hour"], r["section"]] for r in ref})
    d, pd = parse_plan(task, "```json\n" + json.dumps({"plan": ref}) + "\n```")
    assert a == b == c == d and not (pa or pb or pc or pd)
    with pytest.raises(PlanError):
        parse_plan(task, "not json")
    _, problems = parse_plan(task, ref + [{"ship": 999, "berth_hour": 1, "section": 2}])
    assert problems and "unknown ship" in problems[0]


def test_violations(task):
    plan = plan_from_list(task.reference["optimal_plan"])
    s = task.ships[0]
    bad = dict(plan)
    bad[s.id] = (s.arrival - 1, plan[s.id][1])
    assert any("cannot arrive" in v["problem"] for v in evaluate(task, bad).violations)
    bad = dict(plan)
    bad[s.id] = (plan[s.id][0], task.last_section)
    assert not evaluate(task, bad).feasible
    other = task.ships[1]
    bad = dict(plan)
    bad[other.id] = plan[s.id] if other.sections <= task.last_section - plan[s.id][1] + 1 else bad[other.id]
    bad[other.id] = (max(plan[s.id][0], other.arrival), plan[s.id][1])
    if bad[other.id][0] < plan[s.id][0] + s.handling:
        assert any("overlaps ship" in v["problem"] for v in evaluate(task, bad).violations)
    missing = dict(plan)
    missing.pop(s.id)
    res = evaluate(task, missing)
    assert not res.feasible and res.clean_fraction < 1


def test_reward_bands_are_ordered_and_continuous():
    naive, opt = 300, 200
    assert score(None, False, 0.0, naive, opt)[0] == 0.0
    assert score(None, False, 0.999, naive, opt)[0] < INFEASIBLE_CAP
    assert abs(score(10**9, True, 1.0, naive, opt)[0] - INFEASIBLE_CAP) < 1e-6
    assert score(naive + 1, True, 1.0, naive, opt)[0] < NAIVE_LEVEL
    assert score(naive, True, 1.0, naive, opt)[0] == NAIVE_LEVEL
    assert score(250, True, 1.0, naive, opt)[0] == pytest.approx(0.8)
    assert score(opt, True, 1.0, naive, opt)[0] == 1.0
    rewards = [score(c, True, 1.0, naive, opt)[0] for c in range(2000, opt - 1, -10)]
    assert rewards == sorted(rewards)


def test_build_is_deterministic():
    calls = load_calls()
    a, _ = build_task(calls, "24B", 30, 0, "hard", "train", time_limit=30)
    b, _ = build_task(calls, "24B", 30, 0, "hard", "train", time_limit=30)
    assert a.to_dict(public=True) == b.to_dict(public=True)
    assert a.reference["optimal_cost"] == b.reference["optimal_cost"]
