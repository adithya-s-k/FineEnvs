"""The dock-v1 packs: splits never share a week, tiers are balanced, every reference reproduces, lazy policies stay
low, and the optimum is the only way to 1.0."""

from pathlib import Path

import pytest

from berth_core import TaskPack, evaluate, grade, plan_from_list
from berth_core.baselines import greedy, published, serial
from berth_core.dock import HELD_OUT, MAX_GREEDY_REWARD, MAX_NAIVE_REWARD, TIERS
from berth_core.solve import naive_replan

ROOT = Path(__file__).resolve().parents[2] / "tasks"
EVAL, TRAIN = ROOT / "dock-v1-eval", ROOT / "dock-v1-train"
have = lambda p: (p / "tasks.jsonl").is_file() or (p / "tasks.jsonl.gz").is_file()
EV = TaskPack(EVAL) if have(EVAL) else None
TR = TaskPack(TRAIN) if have(TRAIN) else None


def weeks(t):
    n = int(t.task_id.split("x")[1].split("-")[0])
    return set(range(t.week, t.week + n))


@pytest.mark.skipif(EV is None, reason="eval pack not built")
def test_eval_pack_shape():
    assert len(EV.tasks) == 50
    counts = {k: sum(t.difficulty == k for t in EV.tasks) for k in TIERS}
    assert min(counts.values()) >= 9
    assert all(weeks(t) <= HELD_OUT for t in EV.tasks)
    assert {t.split for t in EV.tasks} == {"eval"}
    assert len({t.quay for t in EV.tasks}) == 2


@pytest.mark.skipif(TR is None, reason="train pack not built")
def test_train_pack_shape():
    assert len(TR.tasks) == 1050
    counts = {k: sum(t.difficulty == k for t in TR.tasks) for k in TIERS}
    assert min(counts.values()) >= 250
    assert all(not (weeks(t) & HELD_OUT) for t in TR.tasks)
    assert {t.split for t in TR.tasks} == {"train"}


def _sample():
    out = []
    for pk in (EV, TR):
        if pk:
            out += pk.tasks[:: max(1, len(pk.tasks) // 40)]
    return out


@pytest.mark.parametrize("t", _sample(), ids=lambda t: t.task_id)
def test_references_and_lazy_policies(t):
    ref = t.reference
    opt = plan_from_list(ref["optimal_plan"])
    assert evaluate(t, opt).cost == ref["optimal_cost"]
    assert grade(t, opt).reward == 1.0
    assert ref["optimal_cost"] - ref["lower_bound"] <= 0.01 * max(1, ref["optimal_cost"])
    naive = naive_replan(t)
    assert plan_from_list(ref["naive_plan"]) == naive
    assert grade(t, naive).reward <= MAX_NAIVE_REWARD + 1e-9
    assert grade(t, greedy(t)).reward <= MAX_GREEDY_REWARD + 1e-9
    assert grade(t, serial(t)).reward < 0.3
    assert grade(t, published(t)).reward < 0.2 or evaluate(t, published(t)).feasible


@pytest.mark.skipif(not (EV and TR), reason="packs not built")
def test_no_duplicate_ids_across_packs():
    TaskPack([EVAL, TRAIN])
