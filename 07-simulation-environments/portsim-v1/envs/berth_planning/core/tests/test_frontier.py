from pathlib import Path

import pytest

from berth_core import TaskPack, evaluate, grade, parse_plan, plan_from_list
from berth_core.reward import score_v2
from berth_core.solve import naive_replan

ROOT = Path(__file__).resolve().parents[2] / "tasks" / "berth-frontier-v1"
pytestmark = pytest.mark.skipif(not (ROOT / "tasks.jsonl").is_file(), reason="frontier pack not built")
PACK = TaskPack(ROOT) if (ROOT / "tasks.jsonl").is_file() else None


@pytest.mark.parametrize("t", PACK.tasks if PACK else [], ids=lambda t: t.task_id)
def test_frontier_references(t):
    opt = plan_from_list(t.reference["optimal_plan"])
    naive = plan_from_list(t.reference["naive_plan"])
    assert evaluate(t, opt).cost == t.reference["optimal_cost"]
    assert evaluate(t, naive).cost == t.reference["naive_cost"]
    assert naive_replan(t) == naive
    assert grade(t, opt).reward == 1.0


def test_crane_rules():
    t = PACK.tasks[0]
    opt = plan_from_list(t.reference["optimal_plan"])
    sid, entry = next((k, v) for k, v in opt.items())
    s = t.ships[sid]
    bad = dict(opt)
    bad[sid] = (entry[0], entry[1], s.max_cranes + 1)
    assert any("cranes but can be worked by" in v["problem"] for v in evaluate(t, bad).violations)
    # all ships at max cranes, berthing at arrival on their planned sections: pool or movements must break somewhere
    greedy = {x.id: (x.arrival, x.planned_section or t.first_section, x.max_cranes) for x in t.ships}
    probs = " ".join(v["problem"] for v in evaluate(t, greedy).violations)
    assert "over the pool" in probs or "ships move" in probs or "overlaps" in probs
    # a plan without cranes falls back to the planned crane counts
    rows = [{"ship": k, "berth_hour": v[0], "section": v[1]} for k, v in opt.items()]
    plan, problems = parse_plan(t, rows)
    assert not problems and all(plan[x.id][2] == x.std_cranes for x in t.ships)


def test_reward_v2_shape():
    naive, opt = 1000, 200
    assert score_v2(opt, True, 1.0, naive, opt)[0] == 1.0
    assert score_v2(opt - 10, True, 1.0, naive, opt)[0] == 1.0
    assert score_v2(naive, True, 1.0, naive, opt)[0] == pytest.approx(0.2 + 0.8 * 2.718281828 ** -4, abs=1e-6)
    assert score_v2(None, False, 0.5, naive, opt)[0] == pytest.approx(0.1)
    xs = [score_v2(c, True, 1.0, naive, opt)[0] for c in range(3000, opt - 1, -25)]
    assert xs == sorted(xs) and min(xs) > 0.2
