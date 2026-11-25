"""The hard gate: the reference follower's decisions, confirmed watch by watch through the day and through the env's
tools, are never refused or held, freeze what the follower pinned, and fly to the follower's cost exactly. HiGHS can
return a different optimal solution to a watch from run to run, so each run is replayed against itself."""

import json
from unittest import mock

import pytest
from airops_core.core.known import frozen, view
from airops_core.core.model import Task, load_pack
from airops_core.core.reward import reward
from airops_core.core.sim import execute
from airops_core.day import Day, _entry, changes_to, play
from airops_core.refs import follower, mip
from paths import FIXTURES

pytestmark = pytest.mark.anyio
SMALL = "b6-jfk-20230626-1400-1700"
SLOW = ("b6-bos-20240107-0600", "b6-jfk-20230626-1400-all", "b6-jfk-20230626-1200")
"""The fixture days whose follower takes minutes to solve."""


def follower_plans(task: Task) -> tuple[list[dict[int, dict]], follower.Followed]:
    """The follower's plan after each watch, read off its solves, and its run."""
    solves = []

    def solve(*args, **kwargs):
        solution = mip.solve(*args, **kwargs)
        solves.append((set(kwargs["fixed"] if "fixed" in kwargs else args[3]), solution))
        return solution

    with mock.patch.object(follower, "solve", solve):
        followed = follower.follow(task)
    plans, plan = [], {}
    for pinned, solution in solves:
        plan = {i: plan[i] if i in pinned else d for i, d in solution.plan.items()}
        plans.append(plan)
    assert plans[-1] == followed.plan
    return plans, followed


def played(task: Task, plans: list[dict]) -> tuple[Day, list[list[dict]]]:
    """The follower's day played in process, and the changes it confirms at each watch."""
    rows = []

    def policy(day: Day) -> dict:
        rows.append(changes_to(day, plans[day.watch]))
        return plans[day.watch]
    day = play(task, policy)
    assert day.refusals == [] and all(c["by"] == "planner" for c in day.changes)
    return day, rows


@pytest.fixture(scope="module")
def followed():
    return {}


def plans_of(task, cache: dict) -> tuple[list[dict], object]:
    if task.task_id not in cache:
        cache[task.task_id] = follower_plans(task)
    return cache[task.task_id]


@pytest.mark.parametrize("task_id", [pytest.param(t.task_id, marks=pytest.mark.slow) if t.task_id in SLOW else t.task_id
                                     for t in load_pack(FIXTURES)])
def test_the_followers_decisions_replayed_through_the_day_cost_what_the_follower_costs(fixtures, followed, task_id):
    task = fixtures[task_id]
    plans, run = plans_of(task, followed)
    day, rows = played(task, plans)
    assert len(rows) == len(plans) == len(day.watches) and rows[0]
    assert day.refusals == [] and {c["by"] for c in day.changes} == {"planner"}
    assert {i: _entry(e) for i, e in day.plan.items()} == {
        i: _entry(e) for i, e in plans[-1].items() if _entry(e) != _entry(None)}
    pinned = set()
    for k, w in enumerate(day.watches[1:], 1):
        pinned |= frozen(task, execute(task, plans[k - 1], *view(task, w.t)), w.t)
        assert day.freezes[k]["frozen"] == sorted(pinned)
    ref = task.reference
    assert day.grade["cost"]["total"] == run.cost == execute(task, plans[-1]).cost.total
    assert day.grade["reward"] == round(reward(run.cost, ref["anchor"], ref["floor"]), 6)
    assert day.audit() == {"ok": True, "problems": []}


async def test_the_same_through_the_tools_and_the_clock(fixtures, followed, env, tools, gateway, http):
    task = fixtures[SMALL]
    plans, run = plans_of(task, followed)
    day, rows = played(task, plans)
    env.live_load(SMALL)
    await env.sync_time(f"{gateway.url}/clock/time")
    for k, changes in enumerate(rows):
        if changes:
            out = await tools("confirm_changes", changes=changes)
            assert (out["refused"], out["problems"], out["held_to_freeze_line"]) == ([], [], [])
        out = await tools("advance")
        if k < len(rows) - 1:
            assert out["watch"] == k + 1
    assert out == {"done": True, "cost": run.cost, "reward": day.grade["reward"], "messages": []}
    assert len(gateway.puts) == len(rows) - 1
    data = (await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})).json()
    got = data["result"]["parts"][0]["data"]
    assert got["grade"] == json.loads(json.dumps(day.grade))
    assert got["calls_used"] == sum(bool(c) for c in rows) + len(rows)
