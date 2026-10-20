import random

import pytest

from airops_core.core.sim import FLOWN, execute
from airops_core.refs import mip

SMALL = ["b6-jfk-20230626-1400-1700", "b6-jfk-20230627-1400"]


@pytest.mark.parametrize("task_id", SMALL)
def test_the_hindsight_optimum_is_proven_and_the_env_reproduces_it_exactly(by_id, task_id):
    task = by_id[task_id]
    sol = mip.solve(task)
    assert sol.proven and sol.objective == task.reference["hindsight"]
    run = execute(task, sol.plan)
    assert run.cost.total == sol.objective
    for i, d in sol.deps.items():
        assert (run.flights[i].dep, run.flights[i].status == FLOWN) == (d, d is not None)
    assert sol.stored() == task.reference["hindsight_plan"]


def test_the_floor_drops_the_aircraft_and_bounds_the_optimum(by_id):
    task = by_id[SMALL[0]]
    floor = mip.solve(task, floor=True)
    assert floor.objective == task.reference["floor"] < task.reference["hindsight"]
    assert all(d is None or d >= task.earliest(task.flight(i), task.flight(i).lb) for i, d in floor.deps.items())


def test_no_plan_beats_the_optimum(by_id):
    task = by_id[SMALL[0]]
    rng = random.Random(3)
    ids = [f.id for f in task.flights]
    for _ in range(60):
        plan = {i: {"cancel": rng.random() < 0.15, "not_before": task.start + rng.randrange(0, 600)}
                for i in rng.sample(ids, rng.randrange(1, len(ids)))}
        assert execute(task, plan).cost.total >= task.reference["hindsight"]


def test_pins_and_the_earliest_minute_are_kept(by_id):
    task = by_id[SMALL[0]]
    a, b = task.flights[3], task.flights[10]
    sol = mip.solve(task, fixed={a.id: None, b.id: task.deadline(b, b.lb)}, earliest=task.start + 120)
    assert sol.deps[a.id] is None and sol.deps[b.id] == task.deadline(b, b.lb)
    assert all(d is None or d >= task.start + 120 for i, d in sol.deps.items() if i not in (a.id, b.id))
    assert execute(task, sol.plan).cost.total >= task.reference["hindsight"]


def test_caps_bind_the_optimum(by_id):
    task = by_id[SMALL[0]]
    shut = {k: dict.fromkeys(range(13, 20), 0) for k in ("JFK:dep", "JFK:arr")}
    sol = mip.solve(task, caps=shut)
    run = execute(task, sol.plan, caps=shut)
    assert run.cost.total == sol.objective > task.reference["hindsight"]
    assert not any(r.dep // 60 in range(13, 20) for i, r in run.flights.items()
                   if r.status == FLOWN and task.flight(i).orig == "JFK")
