"""The aircraft MIP kept in HiGHS: priced flat, held, cancelled or kept to minutes by column bounds, it finds the
optimum the MIP built again with those prices and lower bounds finds, and its plans replay exactly in the sim."""

import random
from dataclasses import replace

import pytest
from day_helpers import broken_day, moved_day, network_day, quick_day, random_day

from airops_core.core.sim import FLOWN
from airops_core.programs import crewplan, mip, sim
from airops_core.programs.model import Task


def tasks() -> dict[str, Task]:
    return {"moved": moved_day(lb=60), "broken": broken_day(), "quick": quick_day(),
            **{f"random{s}": random_day(random.Random(s)) for s in range(4)},
            **{f"network{s}": network_day(random.Random(s)) for s in range(4)}}


def replays(task: Task, sol: mip.Solution, world: sim.World) -> bool:
    run = sim.execute(task, sol.plan(task.start), world)
    return all((run.flights[i].dep if run.flights[i].status == FLOWN else None) == d for i, d in sol.deps.items())


@pytest.mark.parametrize("name", list(tasks()))
def test_the_kept_mip_finds_the_optimum_notice_priced_flat_and_held(name):
    task = tasks()[name]
    world = sim.truth(task)
    air = crewplan.Aircraft(task, world)
    a, _ = air.optimum(air.notice, air.ub)
    assert a.objective == mip.solve(task).objective == sim.execute(task, a.plan(task.start)).cost.total
    flat, _ = air.optimum(air.flat, air.ub)
    assert flat.objective == mip.solve(task, flat=True, canonical=False).objective
    flown = [i for i, d in a.deps.items() if d is not None]
    i = flown[0]
    held = {i: a.deps[i] + 25}
    sol, _ = air.solve(air.notice, air.upper(held=held), time_limit=60)
    lb = dict(world.lb)
    lb[i] = max(lb[i], held[i] - task.flight(i).sched_dep)
    later = replace(world, lb=lb)
    assert sol.objective == mip.solve(task, later, canonical=False).objective and replays(task, sol, later)
    gone, _ = air.solve(air.notice, air.upper(dropped={i}), time_limit=60)
    assert gone.deps[i] is None
    assert gone.objective == mip.solve(task, fixed={i: None}, canonical=False).objective


@pytest.mark.parametrize("seed", range(4))
def test_a_plan_found_with_cancellations_priced_to_fly_is_valued_at_the_true_prices(seed):
    task = network_day(random.Random(seed))
    air = crewplan.Aircraft(task, sim.truth(task))
    a, _ = air.optimum(air.notice, air.ub)
    sol, _ = air.solve(air.cost(fly=[f.id for f in task.flights]), air.ub, value=air.notice, time_limit=60)
    assert sum(d is None for d in sol.deps.values()) <= sum(d is None for d in a.deps.values())
    assert sol.objective == sim.execute(task, sol.plan(task.start)).cost.total >= a.objective
