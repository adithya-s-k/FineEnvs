import random
from dataclasses import replace

import numpy as np
import pytest
from day_helpers import (
    broken_day,
    busy_day,
    crewed,
    day,
    flight,
    moved_day,
    network_day,
    quick_day,
    random_day,
    segment,
    shortened,
    stop,
    tail,
    three_into_a_program,
    turned_through_day,
)

from airops_core.programs import mip, references, sim
from airops_core.programs.cost import cancel_price
from airops_core.programs.model import Task

RANDOM = range(8)
NETWORK = range(8)


def days() -> dict[str, Task]:
    stopped = [flight(0, "BUF", "JFK", 120, 60, pax=150), flight(1, "DCA", "JFK", 110, 60, pax=40),
               flight(2, "JFK", "DCA", 250, 60), flight(3, "JFK", "BUF", 260, 60, pax=20)]
    return {"slots": three_into_a_program(lb=100),
            "stop": day(stopped, [tail("N1", "BUF", end="DCA"), tail("N2", "DCA", end="BUF")],
                        segments=(segment("g", 150, 1000, 150, 400, (190, 200, 260)),), stops=(stop(100, 160),)),
            "nominal": day([flight(0, "BUF", "JFK", 100, 60, lb=30), flight(1, "DCA", "JFK", 130, 60),
                            flight(2, "MCO", "JFK", 140, 60), flight(3, "JFK", "MCO", 200, 150, pax=180)],
                           [tail("N1", "BUF"), tail("N2", "DCA"), tail("N3", "MCO")], aar=1),
            "moved": moved_day(lb=60), "broken": broken_day(), "busy": busy_day(), "quick": quick_day(),
            "lone_quick": replace(quick_day(), tails=quick_day().tails[:1]), "families": families(),
            "crew": crew(), "turned_through": turned_through_day(),
            **{f"random{seed}": random_day(random.Random(seed)) for seed in RANDOM},
            **{f"network{seed}": network_day(random.Random(seed)) for seed in NETWORK}}


def crew() -> Task:
    """Behind a ground stop, N1's crew at BUF times out and rests to 730; N2 lands from DCA in time for JFK's reserves
    to take it to PWM, whose crew must leave by 480 or cancel, its rest ending past its deadline."""
    flights = [crewed(flight(0, "BUF", "JFK", 100, 60, lb=50, deadline=900), 130),
               crewed(flight(1, "JFK", "PWM", 250, 70, deadline=700), None),
               crewed(flight(2, "PWM", "JFK", 400, 70, deadline=640), 480),
               crewed(flight(3, "DCA", "JFK", 120, 60, deadline=820), 200)]
    return day(flights, [tail("N1", "BUF", end="JFK"), tail("N2", "DCA", end="JFK")], stops=(stop(100, 160),))


def families() -> Task:
    """A320s and E190s whose blocks overrun, behind a ground stop, each family with a ferry to end the day."""
    def e190(f):
        return replace(f, family="E190")

    flights = [flight(0, "BUF", "JFK", 100, 80, lb=20, sched_block=60), e190(flight(1, "DCA", "JFK", 110, 90, pax=90,
                                                                                     sched_block=70)),
               flight(2, "JFK", "MCO", 250, 160, sched_block=150), e190(flight(3, "JFK", "PWM", 260, 70, pax=60))]
    tails = [tail("N1", "BUF", end="BOS"), replace(tail("N2", "DCA", end="BOS"), family="E190", seats=100)]
    return replace(day(flights, tails, stops=(stop(100, 160),)), turn={"A320": 30, "E190": 25})


@pytest.fixture(scope="module", params=["task", "crewed"])
def fixture_day(request) -> Task:
    """The fixture day, and the same day with the crew model."""
    return shortened(request.getfixturevalue(request.param))


def feasible(net: mip.Network, x) -> bool:
    return all(lo - 1e-6 <= sum(v * x[c] for c, v in coefs.items()) <= up + 1e-6 for coefs, lo, up in net.m.rows)


def replays(task: Task, sol: mip.Solution) -> bool:
    run = sim.execute(task, sol.plan(task.start))
    return run.cost.total == sol.objective and all(
        (run.flights[i].dep if run.flights[i].status == "flown" else None) == d for i, d in sol.deps.items())


@pytest.mark.parametrize("name", list(days()))
def test_every_optimum_replays_exactly_in_the_sim(name):
    task = days()[name]
    sol = mip.solve(task)
    assert sol.proven and replays(task, sol)
    assert sol.objective <= sim.execute(task, {}).cost.total


def test_the_fixture_optimum_replays_and_no_plan_beats_it(fixture_day):
    task = fixture_day
    sol = mip.solve(task, start=sim.execute(task, {}))
    assert sol.proven and sol.started and replays(task, sol)
    rng = random.Random(5)
    plans = [{}, references.airline_plan(task)]
    for _ in range(12):
        ids = rng.sample([f.id for f in task.flights], rng.randrange(1, len(task.flights)))
        plans.append({i: {"cancel": rng.random() < 0.2, "decided": task.start + rng.randrange(0, 300),
                          "not_before": task.start + rng.randrange(0, 400)} for i in ids})
    for plan in plans:
        run = sim.execute(task, plan)
        assert run.cost.total >= sol.objective
        price = {f.id: cancel_price(f, task.start if run.flights[f.id].decided is None else run.flights[f.id].decided)
                 for f in task.flights}
        net = mip.Network(task, price=price)
        x = net.vector(run)
        assert x is not None and feasible(net, x)
        assert round(sum(c * v for c, v in zip(net.m.cost, x, strict=True))) == run.cost.total


@pytest.mark.parametrize("name", list(days()))
def test_every_day_the_sim_runs_is_a_solution_of_the_model_at_its_cost(name):
    task = days()[name]
    rng = random.Random(11)
    for _ in range(8):
        plan = {f.id: {"cancel": rng.random() < 0.2, "decided": rng.randrange(0, 200),
                       "not_before": rng.randrange(0, 400), "claim": rng.choice([None, 180, 240, 300])}
                for f in task.flights if rng.random() < 0.6}
        run = sim.execute(task, plan)
        decided = {f.id: run.flights[f.id].decided for f in task.flights}
        net = mip.Network(task, price={f.id: cancel_price(f, decided[f.id] or 0) for f in task.flights})
        x = net.vector(run)
        assert x is not None and feasible(net, x)
        assert round(sum(c * v for c, v in zip(net.m.cost, x, strict=True))) == run.cost.total


def test_a_day_with_legs_cancelled_on_either_side_of_its_breaks_is_a_solution_at_its_cost():
    task = broken_day()
    for plan in ({2: {"cancel": True}}, {1: {"cancel": True}, 2: {"cancel": True}}, {0: {"cancel": True}},
                 {0: {"cancel": True}, 1: {"cancel": True}}, {1: {"cancel": True}, 3: {"cancel": True}}):
        run = sim.execute(task, plan)
        net = mip.Network(task)
        x = net.vector(run)
        assert x is not None and feasible(net, x)
        assert round(sum(c * v for c, v in zip(net.m.cost, x, strict=True))) == run.cost.total
    sol = mip.solve(task, fixed={2: None})
    assert sol.deps[3] is None and replays(task, sol)


@pytest.mark.parametrize("name", ["slots", "stop", "moved", "broken", "busy", "quick", "crew", "random3", "network2"])
def test_a_day_with_its_history_pinned_is_a_solution_of_the_model_with_those_pins(name):
    task = days()[name]
    first = sim.execute(task, {})
    plan = {i: {"left": r.dep - 5} for i, r in first.flights.items() if r.status == "flown" and i % 2 == 0}
    run = sim.execute(task, plan)
    decided = {f.id: run.flights[f.id].decided for f in task.flights}
    net = mip.Network(task, price={f.id: cancel_price(f, decided[f.id] or 0) for f in task.flights},
                      fixed={i: run.flights[i].dep for i in plan})
    x = net.vector(run)
    assert x is not None and feasible(net, x)
    assert round(sum(c * v for c, v in zip(net.m.cost, x, strict=True))) == run.cost.total


def test_a_canonical_pass_out_of_time_keeps_the_proven_plan_flagged_as_not_canonical(monkeypatch):
    task = days()["slots"]
    proven = mip.solve(task, canonical=False)
    assert mip.solve(task).canonical and not proven.canonical
    run, calls = mip._Started.run, []

    def out_of_time(self, *args):
        calls.append(args)
        got = run(self, *args)
        return ("Time limit reached", *got[1:5], np.zeros_like(got[5])) if len(calls) == 2 else got

    monkeypatch.setattr(mip._Started, "run", out_of_time)
    sol = mip.solve(task, fix=False)
    assert len(calls) == 2 and sol.proven and not sol.canonical
    assert (sol.status, sol.objective, sol.deps) == ("Optimal", proven.objective, proven.deps)


def test_the_objective_is_the_cost_of_the_canonical_plan_when_it_beats_the_first_within_the_gap(monkeypatch):
    task = days()["moved"]
    best = mip.solve(task)
    run, calls = mip._Started.run, []

    def loose(self, *args):
        calls.append(args)
        got = run(self, *args)
        return (got[0], got[1] + 50, *got[2:]) if len(calls) == 1 else got

    monkeypatch.setattr(mip._Started, "run", loose)
    sol = mip.solve(task, fix=False)
    assert len(calls) == 2 and sol.canonical and best.objective <= sol.objective < best.objective + 50
    assert sim.execute(task, sol.plan(task.start)).cost.total == sol.objective


def test_a_coarse_step_keeps_every_slots_edct():
    task = three_into_a_program()
    sol = mip.solve(task, step=7)
    assert sol.objective == mip.solve(task).objective and replays(task, sol)
    assert {d for d in sol.deps.values()} == {120, 180, 240}


def test_pins_and_the_earliest_minute_are_kept(fixture_day):
    task = fixture_day
    a, b = task.flights[3], task.flights[10]
    sol = mip.solve(task, fixed={a.id: None, b.id: b.deadline}, earliest=task.start + 120)
    assert sol.deps[a.id] is None and sol.deps[b.id] == b.deadline
    assert all(d is None or d >= task.start + 120 for i, d in sol.deps.items() if i not in (a.id, b.id))


def test_a_faster_turn_jetblue_flew_lowers_the_optimum_and_the_optimum_uses_it():
    alone = replace(quick_day(), tails=quick_day().tails[:1])
    plain = replace(alone, tails=tuple(replace(t, connections=()) for t in alone.tails))
    sol = mip.solve(alone)
    assert sol.objective < mip.solve(plain).objective and replays(alone, sol)
    assert sol.deps[1] == 170
    net = mip.Network(alone)
    x = net.vector(sim.execute(alone, sol.plan(alone.start)))
    assert sum(x[c] for c in net.turn.values()) == 1


def test_the_turn_jetblue_flew_through_a_move_lowers_the_optimum_and_the_optimum_uses_it():
    task = turned_through_day()
    plain = turned_through_day(None)
    sol = mip.solve(task, fixed={1: None})
    assert sol.deps[2] == 100 and replays(task, sol)
    assert sol.objective < mip.solve(plain, fixed={1: None}).objective
    net = mip.Network(task)
    x = net.vector(sim.execute(task, {1: {"cancel": True}}))
    assert x is not None and feasible(net, x) and sum(x[c] for c in net.through.values()) == 1


def test_a_pinned_auto_cancel_carries_no_aircraft_through_its_break_and_a_pinned_cancellation_does():
    task = broken_day()
    lapsed = mip.solve(task, fixed={1: None}, lapsed={1})
    assert lapsed.lapsed == {1: task.flight(1).deadline + 1} and (lapsed.deps[2], lapsed.deps[3]) == (None, None)
    assert replays(task, lapsed)
    cancelled = mip.solve(task, fixed={1: None})
    assert not cancelled.lapsed and (cancelled.deps[2], cancelled.deps[3]) == (400, 640) and replays(task, cancelled)
    assert lapsed.objective > cancelled.objective


def test_the_flat_optimum_prices_an_auto_cancel_flat_as_the_flat_cost_does():
    task = broken_day()
    sol = mip.solve(task, fixed={1: None}, lapsed={1}, flat=True)
    run = sim.execute(task, sol.plan(task.start))
    assert sol.lapsed == {1: task.flight(1).deadline + 1} and run.flights[1].status == "auto_cancelled"
    assert sol.objective == run.cost.flat < run.cost.total
    net = mip.Network(task, flat=True)
    x = net.vector(run)
    assert x is not None and feasible(net, x)
    assert round(sum(c * v for c, v in zip(net.m.cost, x, strict=True))) == run.cost.flat


@pytest.mark.parametrize("name", list(days()))
def test_the_lp_first_canonical_plan_is_the_plan_without_it(name):
    task = days()[name]
    fast, slow = mip.solve(task), mip.solve(task, fix=False)
    assert fast.canonical and slow.canonical and fast.proven
    assert (fast.objective, fast.deps, fast.lapsed) == (slow.objective, slow.deps, slow.lapsed)


def test_the_lp_first_fixture_plan_is_the_plan_without_it(fixture_day):
    task = fixture_day
    fast, slow = mip.solve(task, start=sim.execute(task, {})), mip.solve(task, start=sim.execute(task, {}), fix=False)
    assert fast.canonical and (fast.objective, fast.deps) == (slow.objective, slow.deps)


def test_a_day_whose_lp_relaxation_is_fractional_proves_through_the_mip_to_the_same_plan():
    task = network_day(random.Random(21), 6)
    assert not mip.Network(task).m.relax(60, 1).integral
    fast, slow = mip.solve(task), mip.solve(task, fix=False)
    assert fast.proven and fast.canonical and replays(task, fast)
    assert (fast.objective, fast.deps) == (slow.objective, slow.deps)
