import math
import statistics

from airops_core.core.known import CAP_LOOK, LOOK, known_caps, known_lb
from airops_core.core.sim import as_plan, execute
from airops_core.refs import follower, policies

SMALL = "b6-jfk-20230626-1400-1700"


def test_with_perfect_forecasts_the_follower_lands_on_hindsight(by_id, monkeypatch):
    task = by_id[SMALL]
    monkeypatch.setattr(follower, "forecast_lb", lambda task, t, mode: task.lb)
    monkeypatch.setattr(follower, "forecast_caps", lambda task, t, mode: task.caps)
    run = follower.follow(task)
    assert run.cost == task.reference["hindsight"]
    assert all(w["planned"] == task.reference["hindsight"] for w in run.watches)


def test_the_follower_is_deterministic_and_its_plan_replays(by_id):
    task = by_id[SMALL]
    run = follower.follow(task)
    assert run.cost == task.reference["follower"]
    assert run.stored() == task.reference["follower_plan"]
    assert execute(task, as_plan(run.stored())).cost.total == run.cost
    assert run.watches[-1]["planned"] == run.cost
    assert [w["frozen"] for w in run.watches] == sorted(w["frozen"] for w in run.watches)


def test_persistence_carries_recent_delays_and_caps_forward(by_id):
    task = by_id["b6-jfk-20230626-1400"]
    t = task.start + 60
    lb = follower.forecast_lb(task, t, follower.PERSIST)
    recent = [f for f in task.flights if t - 60 <= f.sched_dep <= t + LOOK]
    for f in task.flights:
        if f.sched_dep <= t + LOOK:
            assert lb[f.id] == f.lb
        else:
            pool = [g.lb for g in recent if g.orig == f.orig] or [g.lb for g in recent]
            assert lb[f.id] == round(statistics.mean(pool))
    caps = follower.forecast_caps(task, t, follower.PERSIST)
    for key, hours in caps.items():
        known = sorted(h for h in task.caps[key] if 60 * h <= t + CAP_LOOK)
        guess = math.ceil(statistics.mean(task.caps[key][h] for h in known[-3:]))
        for h, v in hours.items():
            assert v == (task.caps[key][h] if h in known else guess)


def test_blind_plans_on_the_day_as_known(by_id):
    task = by_id["b6-jfk-20230626-1400"]
    for t in (task.start, task.start + 180):
        assert follower.forecast_lb(task, t, follower.BLIND) == known_lb(task, t)
        assert follower.forecast_caps(task, t, follower.BLIND) == known_caps(task, t)


def test_the_airline_plan_is_what_jetblue_did(by_id):
    task = by_id["b6-jfk-20230626-1400"]
    plan = policies.airline(task)
    assert {i for i, v in plan.items() if v["cancel"]} == {f.id for f in task.flights if f.actual.cancelled}
    run = execute(task, plan)
    assert sum(r.status != "flown" for r in run.flights.values()) >= sum(f.actual.cancelled for f in task.flights)
    assert policies.naive(task) == {}
