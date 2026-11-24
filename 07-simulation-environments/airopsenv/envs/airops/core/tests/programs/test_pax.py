import importlib.util
import random
from dataclasses import replace

import numpy as np
import pytest
from day_helpers import day, flight, network_day, random_day, tail
from program_helpers import TOOLS
from test_hindsight_mip import feasible
from v2_digest import DATE, PACK, digests

from airops_core.occ.pack import Pack
from airops_core.programs import crews, crewsim, mip, pax, sim
from airops_core.programs.cost import REFUND
from airops_core.programs.pax import Group, Passengers, Seat

SPEC = importlib.util.spec_from_file_location("passenger_bound", TOOLS / "passenger_bound.py")
pax_bound = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pax_bound)
V2 = {"naive": "8fe4c8699ae9d9414885459fb5b98b2b170bb00b74e8500006f667ab5c626092",
      "B0": "dd5d95c9ea7b50abf649143b257d279641466bff4ae4d01219d444897da7f410",
      "best": "65f95cb9c70a4d5ed7f807da09bb4a9fb3ad4277ed64d43a8e0a9d4d402aea94",
      "mip": "e9e25057c8e6ce9493014919532012c04dc930a7aa05cce8015cb2c811e58715"}


def test_passengers_off_leave_every_v2_output_byte_identical_on_a_pack_date():
    """The digests are v2's own (this code at v0.3.3, before v3) on the same date."""
    assert digests() == V2
    pack = Pack(PACK)
    w = pack.world(next(t for t, v in pack.tasks.items() if v["date"] == DATE))
    b0 = crewsim.execute(w.task, {}, w.units, local=w.local, airports=w.airports, desk=True)
    best = crewsim.execute(w.task, w.best_plan, w.units, local=w.local, airports=w.airports, fill=False)
    assert b0.total == pytest.approx(w.refs["B0"]["notice"], abs=0.05)
    assert best.total == pytest.approx(w.refs["best"], abs=0.5)
    assert set(b0.run.cost.to_dict()) == {"delay", "cancel", "notice", "ferry", "gate", "overrun", "saved", "total",
                                          "flat"}


def test_a_pack_date_books_its_loads_deterministically_into_connections_as_sold():
    pack = Pack(PACK)
    w = pack.world(next(t for t, v in pack.tasks.items() if v["date"] == DATE))
    task = w.task
    flows = pax.Flows.load(DATE)
    one, two = (pax.book(task, flows, pax.Dwell.load()) for _ in range(2))
    assert one == two and flows.label == "DB1B 2025Q1"
    booked = one.booked()
    assert all(booked[f.id] == f.pax for f in task.flights)
    pairs = [g for g in one.groups if len(g.legs) == 2]
    assert pairs and sum(g.pax for g in pairs) < 0.1 * sum(g.pax for g in one.groups)
    for g in pairs:
        i, j = task.flight(g.legs[0]), task.flight(g.legs[1])
        assert i.dest == j.orig in pax.CONNECT and i.orig != j.dest
        assert pax.mct(j.orig) <= j.sched_dep - i.sched_arr <= pax.DWELL
    double = pax.book(task, flows.scaled(2), pax.Dwell.load())
    assert sum(g.pax for g in double.groups if len(g.legs) == 2) > sum(g.pax for g in pairs)


def test_a_date_reads_its_own_survey_period_or_the_nearest():
    periods = {"2023Q3", "2024Q3", "2025Q2", "2025-07", "2026-01"}
    assert pax.period("2024-07-13", periods) == "2024Q3"
    assert pax.period("2025-07-08", periods) == "2025-07"
    assert pax.period("2025-06-30", periods) == "2025Q2"
    assert pax.period("2026-01-19", periods) == "2026-01"
    assert pax.period("2024-06-07", periods) == "2024Q3"
    assert pax.period("2025-12-02", periods) == "2026-01"


def connecting_day(lb: int = 0, cancel: int | None = None) -> tuple:
    """BUF-JFK lands at 360 (+ lb) for 20 connecting onto JFK-PWM at 420; the next JFK-PWM is at 600."""
    flights = [flight(0, "BUF", "JFK", 300, 60, pax=100, lb=lb, deadline=900),
               flight(1, "JFK", "PWM", 420, 60, pax=100, deadline=900),
               flight(2, "JFK", "PWM", 600, 60, pax=100, deadline=900)]
    task = day(flights, [tail("N1", "BUF", end="JFK"), tail("N2", "JFK", end="PWM"), tail("N3", "JFK", end="PWM")],
               start=240)
    groups = (Group(0, (0,), 80), Group(1, (0, 1), 20), Group(2, (1,), 80), Group(3, (2,), 100))
    plan = {} if cancel is None else {cancel: {"cancel": True, "decided": 240}}
    return task, Passengers(groups), plan


def test_a_missed_connection_is_rebooked_onto_the_next_flight_and_its_delay_counted_at_the_destination():
    task, passengers, plan = connecting_day(lb=60)
    run = sim.execute(task, plan, pax=passengers)
    d = run.pax
    assert d.misconnected() == 20 and d.outcomes() == {"flew": 260, "rebooked": 20, "stranded": 0, "refunded": 0}
    rebooked = next(p for p in d.parts if p.kind == "rebooked")
    assert rebooked.legs == (0, 2) and rebooked.delay == 660 - 480
    assert run.cost.delay == d.delay == 80 * 60 + 20 * 180 and run.cost.passengers
    assert (run.cost.to_dict()["lodging"], run.cost.to_dict()["fare"], run.cost.cancel) == (0, 0, 0)


def test_crews_keep_their_seats_and_who_finds_none_today_takes_a_later_days_first_seat_a_night_each():
    task, passengers, plan = connecting_day(lb=60)
    runs = sim.execute(task, plan).flights
    d = pax.fly(task, passengers, runs, {2: 40})
    assert d.outcomes() == {"flew": 260, "rebooked": 10, "stranded": 10, "refunded": 0}
    night = next(p for p in d.parts if p.kind == "stranded")
    assert (night.at, night.arr, night.delay, night.nights) == ("JFK", 480 + 1440, 1440, 1)
    assert d.lodging == round(10 * crews.lodging("JFK", task.date))
    later = (Seat("B6 9 JFK +1", "JFK", "PWM", 1440 + 360, 1440 + 420, 4, 1),
             Seat("B6 11 JFK +2", "JFK", "PWM", 2880 + 120, 2880 + 180, 50, 2))
    d = pax.fly(task, replace(passengers, later=later), runs, {2: 40})
    assert sorted((p.arr, p.pax, p.nights) for p in d.parts if p.kind == "stranded") == [(1860, 4, 1), (3060, 6, 2)]
    assert d.lodging == round(4 * crews.lodging("JFK", task.date) + 12 * crews.lodging("JFK", task.date))


def test_who_finds_no_jetblue_seat_within_five_days_is_refunded_for_the_wait_and_the_fare():
    task, passengers, plan = connecting_day(cancel=1)
    later = (Seat("B6 9 JFK +1", "JFK", "PWM", 1800, 1860, 0, 1),)
    passengers = replace(passengers, later=later, fare=200.0)
    run = sim.execute(task, plan, pax=passengers)
    d = run.pax
    assert d.outcomes() == {"flew": 180, "rebooked": 50, "stranded": 0, "refunded": 50}
    assert {(p.group, p.pax, p.delay) for p in d.parts if p.kind == "refunded"} == {(2, 30, 0), (1, 20, 360 - 300)}
    assert d.fare == run.cost.fare == 50 * 200 and run.cost.cancel == REFUND * 100


def earlier_day(booked: int) -> tuple:
    """JFK-PWM at 420 with 50 seats free; the 100 booked on JFK-PWM at `booked` lose it to a cancellation at 240."""
    flights = [flight(0, "JFK", "PWM", 420, 60, pax=100, deadline=900),
               flight(1, "JFK", "PWM", booked, 60, pax=100, deadline=900)]
    task = day(flights, [tail("N1", "JFK", end="PWM"), tail("N2", "JFK", end="PWM")], start=240)
    return task, Passengers((Group(0, (0,), 100), Group(1, (1,), 100)), fare=200.0)


def test_a_cancellation_rebooks_from_its_decision_onto_a_flight_less_than_3_h_earlier_for_free():
    task, passengers = earlier_day(590)
    d = sim.execute(task, {1: {"cancel": True, "decided": 240}}, pax=passengers).pax
    early = next(p for p in d.parts if p.group == 1 and p.kind == "rebooked")
    assert (early.legs, early.pax, early.delay) == ((0,), 50, 0)


def test_a_flight_3_h_or_more_earlier_is_taken_only_when_no_later_seat_is_left_within_five_days():
    task, passengers = earlier_day(600)
    plan = {1: {"cancel": True, "decided": 240}}
    d = sim.execute(task, plan, pax=passengers).pax
    assert {(p.kind, p.pax) for p in d.parts if p.group == 1} == {("stranded", 100)}
    full = replace(passengers, later=(Seat("B6 9 JFK +1", "JFK", "PWM", 1800, 1860, 0, 1),))
    d = sim.execute(task, plan, pax=full).pax
    assert {(p.kind, p.pax, p.legs) for p in d.parts if p.group == 1} == {("rebooked", 50, (0,)), ("refunded", 50, ())}
    best = pax.optimal(task, full, sim.execute(task, plan).flights)
    assert {(p.kind, round(p.pax)) for p in best.parts if p.group == 1} == {("rebooked", 50), ("refunded", 50)}


def test_a_cancellation_rebooks_first_come_first_served_and_costs_the_refund_exposure_not_523():
    task, passengers, plan = connecting_day(cancel=1)
    run = sim.execute(task, plan, pax=passengers)
    d = run.pax
    assert d.outcomes() == {"flew": 180, "rebooked": 50, "stranded": 50, "refunded": 0}
    assert {(p.group, p.kind, p.pax) for p in d.parts if p.kind != "flew"} == {
        (2, "rebooked", 50), (2, "stranded", 30), (1, "stranded", 20)}
    assert d.outcomes()["refunded"] == 0
    assert run.cost.cancel == REFUND * 100
    assert sim.execute(task, plan).cost.cancel == 523 * 100


def test_the_default_rebooking_takes_the_earliest_arrival_where_an_optimal_assignment_saves_a_later_passenger():
    """X, off a cancelled flight at 240, takes the earliest arrival's last seats; Y, whose outbound is cancelled too,
    lands at 415, too late for the slow flight at 420, and finds the fast one full. The optimal assignment swaps
    them."""
    flights = [flight(0, "BUF", "JFK", 300, 115, pax=100, deadline=900),
               flight(1, "JFK", "PWM", 300, 60, pax=10, deadline=900),
               flight(2, "JFK", "PWM", 430, 60, pax=10, deadline=900),
               flight(3, "JFK", "PWM", 420, 140, pax=140, deadline=900),
               flight(4, "JFK", "PWM", 480, 60, pax=140, deadline=900)]
    task = day(flights, [tail(f"N{i}", f.orig, end=f.dest) for i, f in enumerate(flights)], start=240)
    passengers = Passengers((Group(0, (1,), 10), Group(1, (0, 2), 10), Group(2, (0,), 90), Group(3, (3,), 140),
                             Group(4, (4,), 140)), fare=200.0)
    runs = sim.execute(task, {1: {"cancel": True, "decided": 240}, 2: {"cancel": True, "decided": 240}}).flights
    fcfs, best = pax.fly(task, passengers, runs), pax.optimal(task, passengers, runs)
    assert next(p for p in fcfs.parts if p.group == 0).legs == (4,)
    assert next(p for p in fcfs.parts if p.group == 1).kind == "stranded"
    assert next(p for p in best.parts if p.group == 0).legs == (3,)
    assert next(p for p in best.parts if p.group == 1).legs == (0, 4)
    assert best.delay == 10 * (560 - 360) + 10 * (540 - 490) < fcfs.delay


def mixed(seed: int):
    task = network_day(random.Random(seed), 6) if seed % 2 else random_day(random.Random(seed), 4)
    return task, pax.book(task, pax.Flows.synthetic(task, 0.4), pax.Dwell.uniform(), refunded=250.0)


@pytest.mark.parametrize("seed", range(10))
def test_every_day_the_sim_runs_is_a_solution_of_the_passenger_model_at_its_proxy_cost(seed):
    task, passengers = mixed(seed)
    rng = random.Random(seed)
    for _ in range(5):
        plan = {f.id: {"cancel": rng.random() < 0.25, "decided": rng.randrange(0, 200),
                       "not_before": rng.randrange(0, 400)} for f in task.flights if rng.random() < 0.6}
        run = sim.execute(task, plan)
        proxy = pax.Proxy(task, passengers, {i: r.decided for i, r in run.flights.items() if r.decided is not None})
        net = mip.Network(task, price={f.id: proxy.price(f, run.flights[f.id].decided or 0) for f in task.flights},
                          pax=proxy)
        x = net.vector(run)
        assert x is not None and feasible(net, x)
        assert float(np.dot(net.m.cost, x)) == pytest.approx(proxy.cost(run), abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_every_passenger_optimum_replays_at_its_objective(seed):
    task, passengers = mixed(seed)
    proxy = pax.Proxy(task, passengers)
    sol = mip.solve(task, pax=proxy, start=sim.execute(task, {}))
    run = sim.execute(task, sol.plan(task.start), pax=passengers)
    assert sol.proven and abs(sol.objective - proxy.cost(run)) <= 1
    assert sol.objective <= proxy.cost(sim.execute(task, mip.solve(task).plan(task.start))) + 1
    assert sol.bound <= sol.objective + 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_the_rounded_relaxation_is_a_plan_the_sim_plays_and_a_start_the_model_holds(seed):
    task, passengers = mixed(seed)
    proxy = pax.Proxy(task, passengers)
    net = mip.Network(task, pax=proxy)
    run = sim.execute(task, net.rounded())
    x = net.vector(run)
    assert x is not None and feasible(net, x)
    assert float(np.dot(net.m.cost, x)) == pytest.approx(proxy.cost(run), abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_the_capacity_aware_floor_is_at_most_p_on_every_day_the_sim_plays(seed):
    task, passengers = mixed(seed)
    arrive = pax_bound.arrivals(task, sim.truth(task))
    lower = pax.Proxy.lower(task, passengers, arrive)
    rng = random.Random(seed)
    plans = [{}, mip.solve(task, pax=pax.Proxy(task, passengers), start=sim.execute(task, {})).plan(task.start)]
    for _ in range(10):
        plans.append({f.id: {"cancel": rng.random() < 0.3, "decided": rng.randrange(0, 200),
                             "not_before": rng.randrange(0, 400)} for f in task.flights if rng.random() < 0.6})
    played = []
    for plan in plans:
        run = sim.execute(task, plan, pax=passengers)
        assert all(r.arr >= arrive[i] for i, r in run.flights.items() if r.status == "flown")
        assert lower.cost(run) <= run.cost.total + 1e-6
        played.append(run.cost.total)
    sol = mip.solve(task, pax=lower, start=sim.execute(task, {}))
    assert sol.proven and sol.bound <= sol.objective + 1e-6 <= min(played) + 2e-6
