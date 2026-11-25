"""The lower bounds by hand (a unit's reach, its least crew cost, the covering bound's prices) and on random small days:
every crew-aware sim day, under any plan, flies each flight at a minute `reach` keeps, costs each unit at least its
least relaxed cost (its end of the day at its least share of its base's), and costs at least the bound; the best day
found replays exactly and is no cheaper than the bound."""

import random
import time
from dataclasses import replace

import numpy as np
import pytest
from day_helpers import day, flight, tail

from airops_core.core.sim import FLOWN
from airops_core.programs import crewbound, crewday, crewplan, crewreplay, crews, crewsim, sim
from airops_core.programs.crews import Unit


def unit(uid: str, kind: str = "pilot", station: str = "JFK", report: int | None = 540, **kw) -> Unit:
    hour = kw.pop("hour", report // 60 % 24 if report is not None else None)
    ready = kw.pop("ready", report + crews.BRIEF if report is not None else 0)
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), kw.pop("home", "JFK"), kw.pop("end", station), station,
                ready, report, hour, at=station, **kw)


def local(s: str, m: int) -> int:
    return m


def end_of(task):
    return lambda u, s: crews.reposition(u, s, task.airports, task.date)


def least_end(task, units):
    return crews.Ends(units, task.airports, task.date).least


def flights_of(task, world=None):
    world = world or sim.truth(task)
    air = crewplan.Aircraft(task, world)
    return air, crewbound.Flights.of(task, world.block, air.net.copies)


def test_a_unit_reaches_a_flight_until_it_times_out_and_again_after_rest():
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=2000)], [tail("N1", "JFK")])
    _, fl = flights_of(task)
    units = [unit("P", report=0, hour=0, ready=60, landed=40), unit("C", "cabin", report=0, hour=0, ready=60,
                                                                    landed=40)]
    ds = crewbound.reach(task, fl, units, local)[0]
    last = crews.pilot_limit(0, 0, 1) - 60
    rested = 40 + crews.DEBRIEF + crews.REST + crews.BRIEF
    assert ds.min() == 600 and ds[ds <= last].max() == last
    assert not ((ds > last) & (ds < rested)).any() and ds[ds >= rested].min() == rested


def test_a_flight_of_a_group_no_unit_flies_has_no_minute():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    _, fl = flights_of(task)
    assert len(crewbound.reach(task, fl, [unit("P")], local)[0]) == 0


def test_a_reserve_reaches_from_its_first_report_and_a_standby_leaves_by_its_shift_end():
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=2000)], [tail("N1", "JFK")])
    _, fl = flights_of(task)
    short = [unit(k, kind, report=None, hour=None, reserve="short", rap=480, until=900, lead=150)
             for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    ds = crewbound.reach(task, fl, short, local)[0]
    assert ds.min() == crews.on_hour(480 + 150) + crews.BRIEF
    standby = [unit(k, kind, report=300, reserve="standby", until=660) for k, kind in (("SP", "pilot"),
                                                                                       ("SC", "cabin"))]
    ds = crewbound.reach(task, fl, standby, local)[0]
    assert ds.min() == 600 and ds[ds < 660 + crews.REST].max() == 660


def test_a_unit_flies_home_free_rests_for_it_at_a_hotel_or_pays_its_end_of_day():
    task = day([flight(0, "BOS", "JFK", 1000, 60, pax=100, deadline=1100), flight(1, "DCA", "MCO", 700, 120)],
               [tail("N1", "BOS"), tail("N2", "DCA")])
    _, fl = flights_of(task)
    units = [unit("P", station="BOS", end="JFK"), unit("Q", station="BUF", end="JFK"),
             unit("C", "cabin", station="BOS", end="JFK", report=0, hour=0, ready=60)]
    reached = {f.id: fl.minutes[f.id] for f in task.flights}
    least = crewbound.least(task, fl, units, local, end_of(task), reached)
    assert least[0] == pytest.approx(0.0)
    assert least[1] == pytest.approx(crews.reposition(units[1], "BUF", task.airports, task.date))
    assert 1000 + 60 + crews.DEBRIEF > crews.CABIN_DUTY and 0 + crews.REST + crews.BRIEF <= 1000
    assert least[2] == pytest.approx(crews.hotel("BOS", task.date) * units[2].size)


def test_a_choice_the_lp_takes_is_free_and_later_minutes_cost_their_reduced_cost():
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=700)], [tail("N1", "JFK")])
    air, _ = flights_of(task)
    lp = air.relax(air.notice, air.ub)
    ch = crewbound.prices(air, lp, air.ub, np.inf)[0]
    assert ch[0][0] == 600 and ch[1][0] == pytest.approx(0.0)
    assert np.all(np.diff(ch[1]) >= -1e-9) and ch[1][-1] > 0 and np.isfinite(ch[2])


STATIONS = ("JFK", "BOS", "DCA", "MCO", "BUF")


def random_day(seed: int):
    rng = random.Random(seed)
    flights = []
    for i in range(rng.randint(5, 9)):
        a, b = rng.sample(STATIONS, 2)
        dep = rng.randrange(360, 1300, 5)
        flights.append(flight(i, a, b, dep, rng.randrange(50, 180, 5), pax=rng.randint(100, 150),
                              deadline=dep + rng.choice((120, 240, 480))))
    tails = [tail(f"N{f.id}", f.orig) for f in flights]
    task = day(flights, tails, start=300)
    by = sorted(flights, key=lambda f: f.sched_dep)
    units = []
    for n in range(rng.randint(2, 4)):
        trip = tuple(sorted((f.id for f in rng.sample(by, rng.randint(1, 3))), key=lambda i: task.flight(i).sched_dep))
        first = task.flight(trip[0])
        rep = first.sched_dep - crews.REPORT
        for kind in crews.KINDS:
            units.append(unit(f"{kind[0]}{n}", kind, first.orig, rep, trip=trip, end=rng.choice(STATIONS)))
    for kind in crews.KINDS:
        units.append(unit(f"S{kind}", kind, "JFK", 300, reserve="standby", until=660))
        units.append(unit(f"R{kind}", kind, "JFK", None, hour=None, reserve="short", rap=rng.choice((120, 780)),
                          until=900, lead=150))
    return task, units


def played(task, units, rng, n=6):
    """Crew sim days under random plans: holds, cancellations, none; crewed by the trip rule or the crew desk."""
    world = sim.truth(task)
    days = [crewsim.execute(task, {}, units, world, local, desk=desk) for desk in (False, True)]
    for k in range(n):
        plan = {}
        for f in task.flights:
            r = rng.random()
            if r < 0.15:
                plan[f.id] = {"cancel": True}
            elif r < 0.5:
                plan[f.id] = {"not_before": f.sched_dep + rng.randrange(0, 300, 10)}
        days.append(crewsim.execute(task, plan, units, world, local, desk=k % 2 == 1))
    return days


def spent(task, units, d: crewsim.CrewDay) -> dict[str, float]:
    """Each unit's crew cost in the day, its end of the day its share of its base's."""
    at = {u.id: d.units[u.id]["station"] if u.id in d.units else u.station for u in units}
    out = crews.Ends(units, task.airports, task.date).cost(at)
    for uid, r in d.units.items():
        out[uid] += r["cost"]["deadhead"] + r["cost"]["hotel"]
        assert r["cost"]["end"] == pytest.approx(out[uid] - r["cost"]["deadhead"] - r["cost"]["hotel"])
    return out


@pytest.mark.parametrize("seed", range(10))
def test_every_sim_day_flies_where_reach_keeps_and_costs_each_unit_its_least_at_least(seed):
    task, units = random_day(seed)
    air, fl = flights_of(task)
    reached = crewbound.reach(task, fl, units, local)
    least = crewbound.least(task, fl, units, local, least_end(task, units), reached)
    for d in played(task, units, random.Random(seed)):
        for i, r in d.run.flights.items():
            if r.status == FLOWN and i not in d.uncrewed:
                assert r.dep in set(reached[i].tolist()), (i, r.dep)
        cost = spent(task, units, d)
        for b, u in enumerate(units):
            assert cost[u.id] >= least[b] - 1e-6, u.id


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("flat", [False, True])
def test_no_sim_day_costs_less_than_the_bound_and_the_best_day_replays_exactly(seed, flat):
    task, units = random_day(seed)
    world = sim.truth(task)
    air, fl = flights_of(task, world)
    reached = crewbound.reach(task, fl, units, local)
    least = crewbound.least(task, fl, units, local, least_end(task, units), reached)
    j = crewplan.joint(task, world, units, local, task.airports, air, fl, reached, least, 60, 60, flat=flat,
                       crew_limit=20, days=(("naive", {}),))
    assert j.found.exact and j.cover.converged
    assert j.bound >= j.optimum.bound and j.bound <= j.found.value + 1e-6
    for d in [*played(task, units, random.Random(seed)), j.found.day]:
        assert crewplan.worth(d, flat) >= j.bound - 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_a_sim_days_crews_are_crew_replay_columns_at_their_cost_and_start_it(seed):
    task, units = random_day(seed)
    world = sim.truth(task)
    d = crewsim.execute(task, {}, units, world, local)
    deps = {i: r.dep for i, r in d.run.flights.items() if r.status == FLOWN}
    p = crewreplay.Problem(crewday.task_legs(task, deps, world.block), units, task.date, local, end_of(task),
                           call=task.start, ends=crews.Ends(units, task.airports, task.date))
    cols = crewday.columns_of(p, d)
    cost = p.crew_cost({u.id: p.priced(u, cols[u.id]) if u.id in cols else p.idles(u)[0] for u in units})
    assert cost == pytest.approx(d.crew_cost)
    s = crewreplay.solve(p, time_limit=20, polish=5, starts=[cols])
    started = sum(p.priced(u, cols[u.id]).cost if u.id in cols else p.idles(u)[0].cost for u in units)
    assert s.uncrewed == () and s.objective <= started + 1e-6


def test_a_pilot_out_of_flight_time_reaches_the_flight_only_after_rest():
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=2000)], [tail("N1", "JFK")])
    _, fl = flights_of(task)
    tired = unit("P", report=540, minutes=crews.flight_time(9) - 59, flown=1, landed=560, ready=580)
    ds = crewbound.reach(task, fl, [tired, unit("C", "cabin")], local)[0]
    assert ds.min() == 560 + crews.DEBRIEF + crews.REST + crews.BRIEF
    assert crewbound.reach(task, fl, [replace(tired, minutes=0), unit("C", "cabin")], local)[0].min() == 600


@pytest.mark.parametrize("seed", range(6))
def test_every_sim_days_crews_are_relaxed_days_at_their_cost(seed):
    task, units = random_day(seed)
    air, fl = flights_of(task)
    reached = crewbound.reach(task, fl, units, local)
    least = least_end(task, units)
    board = crewbound.Board(task, fl, crewbound.Legs.of(fl, reached), units, local, least)
    for d in played(task, units, random.Random(seed)):
        cost = spent(task, units, d)
        days = board.days_of(d)
        assert {units[b].id for b, _ in days} == set(d.units)
        for b, relaxed in days:
            r = d.units[units[b].id]
            assert relaxed.cost == pytest.approx(r["cost"]["deadhead"] + r["cost"]["hotel"]
                                                 + least(units[b], r["station"]))
            assert relaxed.cost <= cost[units[b].id] + 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_the_best_days_departures_crewed_again_replay_at_their_claimed_cost(seed):
    task, units = random_day(seed)
    world = sim.truth(task)
    air, _ = flights_of(task, world)
    a, _ = air.optimum(air.notice, air.ub)
    found = crewplan.search(task, world, units, local, task.airports, air, [("optimum", a)], time.monotonic() + 200,
                            crew_limit=10, days=(("naive", {}),))
    again = [t for t in found.attempts if t["try"] == "best day, crews again"]
    assert found.exact and len(again) == 1
    assert not again[0]["uncrewed"] and again[0]["claimed"] == pytest.approx(again[0]["replayed"], abs=0.5)


@pytest.mark.parametrize("seed", range(6))
def test_the_lagrangian_at_the_lps_duals_is_the_lp_plus_the_units_least_reduced_costs(seed):
    task, units = random_day(seed)
    world = sim.truth(task)
    air, fl = flights_of(task, world)
    reached = crewbound.reach(task, fl, units, local)
    keep = {i: set(v.tolist()) for i, v in reached.items()}
    up = air.upper(keep=keep)
    lp = air.relax(air.notice, up)
    choices = crewbound.prices(air, lp, up, 2e5)
    legs = crewbound.Legs.of(fl, {i: c[0] for i, c in choices.items()}, {i: c[1] for i, c in choices.items()})
    board = crewbound.Board(task, fl, legs, units, local, lambda u, s: 0.0)
    m = crewbound.Cover(fl, units, choices, board, crews.Ends(units, task.airports, task.date))
    for _ in range(3):
        z, pi, mu, sigma = m.solve()
        value, days = board.price({k: pi[k][legs.flight] - legs.price / 2 for k in crews.KINDS}, trace=True,
                                  ends=m.ends_at(sigma))
        assert m.lagrangian(pi, sigma, value) == pytest.approx(z + np.minimum(value - mu, 0).sum(), rel=1e-6,
                                                               abs=1e-3)
        for b, best in enumerate(days):
            assert m.reduced(b, best, pi, mu, sigma) == pytest.approx(value[b] - mu[b], abs=1e-6)
            m.add(b, best)


def test_the_covering_bound_sends_each_bases_crews_to_its_trips_tonight_together():
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=700), flight(1, "JFK", "BOS", 610, 60, deadline=700)],
               [tail("N1", "JFK"), tail("N2", "JFK")])
    _, fl = flights_of(task)
    units = [unit(f"{k}{n}", kind, trip=(n,), end=end) for n, end in ((0, "JFK"), (1, "DCA"))
             for k, kind in (("P", "pilot"), ("C", "cabin"))]
    ends = crews.Ends(units, task.airports, task.date)
    choices = {f.id: (np.array([f.sched_dep]), np.array([0.0]), np.inf) for f in task.flights}
    covered = crewbound.cover(task, fl, units, local, ends, choices, time_limit=30)
    both = sum(ends.move(u, "BOS", u.end) for u in units)
    assert covered.converged and covered.bound == pytest.approx(both)
    assert both > sum(ends.least(u, "BOS") for u in units)


@pytest.mark.parametrize("seed", range(8))
def test_a_crew_desk_day_named_replays_exactly(seed):
    task, units = random_day(seed)
    world = sim.truth(task)
    d = crewsim.execute(task, {}, units, world, local, desk=True)
    again = crewsim.execute(task, crewday.named(d), units, world, local, fill=False)
    assert again.total == pytest.approx(d.total) and again.uncrewed == d.uncrewed == ()
