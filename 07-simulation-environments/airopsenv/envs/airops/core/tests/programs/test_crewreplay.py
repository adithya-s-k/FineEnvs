"""The crew replay by hand (trips, uncrewed legs, deadhead seats, Table A over actual blocks, short-call reports, the
rest move) and its agreement with the crew sim on random small days: every plan passes the sim's rules move by move
and replays there at the same crew cost, and no plan beats the bound."""

import random
from dataclasses import replace

import pytest
from day_helpers import day, flight, tail

from airops_core.programs import crewday, crewreplay, crews, crewsim
from airops_core.programs.crewreplay import DH, OP, Problem
from airops_core.programs.crews import Unit


def unit(uid: str, kind: str = "pilot", station: str = "JFK", report: int | None = 540, **kw) -> Unit:
    hour = kw.pop("hour", report // 60 % 24 if report is not None else None)
    ready = kw.pop("ready", report + crews.BRIEF if report is not None else 0)
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), kw.pop("home", "JFK"), kw.pop("end", station), station,
                ready, report, hour, at=station, **kw)


def problem(task, units, deps=None, blocks=None, **kw) -> Problem:
    deps = deps or {f.id: f.sched_dep for f in task.flights}
    blocks = blocks or {f.id: f.block for f in task.flights}
    return Problem(crewday.task_legs(task, deps, blocks), units, task.date, lambda s, m: m,
                   lambda u, s: crews.reposition(u, s, task.airports, task.date), call=task.start,
                   ends=crews.Ends(units, task.airports, task.date), **kw)


def solve(task, units, **kw) -> tuple[Problem, crewreplay.Solved]:
    p = problem(task, units, **kw)
    return p, crewreplay.solve(p, time_limit=60, polish=10)


def test_trip_crews_fly_their_trip_at_no_cost():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 720, 60)], [tail("N1", "JFK")])
    units = [unit("P1", trip=(0, 1)), unit("C1", "cabin", trip=(0, 1))]
    p, s = solve(task, units)
    assert s.uncrewed == () and s.crew_cost == pytest.approx(0)
    assert s.columns["P1"].first == ((0, OP), (1, OP)) and s.crews(p) == {i: {"pilot": "P1", "cabin": "C1"}
                                                                          for i in (0, 1)}
    assert s.bound <= s.objective + 1e-6


def test_a_leg_no_crew_can_reach_is_uncrewed_and_bounded():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "DCA", "MCO", 610, 120)],
               [tail("N1", "JFK"), tail("N2", "DCA")])
    p, s = solve(task, [unit("P1", trip=(0,)), unit("C1", "cabin", trip=(0,))])
    assert s.uncrewed == (1,)
    assert s.bound / crewreplay.UNCREWED == pytest.approx(1, abs=0.05)


def test_a_deadhead_needs_free_seats_and_costs_its_pay_and_seats():
    a220 = [unit(k, kind, report=400, trip=(0,), end="BOS") for k, kind in (("P0", "pilot"), ("C0", "cabin"))]
    a220 = [replace(u, group="A220", size=crews.size(u.kind, "A220")) for u in a220]
    riders = [unit("P1", report=400, end="DCA"), unit("C1", "cabin", report=400, end="DCA")]
    tails = [replace(tail("N1", "JFK"), family="A220", seats=140), tail("N2", "BOS")]
    for pax, uncrewed, both in ((134, (), True), (135, (1,), False)):
        task = day([replace(flight(0, "JFK", "BOS", 500, 30, pax=pax), family="A220"),
                    flight(1, "BOS", "DCA", 700, 60)], tails)
        p, s = solve(task, a220 + riders)
        rode = [u for u in ("P1", "C1") if (0, DH) in s.columns[u].first]
        assert s.uncrewed == uncrewed and len(rode) == (2 if both else 1)
        assert not both or all(s.columns[u].first == ((0, DH), (1, OP)) for u in rode)
    p, s = solve(day([replace(flight(0, "JFK", "BOS", 500, 30, pax=134), family="A220"),
                      flight(1, "BOS", "DCA", 700, 60)], tails), a220 + riders)
    assert s.crew_cost == pytest.approx(crews.deadhead("pilot", "A32X", 30, "2024-01-01")
                                        + crews.deadhead("cabin", "A32X", 30, "2024-01-01"))


def test_table_a_counts_actual_blocks_before_each_takeoff():
    flights = [flight(0, "JFK", "BOS", 600, 60, sched_block=30), flight(1, "BOS", "JFK", 700, 30)]
    task = day(flights, [tail("N1", "JFK")])
    pilots = [unit("P1", minutes=470, flown=2, trip=(0, 1)), unit("C1", "cabin", trip=(0, 1))]
    p, s = solve(task, pilots)
    assert s.uncrewed == (1,) and 470 + 60 + 30 > crews.flight_time(9) >= 470 + 30
    p, s = solve(task, pilots, blocks={0: 30, 1: 30})
    assert s.uncrewed == ()


def test_short_call_reports_on_the_last_whole_hour_before_its_departure():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    units = [unit("R", report=None, hour=None, reserve="short", rap=0, until=720, lead=150),
             unit("C1", "cabin", trip=(0,))]
    p, s = solve(task, units)
    assert s.uncrewed == () and s.columns["R"].report == 540


def test_rest_move_flies_next_morning_with_a_hotel():
    task = day([flight(0, "BOS", "JFK", 1665, 60)], [tail("N1", "BOS")])
    units = [unit(k, kind, "BOS", 400, ready=1020, landed=1000, trip=(0,), end="JFK")
             for k, kind in (("P1", "pilot"), ("C1", "cabin"))]
    p, s = solve(task, units)
    assert s.uncrewed == () and s.columns["P1"].rest == ("BOS", 1620) and s.columns["C1"].rest is None
    assert s.crew_cost == pytest.approx(2 * crews.hotel("BOS", task.date))
    early = day([flight(0, "BOS", "JFK", 1655, 60, deadline=1660)], [tail("N1", "BOS")])
    assert solve(early, units)[1].uncrewed == (0,)


def test_a_crew_takes_the_start_it_prefers_unless_only_the_other_crews_a_leg():
    task = day([flight(0, "BOS", "JFK", 600, 60), flight(1, "DCA", "JFK", 620, 70)],
               [tail("N1", "BOS"), tail("N2", "DCA")])
    units = [unit("P1", station="DCA", trip=(1,), end="JFK"), unit("C1", "cabin", station="DCA", trip=(1,), end="JFK")]
    starts = {u.id: (("BOS", 0.0), ("DCA", 5000.0)) for u in units}
    p, s = problem(task, units, starts=starts), None
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.columns["P1"].first == ((0, OP),) and s.columns["P1"].start == "BOS" and s.uncrewed == (1,)
    assert [c.end for c in p.idles(units[0])] == ["BOS", "DCA"]
    only = day([flight(1, "DCA", "JFK", 620, 70)], [tail("N2", "DCA")])
    p = problem(only, units, starts=starts)
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.uncrewed == () and s.columns["P1"].start == "DCA"
    assert s.crew_cost == pytest.approx(2 * 5000.0)


def test_a_ride_a_full_rest_after_the_release_starts_the_second_period_with_a_hotel():
    task = day([flight(0, "DCA", "JFK", 1300, 45, pax=100)], [tail("N1", "DCA")])
    assert 45 < crews.ferry_block(task.airports["DCA"], task.airports["JFK"])
    units = [unit("P1", station="DCA", report=400, ready=605, landed=585, flown=2, end="JFK"),
             unit("P2", station="DCA", report=1200, trip=(0,), end="JFK"),
             unit("C2", "cabin", station="DCA", report=1200, trip=(0,), end="JFK")]
    late = crewreplay.Column("P1", ((0, DH),), end="JFK")
    p = problem(task, units)
    assert p.sat_out(units[0], late)
    assert not p.sat_out(units[0], crewreplay.Column("P1", second=((0, DH),), rest=("DCA", 1200), end="JFK"))
    s = crewreplay.solve(p, time_limit=30, polish=5)
    c = s.columns["P1"]
    assert c.first == () and c.second == ((0, DH),) and c.rest == ("DCA", 1200)
    assert s.crew_cost == pytest.approx(crews.hotel("DCA", task.date) * 2 + crews.deadhead("pilot", "A32X", 45,
                                                                                            task.date))
    d = crewsim.execute(task, crewday.plan_of(p, s), units, fill=False)
    assert d.crew_cost == pytest.approx(s.crew_cost) and d.units["P1"]["hotels"] == [
        {"station": "DCA", "from": 600, "to": 1200}]


STATIONS = ("JFK", "BOS", "DCA", "MCO", "BUF")


def random_day(seed: int):
    rng = random.Random(seed)
    flights = []
    for i in range(rng.randint(6, 12)):
        a, b = rng.sample(STATIONS, 2)
        flights.append(flight(i, a, b, rng.randrange(300, 1400, 5), rng.randrange(50, 200, 5),
                              pax=rng.randint(130, 150), deadline=2000))
    tails = [tail(f"N{f.id}", f.orig) for f in flights]
    task = day(flights, tails)
    by = sorted(flights, key=lambda f: f.sched_dep)
    units = []
    for n in range(rng.randint(2, 4)):
        trip = tuple(f.id for f in rng.sample(by, rng.randint(1, 3)))
        first = task.flight(min(trip, key=lambda i: task.flight(i).sched_dep))
        rep = first.sched_dep - crews.REPORT
        for kind in crews.KINDS:
            units.append(unit(f"{kind[0]}{n}", kind, first.orig, rep, trip=trip, end=rng.choice(STATIONS)))
    for kind in crews.KINDS:
        units.append(unit(f"S{kind}", kind, "JFK", 300, reserve="standby", until=660))
        units.append(unit(f"R{kind}", kind, "JFK", None, hour=None, reserve="short", rap=rng.choice((120, 780)),
                          until=900, lead=150))
    return task, units


@pytest.mark.parametrize("seed", range(12))
def test_every_plan_obeys_the_sim_and_replays_there_at_its_cost(seed):
    task, units = random_day(seed)
    p, s = solve(task, units)
    assert s.bound <= s.objective + 1e-6 and s.lp <= s.objective + 1e-6
    deps = {f.id: f.sched_dep for f in task.flights}
    plan = crewday.plan_of(p, s)
    assert crewsim.audit(task, units, plan, deps, {f.id: f.block for f in task.flights}) == []
    held = {i: {**plan.get(i, {}), "not_before": d} for i, d in deps.items()}
    held["reports"] = plan["reports"]
    for i in s.uncrewed:
        held[i] = {"cancel": True}
    d = crewsim.execute(task, held, units, fill=False)
    assert all(d.run.flights[i].dep == deps[i] for i in deps if i not in s.uncrewed)
    assert d.crews == {i: v for i, v in s.crews(p).items() if i not in s.uncrewed}
    assert d.crew_cost == pytest.approx(s.crew_cost, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_a_unit_alone_finds_a_legal_day_through_any_leg_the_plan_crews(seed):
    task, units = random_day(seed)
    p, s = solve(task, units)
    pricer = crewreplay.Pricer(p)
    for i, kinds in s.crews(p).items():
        for kind, uid in kinds.items():
            assert uid in crewday.alone(p, pricer, kind, p.leg[i])
