"""Units at the start from a crew replay of the record before it with the rest of the day as scheduled, by hand: a
unit follows its column up to the start (actual blocks, its report if a reserve was called) and takes the column's
legs from the start as its trip; the replay looks ahead, so a crew too near its limits is not sent where the day goes
on; a crew that a cancellation before the start would strand is replaced; crews released before the start with
nothing left, and reserves whose period is over, are off; a crew laid over where the record has no aircraft may start
where the record has it instead."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
from day_helpers import day, flight, tail

from airops_core.core.model import Airport
from airops_core.programs import crewday, crewreplay, crews, crewsim, network, pairings
from airops_core.programs.crewday import ahead
from airops_core.programs.crewreplay import DH, OP, Column, Leg, Problem
from airops_core.programs.crews import Unit

AIRPORTS = {c: Airport(c, c, lat, lon, "America/New_York") for c, (lat, lon) in
            {"JFK": (40.64, -73.78), "BOS": (42.36, -71.01), "DCA": (38.85, -77.04),
             "BUF": (42.94, -78.73), "SYR": (43.11, -76.11)}.items()}
DATE = "2024-01-01"


def problem(legs, units):
    return Problem(legs, units, DATE, lambda s, m: m, lambda u, s: crews.reposition(u, s, AIRPORTS, DATE))


def unit(uid, kind="pilot", station="JFK", report=240, **kw) -> Unit:
    hour = kw.pop("hour", None if report is None else report // 60)
    ready = kw.pop("ready", 0 if report is None else report + crews.BRIEF)
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), kw.pop("home", "JFK"), kw.pop("end", station), station,
                ready, report, hour, at=station, **kw)


def later(fid: int, orig: str, dest: str, dep: int, block: int) -> Leg:
    return Leg(ahead(fid), f"B6 {fid}", orig, dest, dep, dep + block, block, "A32X", 20, price=crewday.AHEAD)


def test_a_unit_follows_its_column_and_takes_its_trip_from_the_plan():
    legs = [Leg(101, "B6 1", "JFK", "BOS", 300, 380, 60, "A32X", 20), later(7, "BOS", "DCA", 700, 60),
            later(8, "DCA", "JFK", 800, 60)]
    u = unit("P", trip=(101, ahead(7)))
    p = problem(legs, [u])
    c = Column("P", ((101, OP), (ahead(7), OP), (ahead(8), DH)), end="JFK")
    s = crewday.at_start(p, u, c, 600)
    assert (s.station, s.ready, s.flown, s.minutes, s.landed) == ("BOS", 400, 1, 80, 380)
    assert (s.trip, s.deadheads, s.report) == ((7,), (8,), 240)


def test_a_reserve_reports_when_the_replay_calls_it_before_or_after_the_start_and_flies_its_legs():
    legs = [Leg(101, "B6 1", "JFK", "BOS", 345, 405, 60, "A32X", 20), later(7, "JFK", "DCA", 705, 60)]
    r = unit("R", report=None, hour=None, reserve="short", rap=120, until=840, lead=150)
    p = problem(legs, [r])
    s = crewday.at_start(p, r, Column("R", ((101, OP),), report=300, end="BOS"), 600)
    assert (s.report, s.hour, s.rap, s.station, s.flown) == (300, 5, 120, "BOS", 1)
    s = crewday.at_start(p, r, Column("R", ((ahead(7), OP),), report=660, end="DCA"), 600)
    assert (s.report, s.hour, s.ready, s.station, s.trip) == (660, 11, 660 + crews.BRIEF, "JFK", (7,))
    assert crewday.at_start(p, r, None, 600).report is None
    task = day([flight(7, "JFK", "DCA", 705, 60)], [tail("N1", "JFK")], start=600)
    called = [replace(s, id=k, kind=kind, size=crews.size(kind, "A32X")) for k, kind in (("RP", "pilot"),
                                                                                         ("RC", "cabin"))]
    d = crewsim.execute(task, {}, called)
    assert d.run.flights[7].dep == 705 and d.crews[7] == {"pilot": "RP", "cabin": "RC"}


def test_at_four_a_red_eye_crew_carries_its_duty_so_far_and_the_others_start_rested_where_their_trips_report():
    legs = [Leg(101, "B6 1", "BUF", "JFK", 90, 300, 150, "A32X", 20), later(7, "JFK", "BOS", 420, 60),
            later(8, "BOS", "DCA", 405, 90)]
    eye = [unit(k, kind, "BUF", -30, hour=23, trip=(101, ahead(7)), end="BOS") for k, kind in
           (("PR", "pilot"), ("CR", "cabin"))]
    rested = [unit(k, kind, "BOS", 345, trip=(ahead(8),), end="DCA") for k, kind in (("PM", "pilot"), ("CM", "cabin"))]
    p = problem(legs, eye + rested)
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.uncrewed == () and s.crews(p)[101] == {"pilot": "PR", "cabin": "CR"}
    at = {u.id: crewday.at_start(p, u, s.columns.get(u.id), network.START) for u in p.units}
    r = at["PR"]
    assert (r.station, r.ready, r.report, r.flown, r.minutes, r.landed, r.trip) == (
        "JFK", 300 + crews.CONNECT, -30, 1, 210, 300, (7,))
    m = at["PM"]
    assert (m.station, m.ready, m.report, m.flown, m.minutes, m.landed, m.trip) == (
        "BOS", 345 + crews.BRIEF, 345, 0, 0, None, (8,))


def test_crews_done_before_the_start_and_spent_reserves_are_off():
    legs = [Leg(101, "B6 1", "JFK", "BOS", 300, 380, 60, "A32X", 20)]
    done = unit("P", trip=(101,))
    p = problem(legs, [done])
    assert crewday.at_start(p, done, Column("P", ((101, OP),), end="BOS"), 600) is None
    assert crewday.at_start(p, done, Column("P", ((101, OP),), end="BOS"), 390) is not None
    assert crewday.at_start(p, done, None, 600) is not None
    over = unit("R", report=None, hour=None, reserve="short", rap=0, until=500, lead=150)
    assert crewday.at_start(p, over, None, 600) is None
    standby = unit("S", report=300, reserve="standby", until=660)
    assert crewday.at_start(p, standby, None, 600).until == 660
    moved = crewday.at_start(p, standby, Column("S", ((101, DH),), end="BOS"), 600)
    assert moved.until is None and moved.station == "BOS" and moved.flown == 0


def test_the_replay_looks_ahead_and_sends_the_crew_that_can_go_on():
    legs = [Leg(101, "B6 1", "JFK", "BUF", 650, 740, 90, "A32X", 0), later(1, "BUF", "JFK", 800, 90)]
    tired = [unit(k, kind, report=-30, hour=20, flown=3, minutes=400, trip=(101,)) for k, kind in
             (("PA", "pilot"), ("CA", "cabin"))]
    fresh = [unit(k, kind, report=605, trip=(ahead(1),)) for k, kind in (("PB", "pilot"), ("CB", "cabin"))]
    p = problem(legs, tired + fresh)
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.uncrewed == () and s.crews(p)[101]["pilot"] == "PB"
    b = crewday.at_start(p, p.unit["PB"], s.columns["PB"], 760)
    assert (b.station, b.trip) == ("BUF", (1,))


def test_a_leg_cancelled_before_the_start_does_not_strand_the_morning():
    legs = [Leg(101, "B6 1", "JFK", "BOS", 300, 360, 60, "A32X", 20), later(7, "BOS", "JFK", 700, 60)]
    stranded = [unit(k, kind, "DCA", 200, trip=(100, 101, ahead(7)), end="JFK") for k, kind in
                (("P", "pilot"), ("C", "cabin"))]
    standby = [unit(k, kind, "JFK", 240, reserve="standby", until=600) for k, kind in
               (("SP", "pilot"), ("SC", "cabin"))]
    p = problem(legs, stranded + standby)
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.uncrewed == () and s.crews(p)[101] == {"pilot": "SP", "cabin": "SC"}
    at = {u.id: crewday.at_start(p, u, s.columns.get(u.id), 600) for u in p.units}
    assert (at["SP"].station, at["SP"].trip, at["P"].station, at["P"].trip) == ("BOS", (7,), "DCA", ())


def test_a_flight_left_without_a_crew_names_the_crews_there_and_their_rule():
    legs = [Leg(1, "B6 1", "BOS", "JFK", 1131, 1131 + 255, 255, "A32X", 20)]
    tired = [unit(k, kind, "BOS", 540, hour=9, flown=1, minutes=295, landed=1084, ready=1104, end="BOS")
             for k, kind in (("P", "pilot"), ("C", "cabin"))]
    p = problem(legs, tired)
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.uncrewed == (1,)
    cause = crewday.uncrewed_causes(p, s)[0]
    assert cause["missing"] == {"pilot": "no unit's plan has it; P Table A: 295 + 255 > 540",
                                "cabin": "no unit's plan has it; C 14 h duty from report 540"}
    assert cause["could_alone"] == {"pilot": [], "cabin": []}


def sched(key, orig, dest, sd, block, after=None) -> pairings.Leg:
    return pairings.Leg(key, f"B6 {key}", "A32X", orig, dest, sd, sd + block, block, "A320", 0, after)


def world_of(legs, rows, duties) -> crewday.World:
    built = {("pilot", "A32X"): pairings.Built("pilot", "A32X", duties, [], 0.0, 0.0, 0, 0, True, 0.0)}
    day = SimpleNamespace(prev=pd.DataFrame(rows), cur=pd.DataFrame(columns=list(rows[0])),
                          fleet={"N1": 1, "N2": 1, "N3": 1})
    ctx = SimpleNamespace(day=day, families={"N1": "A320", "N2": "A320", "N3": "A320"})
    return crewday.World(SimpleNamespace(date=DATE), ctx, built, AIRPORTS, legs)


def test_a_crew_laid_over_where_the_record_has_no_aircraft_may_start_where_it_has_one():
    legs = [sched(1, "JFK", "SYR", 1200 - 1440, 80, after=2), sched(2, "SYR", "JFK", 360, 75),
            sched(3, "BOS", "DCA", 400, 90)]
    rows = [{"Cancelled": 0, "sd": 0 - 1440, "ad": 20 - 1440, "Tail_Number": "N1", "end": "BOS", "Origin": "JFK"},
            {"Cancelled": 1, "sd": 1200 - 1440, "ad": None, "Tail_Number": "N1", "end": "SYR", "Origin": "JFK"},
            {"Cancelled": 0, "sd": 400, "ad": 420, "Tail_Number": "N1", "end": "DCA", "Origin": "BOS"},
            {"Cancelled": 0, "sd": 360, "ad": 380, "Tail_Number": "N2", "end": "JFK", "Origin": "SYR"},
            {"Cancelled": 0, "sd": -900, "ad": -890, "Tail_Number": "N3", "end": "BUF", "Origin": "JFK"}]
    duty = pairings.Duty("pilot", "A32X", (2,), (), "SYR", "JFK", 300, 450, 5, 0.0)
    w = world_of(legs, rows, [duty])
    assert crewday.NIGHT <= 300
    assert crewday.rotations_at(legs, crewday.NIGHT) == {("A32X", "SYR"): 1, ("A32X", "BOS"): 1}
    assert crewday.aircraft_at(w, crewday.NIGHT) == {("A32X", "BOS"): 1, ("A32X", "SYR"): 1}
    laid = unit("P", station="SYR", report=300, trip=(2,))
    assert crewday.layovers(w, [laid]) == {}
    rows[3]["Origin"] = rows[3]["end"] = "BOS"
    assert crewday.layovers(world_of(legs, rows, [duty]), [laid]) == {"P": (("BOS", 0.0), ("SYR", crewday.LAYOVER))}


def test_a_crew_starting_elsewhere_stands_there_at_the_start_on_that_station_s_clock():
    legs = [Leg(101, "B6 1", "BOS", "DCA", 400, 490, 90, "A32X", 20, need=("pilot",))]
    u = unit("P", station="SYR", report=300, trip=(102,))
    p = Problem(legs, [u], DATE, lambda s, m: m + (60 if s == "BOS" else 0),
                lambda u, s: crews.reposition(u, s, AIRPORTS, DATE), starts={"P": (("BOS", 0.0), ("SYR", 1e5))})
    s = crewreplay.solve(p, time_limit=30, polish=5)
    assert s.columns["P"].start == "BOS" and s.columns["P"].first == ((101, OP),)
    b = crewday.at_start(p, u, Column("P", start="BOS", end="BOS"), 350)
    assert (b.station, b.at, b.hour, b.report) == ("BOS", "BOS", 6, 300)
