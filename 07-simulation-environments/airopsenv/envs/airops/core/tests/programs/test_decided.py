"""Decisions taken during the day reach the sim at their minute: a hold, a claim, a crew named or let go, a rider and a
reserve's call change nothing before it and act from it on; a unit the crew desk moves, and the crew it replaces, keep
the legs of their trips that still connect, the rest going back to the standard desk at once (and back to them when the
move ends), which leaves a named crew alone and doesn't take a unit while a flight it is named on is to come; a swap
exchanges two trips; and a day carried on in another world from a minute keeps everything before it."""

from dataclasses import replace

from day_helpers import GDP, day, flight, tail, three_into_a_program

from airops_core.programs import crewday, crews, crewsim, occ, sim
from airops_core.programs.crews import Unit


def unit(uid: str, kind: str, station: str = "JFK", report: int | None = 540, trip: tuple = (), **kw) -> Unit:
    hour = report // 60 % 24 if report is not None else None
    ready = kw.pop("ready", report + crews.BRIEF if report is not None else 0)
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), "JFK", station, station, ready, report, hour,
                at=station, trip=tuple(trip), **kw)


def crew(n: int, trip: tuple = (), station: str = "JFK", report: int | None = 540, **kw) -> list[Unit]:
    return [unit(f"P{n}", "pilot", station, report, trip, **kw), unit(f"C{n}", "cabin", station, report, trip, **kw)]


def named(t: int, n: int, to: int | None = None) -> list[list]:
    return [[t, to, "pilot", f"P{n}"], [t, to, "cabin", f"C{n}"]]


def before(d: crewsim.CrewDay, t: int) -> tuple:
    return occ.past(d, t), {i: c for i, c in d.crews.items() if d.run.flights[i].dep < t}


def test_a_hold_decided_later_acts_from_its_minute_only():
    task = day([flight(0, "JFK", "BOS", 300, 60), flight(1, "BUF", "DCA", 250, 60)],
               [tail("N1", "JFK", ready=400), tail("N2", "BUF")])
    free = sim.execute(task, {})
    assert free.flights[0].dep == 400
    held = sim.execute(task, {0: {"holds": [[350, 500]]}})
    assert held.flights[0].dep == 500 and held.flights[1] == free.flights[1]
    assert sim.execute(task, {0: {"holds": [[350, 500], [420, None]]}}).flights[0].dep == 420
    assert sim.execute(task, {0: {"holds": [[450, 600]]}}).flights[0].dep == 400
    assert sim.execute(task, {0: {"holds": [[0, 500]]}}).rows() == sim.execute(task, {0: {"not_before": 500}}).rows()


def test_a_claim_decided_later_binds_a_slot_nobody_took_and_moves_nothing_before_it():
    task = replace(three_into_a_program(), programs={
        h: replace(p, segments=tuple(replace(s, slots=(240, 300, 360)) for s in p.segments))
        for h, p in three_into_a_program().programs.items()})
    free = sim.execute(task, {})
    assert [free.flights[i].dep for i in range(3)] == [180, 240, 300]
    late = sim.execute(task, {2: {"claims": [[200, 240]]}})
    assert late.flights[0] == free.flights[0] and (late.flights[2].dep, late.flights[2].slot) == (240, 300)
    assert (late.flights[1].dep, late.flights[1].slot) == (300, 360) and late.wasted == ()
    assert sim.execute(task, {2: {"claims": [[0, 360]]}}).rows() == sim.execute(task, {2: {"claim": 360}}).rows()
    assert GDP.id == "g"


def test_a_crew_named_later_is_the_plans_only_from_its_minute():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 500, 60)],
               [tail("N1", "JFK"), tail("N2", "JFK")])
    units = crew(1, (0,)) + crew(2)
    run = lambda plan: crewsim.execute(task, plan, units, desk=True, drop=False)  # noqa: E731
    free = run({})
    assert free.crews == {1: {"pilot": "P2", "cabin": "C2"}, 0: {"pilot": "P1", "cabin": "C1"}}
    early = run({0: {"crew": {"pilot": "P2", "cabin": "C2"}}})
    assert 1 not in early.crews and early.crews[0] == {"pilot": "P2", "cabin": "C2"}
    late = run({0: {"names": named(550, 2)}})
    assert before(late, 550) == before(free, 550)
    assert late.crews == {0: {"pilot": "P2", "cabin": "C2"}, 1: {"pilot": "P1", "cabin": "C1"}}
    back = run({0: {"names": named(550, 2, to=580)}})
    assert back.crews == free.crews and back.messages == free.messages


def test_a_unit_the_crew_desk_moves_keeps_the_legs_that_still_connect_and_hands_back_the_rest():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(2, "BOS", "JFK", 760, 60), flight(3, "JFK", "DCA", 610, 60)],
               [tail("N1", "JFK"), tail("N2", "JFK")])
    units = crew(1, (0, 2)) + crew(2, (), "JFK", reserve="standby", until=900) + crew(3, (3,))
    run = lambda plan: crewsim.execute(task, plan, units, desk=True, drop=False)  # noqa: E731
    free = run({})
    assert free.crews[2] == {"pilot": "P1", "cabin": "C1"}
    assert run({0: {"names": named(550, 1)}}).run.rows() == free.run.rows()
    away = run({3: {"names": named(550, 1)}})
    assert away.crews == {0: {"pilot": "P2", "cabin": "C2"}, 2: {"pilot": "P3", "cabin": "C3"},
                          3: {"pilot": "P1", "cabin": "C1"}} and away.messages == ()
    took = run({0: {"names": named(550, 2)}})
    assert took.crews[0] == took.crews[2] == {"pilot": "P2", "cabin": "C2"} and took.messages == ()


def test_a_swap_decided_later_exchanges_the_trips_and_leaves_them_to_the_standard_desk():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 620, 60), flight(2, "DCA", "JFK", 760, 60)],
               [tail("N1", "JFK"), tail("N2", "JFK", legs=(1, 2))])
    units = crew(1, (0,)) + crew(2, (1, 2))
    run = lambda plan: crewsim.execute(task, plan, units, desk=True, drop=False)  # noqa: E731
    swapped = run({"swaps": [[550, "P1", "P2"], [550, "C1", "C2"]]})
    assert swapped.crews == {0: {"pilot": "P2", "cabin": "C2"}, 1: {"pilot": "P1", "cabin": "C1"},
                             2: {"pilot": "P1", "cabin": "C1"}}
    assert before(swapped, 550) == before(run({}), 550)


def test_a_rider_and_a_reserve_called_later_act_from_their_minute():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 760, 60)],
               [tail("N1", "JFK"), tail("N2", "BOS")])
    units = crew(1, (0,)) + crew(2, (1,), "BOS", report=700) + [
        unit("RP", "pilot", report=None, ready=0, reserve="short", rap=300, until=900, lead=120)]
    units = [u for u in units if u.id != "P2"]
    plan = {0: {"rides": [[300, None, "RP"]]}, 1: {"names": [[300, None, "pilot", "RP"]]}, "calls": [[300, "RP", 540]]}
    d = crewsim.execute(task, plan, units)
    assert d.riders[0] == ("RP",) and d.crews[1]["pilot"] == "RP"
    assert d.units["RP"]["call"] == 420 and d.units["RP"]["duties"][0]["report"] == 540
    none = crewsim.execute(task, {}, units)
    assert before(d, 300) == before(none, 300) and none.crews[1]["pilot"] == "P1" and "RP" not in none.units


def test_the_standard_desk_leaves_a_named_crew_alone():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    units = crew(1, (0,), "BUF") + crew(2, (), "JFK", reserve="standby", until=900) + [
        unit("RP", "pilot", report=None, ready=0, reserve="short", rap=300, until=900, lead=120),
        unit("RC", "cabin", report=None, ready=0, reserve="short", rap=300, until=900, lead=120)]
    plan = {0: {"crew": {"pilot": "P2", "cabin": "C2"}, "not_before": 600}}
    desk = crewsim.execute(task, plan, units, desk=True)
    assert desk.crews[0] == {"pilot": "P2", "cabin": "C2"} and "RP" not in desk.units and "RC" not in desk.units
    assert desk.run.rows() == crewsim.execute(task, plan, units, fill=False).run.rows()


def test_a_day_carried_on_in_the_known_world_keeps_everything_before_its_minute():
    flights = [flight(0, "JFK", "BOS", 100, 60), flight(1, "BOS", "JFK", 200, 60, lb=40),
               flight(2, "JFK", "DCA", 500, 60, lb=90), flight(3, "DCA", "JFK", 640, 60)]
    task = day(flights, [tail("N1", "JFK", legs=(0, 1, 2, 3))])
    units = crew(1, (0, 1, 2, 3), report=40)
    truth = crewsim.execute(task, {}, units, desk=True, drop=False)
    pool = crewsim.Pool(task, units, lambda s, m: m, {}, True, True, task.airports, desk=True, drop=False)
    d = sim.Dispatch(task, {}, sim.truth(task), pool)
    d.until(300)
    assert {i: r for i, r in d.runs.items()} == {i: truth.run.flights[i] for i in (0, 1)}
    d.switch(sim.known(task, 300), 300)
    run = d.execution()
    assert [run.flights[i].dep for i in range(4)] == [100, 240, 500, 640]
    assert truth.run.flights[2].dep == 590 and crewday is not None
