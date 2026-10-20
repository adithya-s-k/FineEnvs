"""The scripted two desks by hand: a decision reaches the crew desk at its minute and changes nothing before it, the
future can't leak into a watch, the channel acts only through messages and can cancel as the desk does, the roles
hold, and a round trip is cut only where that lowers the projected cost and never on a committed or Move-bound leg."""

from dataclasses import replace

from day_helpers import day, flight, tail

from airops_core.core.known import FREEZE, LOOK, STEP
from airops_core.core.sim import AUTO_CANCELLED, CANCELLED
from airops_core.programs import crewday, crews, crewsim, occ, pairings, sim
from airops_core.programs.crews import Unit
from airops_core.programs.model import Move


def unit(uid: str, kind: str, station: str, report: int, trip: tuple = ()) -> Unit:
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), "JFK", station, station, report + crews.BRIEF, report,
                report // 60 % 24, at=station, trip=trip)


def crew(n: int, trip: tuple, station: str = "JFK", report: int = 540) -> list[Unit]:
    return [unit(f"P{n}", "pilot", station, report, trip), unit(f"C{n}", "cabin", station, report, trip)]


def case(task, units, trips=()) -> occ.Case:
    return occ.Case(task, tuple(units), lambda s, m: m, task.airports, tuple(trips), sim.truth(task))


def storm(lb: int = 600, later: tuple[int, int] = (0, 0)) -> occ.Case:
    """JFK-BOS 0 and BOS-JFK 1, a round trip on one aircraft, 0 held `lb` minutes; PWM-DCA 2, its crew stranded at
    BUF; JFK-MCO 3, its crew reporting late; `later` (lb, extra block) on every flight from 300 on."""
    extra, more = later
    flights = [flight(0, "JFK", "BOS", 300, 60, lb=lb, deadline=1500),
               flight(1, "BOS", "JFK", 420, 60 + more, lb=extra, deadline=1500, sched_block=60),
               flight(2, "PWM", "DCA", 600, 60 + more, lb=extra, deadline=840, sched_block=60),
               flight(3, "JFK", "MCO", 500, 150 + more, lb=extra, sched_block=150)]
    task = day(flights, [tail("N1", "JFK", legs=(0, 1)), tail("N2", "PWM", legs=(2,)), tail("N3", "JFK", legs=(3,))])
    return case(task, crew(1, (0, 1), report=840) + crew(2, (2,), "BUF") + crew(3, (3,), report=600), ((0, 1),))


def test_a_cancellation_reaches_the_crew_desk_at_its_minute_and_changes_nothing_before_it():
    flights = [flight(0, "BUF", "JFK", 400, 60), flight(1, "JFK", "BOS", 200, 60, deadline=440),
               flight(2, "BUF", "JFK", 220, 60), flight(3, "JFK", "PWM", 330, 60)]
    task = day(flights, [tail("N1", "BUF"), tail("N2", "JFK"), tail("N3", "BUF")])
    units = crew(1, (0,), "BUF", 150) + crew(2, (1,), "DCA") + crew(3, (2, 3), "BUF", 150)
    run = lambda plan: crewsim.execute(task, plan, units, desk=True, drop=False)  # noqa: E731
    at_start = run({0: {"cancel": True, "decided": 0}})
    assert at_start.crews[1] == {"pilot": "P1", "cabin": "C1"} and at_start.riders[2] == ("C1", "P1")
    later = run({0: {"cancel": True, "decided": 300}})
    assert 1 not in later.crews and later.run.flights[1].status == AUTO_CANCELLED
    assert later.status[0][-1] == (300, "cancelled", None) and later.run.flights[0].decided == 300
    assert occ.past(later, 300) == occ.past(run({}), 300) != occ.past(at_start, 300)


def test_with_its_crews_named_a_day_replays_alike_whenever_its_cancellations_reach_the_crews():
    c = storm()
    r = occ.run(c, "quiet-flags")
    assert any(e["decided"] > c.task.start for e in r.plan.values())
    named = crewday.named(r.day)
    replay = crewsim.execute(c.task, named, c.units, fill=False)
    assert replay.run.rows() == r.day.run.rows() and replay.crews == r.day.crews
    early = {i: {**e, "decided": 0} if e.get("cancel") else e for i, e in named.items()}
    again = crewsim.execute(c.task, early, c.units, fill=False)
    assert again.crews == replay.crews and [x.dep for x in again.run.flights.values()] == [
        x.dep for x in replay.run.flights.values()]
    base = occ.run(c, "baseline").day
    desk = crewsim.execute(c.task, {}, c.units, desk=True)
    assert occ.same(base, desk) and base.units == desk.units


def test_the_watches_run_every_hour_from_the_start_while_a_decision_could_take_effect():
    task = storm().task
    assert occ.watches(task) == list(range(0, 1500 - FREEZE + 1, STEP))


def test_the_round_trips_pair_consecutive_out_and_backs_of_the_planned_rotations_from_any_station():
    def leg(key, orig, dest, sd, after=None):
        return pairings.Leg(key, f"B6 {key}", "A32X", orig, dest, sd, sd + 60, 60, "A320", 6, after)

    legs = [leg(5, "BOS", "JFK", -200, 10), leg(10, "JFK", "BOS", 300, 11), leg(11, "BOS", "JFK", 420, 12),
            leg(12, "JFK", "PWM", 520, 13), leg(13, "PWM", "JFK", 640), leg(20, "FLL", "MCO", 400, 21),
            leg(21, "MCO", "FLL", 520, 22), leg(22, "FLL", "SJU", 700)]
    flight_of = {10: 0, 11: 1, 12: 2, 13: 3, 20: 4, 21: 5, 22: 6}
    assert occ.round_trips(legs, flight_of) == ((0, 1), (2, 3), (4, 5))


def test_a_cut_lowers_the_projected_cost_and_never_takes_a_committed_or_move_bound_leg():
    c = storm()
    r = occ.Run("quiet", c.day({}, False))
    t = 300 - LOOK
    view = occ.view(c.task, {}, r.day, t, {})
    assert occ.cut(c, r, t - STEP, set(), {}) == []
    assert occ.cut(c, r, t, set(), {}) == [(0, 1)]
    assert c.project(view | occ.cancels((0, 1), t), t).cost.total < c.project(view, t).cost.total
    assert occ.cut(c, r, t, {0}, {}) == [] and occ.cut(c, r, t, {1}, {}) == []
    moved = replace(c.task, tails=(replace(c.task.tails[0], legs=(0,), repositions=(Move(0, "BOS", "JFK", 360, 400),)),
                                   *c.task.tails[1:]))
    assert occ.cut(replace(c, task=moved, known={}), r, t, set(), {}) == []
    assert occ.cut(storm(lb=30), occ.Run("quiet", storm(lb=30).day({}, False)), t, set(), {}) == []
    q = occ.run(c, "quiet")
    assert {i: (e["decided"], q.causes[i]) for i, e in q.plan.items() if i in (0, 1)} == {0: (t, occ.CUT),
                                                                                         1: (t, occ.CUT)}


def test_talking_cancels_what_the_crew_desk_cannot_crew_and_reads_its_holds():
    c = storm()
    talk, quiet, flags = (occ.run(c, m) for m in ("talk", "quiet", "quiet-flags"))
    assert (talk.plan[2], talk.causes[2]) == ({"cancel": True, "decided": 0}, occ.REQUEST)
    asked = {m["params"]["flight"]: m for m in talk.messages()}
    assert [v for k, v in talk.replies.items() if k[1:3] == (crewsim.CANCEL, 2)] == [
        {"reply": "accept", "replied": 0, "actions": [{"cancel": 2, "at": 0}]}] and 2 not in asked
    assert (asked[3]["type"], asked[3]["params"]["minute"], asked[3]["reply"]) == (crewsim.HOLD, 645, "accept")
    assert quiet.day.run.flights[2].status == AUTO_CANCELLED and 2 not in quiet.plan
    assert (flags.plan[2]["decided"], flags.causes[2]) == (600, occ.FLAG)
    assert talk.day.total < flags.day.total < quiet.day.total


def test_no_decision_changes_anything_before_its_watch():
    c = storm()
    for name in occ.MODES:
        r = occ.run(c, name)
        assert r.acausal == []
        for t in sorted({e["decided"] for e in r.plan.values()}):
            before = {i: e for i, e in r.plan.items() if e["decided"] < t}
            assert occ.past(c.day(before, occ.MODES[name].drop), t) == occ.past(r.day, t)


def test_the_truth_after_the_look_ahead_reaches_no_decision_or_message():
    t = 300 - LOOK
    for name in ("talk", "quiet-flags", "one"):
        a, b = (occ.run(storm(later=later), name) for later in ((0, 0), (90, 25)))
        seen = lambda r: ({i: e for i, e in r.plan.items() if e["decided"] <= t},  # noqa: E731
                          [m for m in r.messages() if m["sent"] <= t])
        assert seen(a) == seen(b) and seen(a)[0]
        assert a.day.run.rows() != b.day.run.rows()


def test_talking_with_ops_ignoring_every_message_is_the_quiet_day():
    c = storm()
    deaf = occ.run(c, "talk", replace(occ.MODES["talk"], reads=()))
    quiet = occ.run(c, "quiet")
    assert occ.same(deaf.day, quiet.day) and deaf.plan == quiet.plan and deaf.messages() == quiet.messages()
    assert not occ.same(occ.run(c, "talk").day, quiet.day)


def test_idle_ops_accepting_each_cancel_at_once_is_the_baseline():
    c = storm()
    once, base = occ.at_once(c), occ.run(c, "baseline")
    assert base.day.dropped and once.plan == {i: {"cancel": True, "decided": s} for i, s in base.day.dropped.items()}
    assert occ.same(once.day, base.day)


def test_ops_only_cancels_at_its_watches_and_the_crew_desk_only_asks():
    c = storm()
    for name, mode in occ.MODES.items():
        r = occ.run(c, name)
        assert occ.roles(r, mode, c.task) == []
        own = [i for i, x in r.day.run.flights.items() if x.status == CANCELLED and i not in r.plan]
        assert (own != []) == mode.drop and (r.day.dropped != {}) == mode.drop
    r = occ.run(c, "talk")
    r.plan[3] = {"cancel": True, "decided": 61, "not_before": 900}
    assert occ.roles(r, occ.MODES["talk"], c.task) == ["ops plan for 3: ['cancel', 'decided', 'not_before']",
                                                       "ops cancelled 3 at 61, not a watch"]
    assert occ.roles(occ.run(c, "baseline"), occ.MODES["quiet"], c.task) == ["crew desk dropped 2",
                                                                             "crew desk cancelled 2"]


def test_ops_projects_the_history_from_the_crew_day_and_the_crew_desks_own_cancellations():
    c = storm()
    base = occ.run(c, "baseline").day
    view = occ.view(c.task, {}, base, 600, {3: 800, 0: 500})
    assert view[2] == {"cancel": True, "decided": 0} and view[3] == {"not_before": 800}
    assert view[0] == {"not_before": 600} and base.run.flights[3].dep < 700
    assert occ.view(c.task, {}, base, 700, {})[3] == {"left": base.run.flights[3].dep}
    assert occ.opened(base, {}, 0, 600) and not occ.opened(base, {}, 2, 600) and not occ.opened(base, {}, 3, 700)
