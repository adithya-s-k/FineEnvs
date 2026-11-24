"""The wind week's excuse: each ship is excused what the truth adds to its problems and cost over the week as known at
its last watch before it froze, with the whole plan evaluated on both. The marine optimum played in invented weather."""

from dataclasses import replace

import synthetic
from berth_core import Plan, plan_from_list, plan_to_list

from port_week_core import marine, wind, world
from port_week_core.marine import MarineWeek
from port_week_core.wind import WindWeek

BUSY = marine.pack().get("dock-24B-w07x1-busy-0")
STANDARD = marine.pack().get("dock-36A-w37x1-standard-0")
APRIL = marine.pack().get("dock-36A-w15x1-standard-0")


def week(v3, above25, above30=(), episodes=lambda init: [], plan: Plan | None = None, news=()) -> WindWeek:
    """The marine optimum (or ``plan``) played through a week whose wind blew in the given hours, each run forecasting
    ``episodes(init)``; ``news`` adds (disruption, notice) pairs."""
    task = wind.transplant(v3, synthetic.record(0, "storm", "2000-01-10", above25, above30, episodes))
    task = replace(task, disruptions=task.disruptions + [e for e, _ in news],
                   notices=task.notices + [n for _, n in news])
    plan = plan or plan_from_list(v3.reference["optimal_plan"])
    return world.play(task, lambda w: plan, week=WindWeek)


def froze(w: WindWeek, ship: int) -> int:
    return next(k for k, f in enumerate(w.frozen) if ship in f["ships"])


def problems(rows: list[dict]) -> list[tuple[int, str]]:
    return [(r["ship"], r["problem"]) for r in rows]


def test_an_ignored_warning_is_charged_and_unforecast_wind_is_excused():
    warned = week(BUSY, range(76, 84), episodes=lambda init: [(77, 83, 28)] if init >= 12 else [])
    assert froze(warned, 8) == 4 and warned.watches[3].hour == 54
    assert wind.forecast(wind.known(warned.task, [e["event_id"] for e in warned.log if e["watch"] <= 3]))["windows"]
    line = "berths at hour 79, inside the no-movement window 76-84"
    assert not warned.grade["feasible"] and problems(warned.grade["violations"]) == [(8, line)]
    unwarned = week(BUSY, range(76, 84))
    assert unwarned.grade["feasible"] and problems(unwarned.grade["excused"]) == [(8, line)]
    assert (unwarned.grade["cost"], unwarned.grade["excused_cost"]) == (BUSY.reference["optimal_cost"], 0)


class PerNotice(WindWeek):
    """The live week's excuse, one notice at a time, which the wind week drops."""
    notice_excuse = True
    excuses = world.Week.excuses


def test_a_false_alarm_retracted_after_the_freeze_leaves_no_waiver():
    plan = plan_from_list(BUSY.reference["optimal_plan"]) | {8: (79, 14, 3)}
    alarm = {"above25": (), "episodes": lambda init: [(117, 121, 28)] if 72 <= init <= 102 else [], "plan": plan}
    w = week(BUSY, **alarm)
    assert froze(w, 8) == 4 and [e["windows"][0]["start"] for e in w.task.disruptions
                                 if e["type"] == "forecast" and e["windows"]] == [117, 117]
    assert (w.grade["cost"], w.grade["raw_cost"], w.grade["excused_cost"]) == (306, 306, 0)
    replayed, excused = WindWeek(w.task), []
    while True:
        replayed.confirm(plan_to_list(plan))
        excused.append(replayed.check([])["excused_cost"])
        if replayed.last:
            break
        replayed.open()
        for n in replayed.watches[replayed.watch].notices:
            replayed.notice(n.event_id, n.name, n.text, "trigger")
    assert excused == [0, 0, 0, 0, 24, 24, 0]
    per_notice = world.play(w.task, lambda _: plan, week=PerNotice)
    assert (per_notice.grade["cost"], per_notice.grade["excused_cost"]) == (282, 24)


def test_the_wind_tug_is_excused_when_unforecast_and_charged_when_forecast():
    line = "tugs short at hour 49: your ships need 2, 0 free"
    unforecast = week(BUSY, [49])
    assert problems(unforecast.grade["excused"]) == [(4, line)] and unforecast.grade["feasible"]
    forecast = week(BUSY, [49], episodes=lambda init: [(48, 50, 28)] if init >= 12 else [])
    assert problems(forecast.grade["violations"]) == [(4, line)]


def test_a_warning_at_another_hour_does_not_stand_for_wind_no_forecast_showed():
    w = week(APRIL, [123], episodes=lambda init: [(136, 138, 28)])
    assert not any(17 in f["ships"] for f in w.frozen)
    last = wind.known(w.task, [e["event_id"] for e in w.log])
    assert [p for r in w.evaluate(last, w.plan).ships if r.ship == 17 for p in r.problems] == [
        "tugs short at hour 139: your ships need 6, 5 free"]
    line = "tugs short at hour 123: your ships need 7, 6 free"
    assert problems(w.grade["excused"]) == [
        (17, "berths at hour 123, inside the no-movement window 123-124"), (17, line), (19, line)]
    assert w.grade["feasible"] and w.grade["reward"] == 1.0


def test_ships_frozen_at_different_watches_held_to_the_same_hour_are_excused_the_move_limit():
    w = week(BUSY, range(74, 79), range(74, 79))
    assert (froze(w, 3), froze(w, 16), froze(w, 8)) == (2, 4, 4)
    assert w.grade["feasible"] and problems(w.grade["excused"]) == [
        (3, "leaves at hour 79, when 3 ships move (limit 2 per hour)"),
        (8, "berths at hour 79, when 3 ships move (limit 2 per hour)"),
        (16, "leaves at hour 79, when 3 ships move (limit 2 per hour)")]
    assert w.excuses()[1] == {3: 24, 16: 25}


def test_a_ship_held_into_a_later_frozen_one_is_excused_for_both_unless_the_later_one_was_warned():
    unforecast = week(STANDARD, range(57, 63))
    assert (froze(unforecast, 3), froze(unforecast, 5), froze(unforecast, 6)) == (1, 2, 2)
    overlaps = [(3, "overlaps ship 5 (NIKOLAS) in sections 21-25 during hours 62-63"),
                (3, "overlaps ship 6 (CTM ISTMO) in sections 26-29 during hours 60-63"),
                (5, "overlaps ship 3 (MSC ALLEGRA) in sections 21-25 during hours 62-63"),
                (6, "overlaps ship 3 (MSC ALLEGRA) in sections 26-29 during hours 60-63")]
    held = (7, "berths at hour 58, inside the no-movement window 57-63")
    assert unforecast.grade["feasible"] and problems(unforecast.grade["excused"]) == overlaps + [held]
    warned = week(STANDARD, range(57, 63), episodes=lambda init: [(58, 62, 28)] if init >= 30 else [])
    assert problems(warned.grade["excused"]) == []
    assert problems(warned.grade["violations"]) == overlaps + [held]


def test_an_overlap_is_judged_as_the_ship_decided_last_saw_it():
    """CMA CGM ADONIS froze on runs that held it alongside to hour 96, into ASIAN MOON's berth at 92; ASIAN MOON was
    decided later, on runs that ended the blow at 89. It blew to 95: the overlap is excused for both."""
    w = week(APRIL, (), range(89, 95), lambda init: [(89, 95, 34)] if init <= 36 else [(89, 89, 34)])
    assert (froze(w, 11), froze(w, 14)) == (3, 4)
    for k, seen in ((2, True), (3, False)):
        view = wind.known(w.task, [e["event_id"] for e in w.log if e["watch"] <= k])
        assert any(p.startswith("overlaps ship 14 ") for r in w.evaluate(view, w.plan).ships if r.ship == 11
                   for p in r.problems) == seen
    assert w.grade["feasible"] and w.grade["reward"] == 1.0 and problems(w.grade["excused"]) == [
        (11, "overlaps ship 14 (ASIAN MOON) in sections 2-5 during hours 92-95"),
        (13, "berths at hour 91, inside the no-movement window 89-95"),
        (14, "berths at hour 92, inside the no-movement window 89-95"),
        (14, "overlaps ship 11 (CMA CGM ADONIS) in sections 2-5 during hours 92-95")]


def test_a_tug_hour_shared_by_ships_frozen_at_different_watches_is_excused_for_both():
    w = week(BUSY, [74], [74])
    assert (froze(w, 3), froze(w, 16)) == (2, 4)
    line = "tugs short at hour 75: your ships need 5, 3 free"
    assert w.grade["feasible"] and problems(w.grade["excused"]) == [(3, line), (16, line)]


def test_an_outage_into_a_forecast_window_is_waived_once():
    outage = ({"type": "tug_outage", "start": 112, "end": 113, "tugs": 8},
              "All 8 of the port's tugs are out of service at hour 112.")
    blew = {"above25": range(100, 112), "episodes": lambda init: [(100, 111, 28)] if init >= 30 else []}
    calm = week(BUSY, **blew)
    w = week(BUSY, **blew, news=[outage])
    assert froze(w, 8) == 4 == next(e["watch"] for e in w.log if w.scheduled[e["event_id"]].event == outage[0])
    assert (calm.grade["cost"], calm.grade["excused_cost"]) == (w.grade["cost"], w.grade["excused_cost"]) == (250, 0)
    assert problems(w.grade["excused"]) == [(8, "tugs short at hour 112: your ships need 3, 0 free")]
    assert w.excused_problems == [] and w.excused_cost == []


def test_a_ship_never_frozen_is_compared_with_the_last_watch_once_the_week_is_done():
    unforecast = week(BUSY, range(163, 166), range(163, 166))
    assert not any(12 in f["ships"] for f in unforecast.frozen)
    lines = [(12, "berths at hour 164, inside the no-movement window 163-166"),
             (13, "berths at hour 164, inside the no-movement window 163-166"),
             (14, "berths at hour 163, inside the no-movement window 163-166")]
    assert unforecast.grade["feasible"] and problems(unforecast.grade["excused"]) == lines
    forecast = week(BUSY, range(163, 166), range(163, 166), lambda init: [(163, 165, 34)] if init >= 96 else [])
    assert problems(forecast.grade["violations"]) == lines


def test_during_the_week_an_open_ship_is_never_excused():
    w = WindWeek(wind.transplant(BUSY, synthetic.record(0, "storm", "2000-01-10", [], [],
                                                        lambda init: [(34, 44, 28)])))
    checked = w.check([{"ship": 3, "berth_hour": 39, "section": 15, "cranes": 5}])
    assert w.view.rules["no_moves"] and not checked["feasible"] and checked["excused_cost"] == 0



def test_with_no_wind_a_feasible_week_grades_as_the_marine_week_does():
    """In an infeasible week the whole-plan excuse may also waive later news's effect on an earlier-frozen ship, which
    the live week's per-notice excuse charged, so only the clean fraction can differ there."""
    calm = synthetic.record(0, "storm", "2000-01-10", [], [], lambda init: [])
    refs = {r["task_id"]: r for r in marine.references()}
    feasible = 0
    for v3_id in marine.task_ids():
        task = wind.transplant(marine.pack().get(v3_id), calm)
        for policy in (lambda w, plans=refs[v3_id]["rolling"]["plans"]: plan_from_list(plans[w.watch]), world.naive):
            ours, v3 = (world.play(task, policy, week=kind).grade for kind in (WindWeek, MarineWeek))
            assert [ours[k] for k in ("feasible", "cost", "raw_cost", "excused_cost")] == [
                v3[k] for k in ("feasible", "cost", "raw_cost", "excused_cost")]
            if ours["feasible"]:
                feasible += 1
                assert ours == v3
    assert feasible == 21
