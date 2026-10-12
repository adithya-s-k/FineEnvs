"""The lines grade (port_week_core.lines.executed, evaluate and LinesWeek.finish): a window that starts before its
ship gets in begins when it does, an hour at anchor costs the ship's sections, first come first served unless the wind
holds the ship whose turn it is, an ETA that proved wrong is never excused, and the floor is the week's at cruising
speed."""

from dataclasses import replace
from functools import cache

import pytest
from berth_core.check import unavoidable_cost
from berth_core.model import Block
from berth_core.solve import naive_replan
from lines_fixtures import require_lines

from port_week_core import lines, marine, world
from port_week_core.lines import LinesWeek

require_lines()

PACK = lines.pack()
E15 = PACK.get("dock-24B-w06x1-busy-0-e15")
E09 = PACK.get("dock-24B-w07x1-busy-0-e09")
CRUISING = {s.id: s.arrival for s in E15.ships}
WEEKS = [pytest.param(t, id=t.task_id) for t in PACK.tasks]


@cache
def naive(task_id: str) -> LinesWeek:
    return world.play(PACK.get(task_id), world.naive, week=LinesWeek)


def cruising(task):
    arrivals = {s.id: s.arrival for s in task.ships}
    return lines.with_arrivals(task, arrivals, arrivals)


def queue(res, ship: int) -> list[str]:
    return [p for p in res.ships[ship].problems if world.rule(p) == "queue"]


def problems(rows: list[dict]) -> list[tuple[int, str]]:
    return [(r["ship"], r["problem"]) for r in rows]


def test_with_arrivals_sets_each_ships_arrival_and_turn_and_nothing_else():
    task = lines.with_arrivals(E15, CRUISING | {3: 42}, CRUISING)
    assert [s.arrival for s in task.ships] == [CRUISING[s.id] if s.id != 3 else 42 for s in E15.ships]
    assert task.rules == E15.rules | {"lines": E15.rules["lines"] | {"turns": list(CRUISING.values())}}
    assert "turns" not in E15.rules["lines"]


def test_a_window_that_starts_before_its_ship_gets_in_begins_when_it_does():
    truth = cruising(E15)
    assert truth.ships[3].arrival == 47
    assert lines.executed(truth, {3: (40, 3, 1)}) == {3: (47, 3, 1)}
    assert lines.executed(truth, {3: (47, 3, 1), 4: (60, 7, 4)}) == {3: (47, 3, 1), 4: (60, 7, 4)}
    early, slid = marine.evaluate(truth, {3: (40, 3, 1)}).ships[3], lines.evaluate(truth, {3: (40, 3, 1)}).ships[3]
    assert [world.rule(p) for p in early.problems] == ["arrival"]
    on_time = marine.evaluate(truth, {3: (47, 3, 1)}).ships[3]
    assert (slid.berth_hour, slid.problems, slid.cost) == (47, [], on_time.cost)


def test_a_late_ship_is_berthed_at_the_first_free_stretch_nearest_its_own_that_fits_and_nobody_else_moves():
    """STEEN gets in at 47 after its window at 40 on sections 3-6: with SEASPAN OCEANIA planned on 2-9 from 49, the
    nearest stretch free for its stay is 10-13; alone, it keeps its own. A ship that gets in before its window waits
    for it."""
    truth = cruising(E15)
    assert lines.executed(truth, {3: (40, 3, 1), 4: (49, 2, 4)}) == {3: (47, 10, 1), 4: (49, 2, 4)}
    assert lines.executed(truth, {3: (40, 3, 1)}) == {3: (47, 3, 1)}
    assert lines.executed(truth, {3: (60, 3, 1)}) == {3: (60, 3, 1)}


def test_a_late_ship_is_berthed_ahead_of_no_earlier_ship_still_waiting_and_late_ships_in_the_order_they_arrive():
    """NAVIOS INDIGO, in at 5 with turn 5, waits at anchor for its window at 60, so STEEN, in at 47, berths no
    earlier. STEEN (in at 47) is placed before SEASPAN OCEANIA (in at 49) whatever the plan's order."""
    truth = cruising(E15)
    assert lines.executed(truth, {3: (40, 3, 1), 1: (60, 12, 3)}) == {1: (60, 12, 3), 3: (60, 3, 1)}
    assert lines.executed(truth, {3: (40, 3, 1), 4: (40, 3, 4)}) == {3: (47, 3, 1), 4: (49, 7, 4)}
    assert lines.executed(truth, {4: (40, 3, 4), 3: (40, 5, 1)}) == {3: (47, 5, 1), 4: (49, 9, 4)}


def test_hour_by_hour_the_port_places_the_late_ships_in_by_then_in_queue_order():
    """With sections 2-14 closed, ONE FREEDOM holds the open stretch until 71. SEASPAN OCEANIA, slowed to 55 at the
    planner's request, keeps its turn, 40, ahead of DALILA, in at 50 unagreed: at 71 both wait and the port berths
    SEASPAN OCEANIA first, where taking them in arrival order would berth DALILA while it waits, a queue break."""
    arrivals = CRUISING | {6: 30, 2: 50, 4: 55}
    task = lines.with_arrivals(replace(E15, blocks=[Block(2, 14, 0, 150, "closed")]), arrivals,
                               arrivals | {4: 40}, agreed=[4])
    plan = {6: (30, 15, 4), 2: (20, 15, 2), 4: (20, 15, 2)}
    assert lines.executed(task, plan) == {6: (30, 15, 4), 4: (71, 15, 2), 2: (124, 15, 2)}
    assert not any(queue(lines.evaluate(task, plan), s) for s in plan)


def test_a_late_ship_waits_out_the_wind_and_with_no_room_for_a_week_its_window_starts_when_it_gets_in():
    """DALILA (334 m) getting in at 45 waits out the wind above 25 kn to 47 and STEEN (138 m) doesn't; NAVIOS INDIGO
    (261 m) in at 98 waits out the hour above 30 kn and takes sections 8-13, clear of the dredging from hour 104;
    SEASPAN OCEANIA (334 m) waits out the blow to 107, and an hour more for tugs while NAVIOS INDIGO leaves. With the
    quay closed all week each window slides to its ship's arrival, once the wind lets it move."""
    held = lines.with_arrivals(E15, CRUISING | {1: 98, 2: 45, 3: 45, 4: 98}, CRUISING)
    assert (44, 47) in held.no_move_windows(held.ships[2]) and (44, 47) not in held.no_move_windows(held.ships[3])
    assert held.no_move_windows(held.ships[1]) == [(98, 99)] and (97, 107) in held.no_move_windows(held.ships[4])
    plan = {1: (90, 9, 5), 2: (40, 7, 4), 3: (40, 3, 1), 4: (90, 7, 4)}
    assert lines.executed(held, plan) == {1: (99, 8, 5), 2: (47, 7, 4), 3: (45, 3, 1), 4: (108, 6, 4)}
    closed = replace(held, blocks=[*held.blocks, Block(2, 22, 0, 400, "closed", "all week", 0)])
    assert lines.executed(closed, plan) == {s: lines._slid(held, s, w) for s, w in plan.items()} == {
        1: (99, 9, 5), 2: (47, 7, 4), 3: (45, 3, 1), 4: (107, 7, 4)}
    truth = cruising(E15)
    assert lines.executed(truth, {2: (45, 7, 4)}) == {2: (45, 7, 4)}
    assert "no_move" in [world.rule(p) for p in lines.evaluate(truth, {2: (45, 7, 4)}).ships[2].problems]


def test_executed_passes_unknown_ship_ids_through():
    plan = {3: (30, 3, 1), len(E15.ships): (5, 2, 2), 99: (0, 4, 1), -1: (3, 1, 1)}
    assert lines.executed(cruising(E15), plan) == plan | {3: (47, 3, 1)}


@pytest.mark.parametrize("task", WEEKS)
def test_an_hour_at_anchor_costs_the_ships_sections_as_an_hour_late_does(task):
    truth = cruising(task)
    plan = naive_replan(truth)
    assert lines.executed(truth, plan) == plan
    ours, theirs = lines.evaluate(truth, plan), marine.evaluate(truth, plan)
    waits = {s.id: max(0, plan[s.id][0] - s.arrival) for s in truth.ships}
    assert any(waits.values())
    for r, was in zip(ours.ships, theirs.ships, strict=True):
        assert r.cost == was.cost + truth.ships[r.ship].sections * waits[r.ship]
        assert [p for p in r.problems if world.rule(p) != "queue"] == was.problems
    assert lines.anchor_hours(truth, ours) == sum(waits.values())


@pytest.mark.parametrize("task", WEEKS)
def test_the_grade_runs_each_window_from_when_its_ship_gets_in_and_never_charges_the_arrival_rule(task):
    week = naive(task.task_id)
    truth = week.truth()
    run = lines.executed(truth, week.plan)
    graded, plain = lines.evaluate(truth, week.plan), marine.evaluate(truth, run)
    assert [r.berth_hour for r in graded.ships] == [run[s.id][0] for s in truth.ships]
    for r, was in zip(graded.ships, plain.ships, strict=True):
        waited = max(0, r.berth_hour - truth.ships[r.ship].arrival)
        assert r.cost == was.cost + truth.ships[r.ship].sections * waited
    rules = {world.rule(v["problem"]) for v in week.grade["violations"] + week.grade["excused"]}
    assert "arrival" not in rules
    assert {s for s in week.plan if run[s] != week.plan[s]} == {
        r.ship for r in marine.evaluate(truth, week.plan).ships if any(world.rule(p) == "arrival" for p in r.problems)}


def test_no_ship_starts_while_one_earlier_in_the_queue_waits_at_anchor_unless_the_wind_holds_it():
    """ONE FREEDOM (366 m, turn 54) waits at anchor for hour 107; VERONICA B (141 m, turn 89) may not berth before it
    at hour 95, but may at 100, when the wind above 25 kn holds the long ship and not the short one."""
    truth = cruising(E15)
    freedom, veronica = truth.ships[6], truth.ships[7]
    assert (freedom.length_m >= 300 > veronica.length_m, freedom.arrival, veronica.arrival) == (True, 54, 89)
    assert (97, 107) in truth.no_move_windows(freedom) and (97, 107) not in truth.no_move_windows(veronica)
    jumped = lines.evaluate(truth, {6: (107, 15, 4), 7: (95, 4, 1)})
    line = "berths at hour 95 while ship 6 ONE FREEDOM, earlier in the queue, waits at anchor"
    assert queue(jumped, 7) == [line] and queue(jumped, 6) == [] and not jumped.feasible
    assert world.rule(line) == "queue" and ("queue", r"earlier in the queue") in world.RULES
    assert queue(lines.evaluate(truth, {6: (107, 15, 4), 7: (100, 4, 1)}), 7) == []
    assert queue(lines.evaluate(truth, {6: (107, 15, 4), 7: (108, 4, 1)}), 7) == []
    assert queue(lines.evaluate(truth, {6: (107, 15, 4), 7: (89, 4, 1)}), 7) == [
        "berths at hour 89 while ship 6 ONE FREEDOM, earlier in the queue, waits at anchor"]


def test_a_ship_the_closures_leave_no_stretch_long_enough_for_holds_no_one_in_the_queue():
    """With sections 2-15 closed until 100, no stretch of 8 is open for ONE FREEDOM waiting since 54, so VERONICA B
    may berth at 95; with 2-14 closed, 15-22 is open for it, and VERONICA B jumps the queue. A stretch open now but
    closing before the ship's stay is out is no room: with 15-22 also closed from 110, ONE FREEDOM's 40 hours from 95
    fit nowhere."""
    plan = {6: (107, 15, 4), 7: (95, 17, 1)}
    jumped = ["berths at hour 95 while ship 6 ONE FREEDOM, earlier in the queue, waits at anchor"]
    for blocks, broken in (([Block(2, 15, 0, 100, "closed")], []), ([Block(2, 14, 0, 100, "closed")], jumped),
                           ([Block(2, 14, 0, 100, "closed"), Block(15, 22, 110, 120, "closed")], [])):
        truth = cruising(replace(E15, blocks=blocks))
        assert queue(lines.evaluate(truth, plan), 7) == broken, blocks
    assert E15.ships[6].handling_for(4) == 40


def test_the_queue_protects_only_a_ship_whose_arrival_was_agreed():
    """Unagreed, ONE FREEDOM's wait at anchor costs its hours there and VERONICA B jumps no queue."""
    plan = {6: (107, 15, 4), 7: (95, 4, 1)}
    unagreed = lines.with_arrivals(E15, CRUISING, CRUISING, agreed=[7])
    assert unagreed.rules["lines"]["agreed"] == [7] and "agreed" not in cruising(E15).rules["lines"]
    assert lines.evaluate(unagreed, plan).ships[7].problems == []
    assert queue(lines.evaluate(lines.with_arrivals(E15, CRUISING, CRUISING, agreed=[6]), plan), 7) == [
        "berths at hour 95 while ship 6 ONE FREEDOM, earlier in the queue, waits at anchor"]
    assert lines.evaluate(unagreed, plan).ships[6].cost == lines.evaluate(cruising(E15), plan).ships[6].cost


def test_a_slowed_ship_keeps_its_turn_but_waits_in_the_queue_only_once_it_has_arrived():
    slowed = lines.with_arrivals(E15, CRUISING | {3: 50}, CRUISING)
    assert (slowed.ships[3].arrival, slowed.rules["lines"]["turns"][3], slowed.ships[4].arrival) == (50, 47, 49)
    assert queue(lines.evaluate(slowed, {3: (60, 3, 1), 4: (50, 7, 4)}), 4) == [
        "berths at hour 50 while ship 3 STEEN, earlier in the queue, waits at anchor"]
    assert queue(lines.evaluate(slowed, {3: (60, 3, 1), 4: (49, 7, 4)}), 4) == []
    by_arrival = lines.with_arrivals(E15, CRUISING | {3: 50}, CRUISING | {3: 50})
    assert queue(lines.evaluate(by_arrival, {3: (60, 3, 1), 4: (50, 7, 4)}), 4) == []
    assert queue(lines.evaluate(by_arrival, {3: (60, 3, 1), 4: (50, 7, 4)}), 3) == []


def test_only_the_first_ship_jumped_is_named():
    truth = cruising(E15)
    res = lines.evaluate(truth, {1: (50, 9, 5), 2: (55, 7, 4), 3: (48, 3, 1)})
    assert queue(res, 3) == ["berths at hour 48 while ship 1 NAVIOS INDIGO, earlier in the queue, waits at anchor"]


def test_an_eta_that_proved_wrong_is_charged_while_the_wind_no_forecast_showed_is_still_excused():
    """Naive on e09: NEXOE MAERSK's window froze at hour 24, after the ETA of hour 20 its agent gave; it got in at 44,
    with PERSEUS alongside its sections, and the port berthed it at the first free stretch, sections 9-13. Its
    lateness and hours at anchor are charged, as the week as known then holds them too with the truth's arrivals; the
    hour more that wind no forecast showed cost it is excused, as are PERSEUS's tugs short at 62. ZIM CHINA's wait from
    107 to 113, for the cranes the outage announced after its last chance took, is excused too."""
    views = []

    def policy(week: LinesWeek):
        views.append(week.view)
        return world.naive(week)

    week = world.play(E09, policy, week=LinesWeek)
    truth, plan = week.truth(), week.plan
    run = lines.executed(truth, plan)

    def then(sid: int):
        decided = next(k for k, f in enumerate(week.frozen) if sid in f["ships"]) - 1
        known = week.past([e["event_id"] for e in week.log if e["watch"] <= decided], truth)
        return views[decided], known

    eta, known = then(1)
    assert (plan[1], next(s.arrival for s in eta.ships if s.id == 1), truth.ships[1].arrival) == ((24, 4, 2), 20, 44)
    assert (lines.executed(eta, plan)[1], lines.executed(known, plan)[1], run[1]) == ((24, 4, 2), (54, 9, 2),
                                                                                      (55, 9, 2))
    assert lines.evaluate(eta, plan).ships[4].berth_hour == 43
    late, held = lines.evaluate(truth, plan).ships[1], lines.evaluate(known, plan).ships[1]
    assert lines.evaluate(eta, plan).ships[1].problems == held.problems == late.problems == []
    assert (lines.evaluate(eta, plan).ships[1].cost, held.cost, late.cost) == (120, 305, 315)
    assert week.excuses()[1] == {1: late.cost - held.cost, 4: 20, 5: 432} == {1: 10, 4: 20, 5: 432}
    assert week.grade["excused_cost"] == 462
    assert (4, "tugs short at hour 62: your ships need 2, 0 free") in problems(week.grade["excused"])
    assert {world.rule(p) for _, p in problems(week.grade["excused"])} == {"no_move", "tugs"}
    assert (truth.no_move_windows(truth.ships[1]), known.no_move_windows(known.ships[1])) == ([(57, 62), (223, 226)],
                                                                                             [])
    assert week.grade["feasible"] and week.grade["violations"] == []


def test_a_draft_is_weighed_with_its_own_excuses():
    """Naive on e09, a draft without PERSEUS earns none for PERSEUS's tugs short at 62, and the dispatch it makes
    changes the cost the news excuses; the earlier weeks weigh every draft with the confirmed plan's excuses."""
    week = naive(E09.task_id)
    draft = {s: w for s, w in week.plan.items() if s != 4}
    assert week.excuses() == ({(2, "tugs"), (3, "tugs"), (3, "no_move"), (4, "tugs")}, {1: 10, 4: 20, 5: 432})
    assert week.draft_excuses(draft) == week.excuses(draft) == ({(2, "tugs"), (3, "tugs"), (3, "no_move")},
                                                                {1: 20, 5: 432, 16: 30})
    assert world.Week.draft_excuses(week, draft) == week.excuses()


@pytest.mark.parametrize("task", WEEKS)
def test_in_the_naive_weeks_no_queue_break_is_excused_and_the_grade_is_the_truths(task):
    week = naive(task.task_id)
    assert "queue" not in {world.rule(p) for _, p in problems(week.grade["excused"])}
    truth = week.truth()
    graded = lines.evaluate(truth, week.plan)
    assert sorted(problems(week.grade["violations"]) + problems(week.grade["excused"])) == sorted(
        (r.ship, p) for r in graded.ships for p in r.problems)
    assert week.grade["raw_cost"] == sum(r.cost for r in graded.ships)
    assert week.grade["anchor_hours"] == lines.anchor_hours(truth, graded)


@pytest.mark.parametrize("task", WEEKS)
def test_the_floor_is_the_weeks_unavoidable_cost_at_cruising_speed(task):
    week = naive(task.task_id)
    assert week.grade["unavoidable_cost"] == unavoidable_cost(cruising(task)) == unavoidable_cost(task)
    assert week.grade["optimal_cost"] == task.reference["optimal_cost"]


def test_slowing_a_ship_does_not_move_the_floor():
    def policy(week: LinesWeek):
        if week.watch == 0:
            week.request("globalliner", 3, 53)
        return world.naive(week)

    week = world.play(E15, policy, week=LinesWeek)
    assert (week.truth().ships[3].arrival, unavoidable_cost(week.truth())) == (53, 332)
    assert week.grade["unavoidable_cost"] == unavoidable_cost(E15) == 308
