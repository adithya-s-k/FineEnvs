"""v1 to v4 untouched by the lines week's hooks: their weeks keep v4's timetable and hold every notice to its lead, the
wind week's target and past are the truth and the week as known as before, and their naive weeks grade as their
stored references."""

import pytest
from wind_fixtures import require_wind

from port_week_core import lines, marine, packs, schedule, wind, world
from port_week_core.marine import MarineWeek
from port_week_core.wind import WindWeek

require_wind()

WIND = wind.pack()
WIND_REFS = {r["task_id"]: r for r in wind.references()}
TASK_ID = "dock-24B-w06x1-busy-0-e15"
OUTCOME = ("cost", "reward", "feasible", "excused_cost")


def test_the_earlier_weeks_keep_v4s_timetable_and_hold_every_notice_to_its_lead():
    assert world.Week.timetable is MarineWeek.timetable is WindWeek.timetable is schedule.schedule
    assert world.Week.untimed == MarineWeek.untimed == WindWeek.untimed == ()
    assert (lines.LinesWeek.timetable, lines.LinesWeek.untimed) == (lines.timetable, ("eta", "call"))
    week = WindWeek(WIND.get(TASK_ID))
    assert week.watches == schedule.schedule(week.task)
    assert world.Week.evaluate is world.evaluate and lines.LinesWeek.evaluate is lines.evaluate
    assert (world.Week.anchor_rows, MarineWeek.anchor_rows, WindWeek.anchor_rows, lines.LinesWeek.anchor_rows) == (
        False, False, False, True)
    assert world.Week.revealing is MarineWeek.revealing is WindWeek.revealing is not lines.LinesWeek.revealing
    assert world.Week.draft_excuses is MarineWeek.draft_excuses is WindWeek.draft_excuses is not (
        lines.LinesWeek.draft_excuses)


def test_a_wind_weeks_target_is_the_week_as_known_until_it_is_done_then_the_truth():
    task = WIND.get(TASK_ID)
    week = WindWeek(task)
    assert week.target() == week.view == wind.known(task, week.revealed) != task
    assert week.past(week.revealed, task) == wind.known(task, week.revealed)
    assert week.past(["closure-1"], week.view) == wind.known(task, ["closure-1"])
    week = world.play(task, world.naive, week=WindWeek)
    assert week.target() is task


@pytest.mark.parametrize(("week", "references", "pack"), [
    pytest.param(world.Week, packs.live_references, packs.live_pack, id="live"),
    pytest.param(MarineWeek, marine.references, marine.pack, id="marine"),
    pytest.param(WindWeek, wind.references, wind.pack, id="wind")])
def test_a_naive_week_grades_as_its_stored_reference(week, references, pack):
    ref = references()[0]
    played = world.play(pack().get(ref["task_id"]), world.naive, week=week)
    assert {key: played.grade[key] for key in OUTCOME} == ref["naive"]
    assert played.audit() == {"ok": True, "problems": []}


def test_a_wind_week_played_naive_grades_exactly_as_before():
    ref = WIND_REFS[TASK_ID]
    played = world.play(WIND.get(TASK_ID), world.naive, week=WindWeek)
    assert {key: played.grade[key] for key in OUTCOME} == ref["naive"] == {
        "cost": 663, "reward": 0.201832, "feasible": True, "excused_cost": 120}
    assert "queue" not in {world.rule(v["problem"]) for v in played.grade["violations"] + played.grade["excused"]}
