from dataclasses import replace

import berth_core
import pytest
from berth_core import load_pack, plan_from_list, plan_to_list

from port_week_core import marine, world
from port_week_core.marine import MarineWeek
from port_week_core.schedule import LEADS, SPEAKERS, label, schedule

PACK = marine.pack()
EXAMPLE = PACK.get("dock-24B-w07x1-busy-0")
V2 = load_pack().get("dock-24B-w07x1-busy-0")


def to_watch(week: world.Week, k: int) -> None:
    while week.watch < k:
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")


def test_the_marine_kinds_have_their_parties_and_a_days_notice():
    assert (SPEAKERS["tug_outage"], SPEAKERS["pilot_shortage"]) == ("Tug company", "Pilot station")
    assert (LEADS["tug_outage"], LEADS["pilot_shortage"]) == (24, 24)


def test_pilot_and_tug_lines_map_to_their_rules():
    assert world.rule("pilots short at hour 54: your ships need 2, 0 free") == "pilots"
    assert world.rule("tugs short at hour 54: your ships need 3, 2 free") == "tugs"


def test_a_live_week_grades_with_berth_core_and_a_marine_week_with_the_marine_checker():
    assert world.Week.evaluate is berth_core.evaluate and MarineWeek.evaluate is marine.evaluate
    plan = plan_from_list(V2.reference["optimal_plan"])
    assert world.grade_week(EXAMPLE, plan, (), {})["feasible"]
    graded = world.grade_week(EXAMPLE, plan, (), {}, evaluate=marine.evaluate)
    assert not graded["feasible"] and graded["violations"] == [
        {"ship": s, "problem": "tugs short at hour 54: your ships need 3, 2 free"} for s in (4, 5)]


def test_the_example_weeks_tuesday_bulletin_brings_the_tug_outage_and_the_pilot_shortage():
    watch = next(w for w in schedule(EXAMPLE) if w.hour == 30)
    assert label(watch.hour) == "Tue 06:00"
    assert [(n.event_id, n.name, n.event_hour, n.text) for n in watch.notices[1:]] == [
        ("tug_outage-8", "Tug company", 54,
         "2 of the port's 8 tugs are out of service from hour 54 to hour 78: 6 tugs in that window."),
        ("pilot_shortage-9", "Pilot station", 54,
         "2 of the port's 7 pilots on duty are unavailable from hour 54 to hour 66: 5 pilots in that window.")]
    assert [w.hour for w in schedule(EXAMPLE)] == [w.hour for w in schedule(V2)]
    gale = next(n for w in schedule(EXAMPLE) for n in w.notices if n.kind == "gale")
    assert gale.text.endswith(" Ships that berth or leave from hour 115 to hour 144 take one more tug.")


def test_v2s_optimum_runs_short_of_tugs_at_hour_54_once_the_news_is_in():
    week = MarineWeek(EXAMPLE)
    rows = [r for r in V2.reference["optimal_plan"] if r["ship"] < len(week.view.ships)]
    assert week.check(rows)["feasible"]
    week.confirm(rows)
    to_watch(week, 1)
    checked = week.check([])
    assert not checked["feasible"]
    assert [(r["ship"], r["problems"]) for r in checked["ships"] if r["problems"]] == [
        (4, ["tugs short at hour 54: your ships need 3, 2 free"]),
        (5, ["tugs short at hour 54: your ships need 3, 2 free"]),
        (16, ["missing from the plan"])]
    situation = week.situation()["situation"]
    assert situation.endswith(marine.section(week.view)) and "- Tug company: 2 of the 8 tugs out" in situation


def test_a_tug_outage_over_a_frozen_departure_is_excused_per_ship_and_rule():
    plan = plan_from_list(EXAMPLE.reference["optimal_plan"])
    res = berth_core.evaluate(EXAMPLE, plan)
    assert (res.ships[8].berth_hour, res.ships[8].departure) == (79, 103)
    assert [h for r in res.ships for h in (r.berth_hour, r.departure)].count(103) == 1
    outage = {"type": "tug_outage", "start": 103, "end": 104, "tugs": 8}
    task = replace(EXAMPLE, disruptions=EXAMPLE.disruptions + [outage],
                   notices=EXAMPLE.notices + ["All 8 of the port's tugs are out of service at hour 103."])
    week = world.play(task, lambda w: plan, week=MarineWeek)
    watch = next(w for w in week.watches if any(n.event_id == "tug_outage-10" for n in w.notices))
    assert watch.hour == 78 and 8 in week.frozen[watch.index]["ships"]
    line = "tugs short at hour 103: your ships need 3, 0 free"
    assert week.excused_problems == [{"event_id": "tug_outage-10", "ship": 8, "rule": "tugs", "problem": line}]
    assert week.grade["feasible"] and week.grade["excused"] == [{"ship": 8, "problem": line}]
    assert week.grade["cost"] == EXAMPLE.reference["optimal_cost"] and week.audit()["ok"]
    assert not world.grade_week(task, plan, (), {}, evaluate=marine.evaluate)["feasible"]


def test_the_hour_0_view_is_the_same_without_the_gale():
    gales = 0
    for task in PACK.tasks:
        kept = [i for i, e in enumerate(task.disruptions) if e["type"] != "gale"]
        calm = replace(task, rules=task.rules | {"no_moves": []}, disruptions=[task.disruptions[i] for i in kept],
                       notices=[task.notices[i] for i in kept])
        gales += len(task.disruptions) - len(kept)
        assert marine.free(MarineWeek(task).view) == marine.free(MarineWeek(calm).view)
        assert marine.section(MarineWeek(task).view) == marine.section(MarineWeek(calm).view)
    assert gales == 3


def test_the_hidden_call_is_missing_from_the_section_until_it_is_announced():
    week = MarineWeek(EXAMPLE)
    assert "16" not in marine.section(week.view).split("\n")[3]
    to_watch(week, 1)
    assert "16" in marine.section(week.view).split("\n")[3]


@pytest.mark.parametrize("ref", marine.references(), ids=lambda r: r["task_id"])
def test_a_whole_marine_week_played_in_process_on_the_stored_rolling_plans(ref):
    task = PACK.get(ref["task_id"])
    plans = ref["rolling"]["plans"]
    week = MarineWeek(task)
    while True:
        assert week.situation()["situation"].endswith(marine.section(week.view))
        target = plan_from_list(plans[week.watch])
        if rows := [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]:
            checked = week.check(rows)
            assert checked["refused"] == [] and checked["entry_problems"] == []
            assert week.confirm(rows)["refused"] == []
        if week.last:
            assert week.finish("done")["reward"] == ref["rolling"]["reward"]
            break
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")
    assert week.audit() == {"ok": True, "problems": []}
    assert {key: week.grade[key] for key in ("cost", "reward", "feasible", "excused_cost")} == {
        key: value for key, value in ref["rolling"].items() if key != "plans"}
