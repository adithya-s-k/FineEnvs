"""A marine week moved into another week's wind (port_week_core.wind): the transplant, the forecast at every watch, the
week as known and that nothing published after a watch reaches it, the wind section, and whole weeks played in
process."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import synthetic
from berth_core import Task, TaskPack, plan_from_list, plan_to_list
from wind_fixtures import (
    BUST_TASK,
    PACK_DIR,
    REAL_DIR,
    REAL_REFERENCES,
    REAL_WEATHER,
    REFERENCES,
    STORM_TASK,
    WEATHER,
    real,
)

from port_week_core import marine, wind, world
from port_week_core.marine import MarineWeek
from port_week_core.schedule import FREEZE_HOURS, LEADS, MAX_TURNS, ONE_WEEK, SPEAKERS, WIND_MAX_TURNS, schedule
from port_week_core.wind import WindWeek

V3 = marine.pack()
WEEKS = {w["id"]: w for w in map(json.loads, WEATHER.read_text().splitlines())}
TASKS = {t.task_id: t for t in TaskPack(PACK_DIR).tasks}
REFS = {r["task_id"]: r for r in map(json.loads, REFERENCES.read_text().splitlines())}
STORM = TASKS[STORM_TASK]
PAIRS = [(s, w) for s in marine.task_ids() for w in WEEKS]
PACKS = [pytest.param(PACK_DIR, WEATHER, REFERENCES, id="fixture"),
         pytest.param(REAL_DIR, REAL_WEATHER, REAL_REFERENCES, id="data", marks=real)]


def packed(pack_dir: Path, weather: Path, references: Path) -> list[tuple[Task, dict, dict]]:
    weeks = {w["id"]: w for w in map(json.loads, weather.read_text().splitlines())}
    refs = {r["task_id"]: r for r in map(json.loads, references.read_text().splitlines())}
    return [(t, weeks[t.task_id[-3:]], refs[t.task_id]) for t in TaskPack(pack_dir).tasks]


def to_watch(week: world.Week, k: int) -> None:
    while week.watch < k:
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")


def test_the_forecast_party_is_port_control_with_the_freeze_lines_lead():
    assert (SPEAKERS["forecast"], LEADS["forecast"]) == ("Barcelona Port Control", FREEZE_HOURS)
    assert (MAX_TURNS, WIND_MAX_TURNS) == (47, 52)
    assert (world.Week.notice_excuse, MarineWeek.notice_excuse, WindWeek.notice_excuse) == (True, True, False)


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_the_transplant_keeps_the_v3_week_and_its_news_and_drops_the_gale(pair):
    v3, weather = V3.get(pair[0]), WEEKS[pair[1]]
    task = wind.transplant(v3, weather)
    assert task.task_id == f"{v3.task_id}-{weather['id']}" and ONE_WEEK.match(task.task_id)
    assert replace(task, task_id="", rules={}, disruptions=[], notices=[]) == replace(
        v3, task_id="", rules={}, disruptions=[], notices=[])
    assert task.rules == v3.rules | {"no_moves": [
        {"start": w["start"], "end": w["end"], "min_length": w["min_length"],
         "reason": "wind above 25 kn" if w["min_length"] else "wind above 30 kn"} for w in weather["windows"]]}
    news = [(e, n) for e, n in zip(v3.disruptions, v3.notices, strict=True) if e["type"] != "gale"]
    assert list(zip(task.disruptions, task.notices, strict=True))[:len(news)] == news
    forecasts = task.disruptions[len(news):]
    hours = wind.watch_hours(v3, weather)
    assert [e["type"] for e in forecasts] == ["forecast"] * len(hours) and [e["hour"] for e in forecasts] == hours
    watches = schedule(task)
    assert [w.hour for w in watches] == hours
    v3_hours = [w.hour for w in schedule(v3)]
    assert v3_hours == [h for h in hours if h in v3_hours] and set(v3_hours) <= set(hours)
    for w in watches:
        last = w.notices[-1]
        assert (last.kind, last.name, last.event["hour"]) == ("forecast", SPEAKERS["forecast"], w.hour)
    gales = [i for i, e in enumerate(v3.disruptions) if e["type"] == "gale"]
    assert [n.event_id for w in watches for n in w.notices if n.kind != "forecast"] == [
        f"{n.kind}-{n.index - sum(g < n.index for g in gales)}" for w in schedule(v3) for n in w.notices
        if n.kind != "gale"]


@pytest.mark.parametrize(("pack_dir", "weather", "references"), PACKS)
def test_the_packed_tasks_are_the_transplants_with_the_anchor_as_reference(pack_dir, weather, references):
    for task, week, ref in packed(pack_dir, weather, references):
        v3 = V3.get(task.task_id.rsplit("-", 1)[0])
        assert replace(task, reference={}) == replace(wind.transplant(v3, week), reference={})
        assert task.reference == v3.reference | {"optimal_cost": ref["optimal_cost"],
                                                 "optimal_plan": task.reference["optimal_plan"],
                                                 "lower_bound": ref["hindsight_cost"], "proven_optimal": True}


@pytest.mark.parametrize(("pack_dir", "weather", "references"), PACKS)
def test_each_forecast_is_the_run_delivered_at_its_watch_from_the_watch_on(pack_dir, weather, references):
    for task, week, _ in packed(pack_dir, weather, references):
        for e, text in zip(task.disruptions, task.notices, strict=True):
            if e["type"] != "forecast":
                continue
            h, run = e["hour"], wind.delivered(week, e["hour"])
            assert list(e) == ["type", "start", "hour", "init", "issued", "horizon", "kn", "windows", "observed"]
            assert (e["start"], e["init"], e["issued"], e["horizon"], e["kn"]) == (
                h + FREEZE_HOURS, run["init"], run["init"] + 9, run["init"] + 72, run["kn"])
            assert e["issued"] <= h and e["horizon"] > h + FREEZE_HOURS
            assert [(w["start"], w["end"], w["min_length"]) for w in e["windows"]] == [
                (max(w["start"], h), w["end"], w["min_length"]) for w in run["windows"] if w["end"] > h]
            assert [{k: w[k] for k in ("start", "end", "min_length")} for w in e["observed"]] == (
                wind.observed_before(week, h))
            assert {w["reason"] for w in e["windows"]} <= {"forecast: wind above 25 kn", "forecast: wind above 30 kn"}
            assert {w["reason"] for w in e["observed"]} <= {"observed: wind above 25 kn", "observed: wind above 30 kn"}
            assert text == wind.notice(e)


def test_a_watch_is_added_when_a_new_run_restricts_more_of_the_next_36_hours_past_the_freeze_line():
    storm = WEEKS["e00"]
    assert wind.watch_hours(V3.get("dock-24B-w07x1-busy-0"), storm) == [0, 30, 42, 54, 72, 84, 90, 120]
    assert wind.watch_hours(V3.get("dock-36A-w37x1-standard-0"), storm) == [0, 36, 48, 60, 66, 72, 102]
    assert wind.watch_hours(V3.get("dock-24B-w07x1-busy-0"), WEEKS["e01"]) == [0, 30, 42, 54, 84, 90, 120]
    at54, at72 = wind.delivered(storm, 54), wind.delivered(storm, 72)
    assert [(w["start"], w["end"], w["min_length"]) for w in at54["windows"]] == [(75, 85, 300), (77, 80, 0)]
    assert [(w["start"], w["end"], w["min_length"]) for w in at72["windows"]] == [(78, 88, 300), (84, 85, 0)]
    calm = synthetic.record(0, "storm", "2000-01-10", [], [], lambda init: [])
    assert wind.watch_hours(V3.get("dock-24B-w07x1-busy-0"), calm) == [0, 30, 42, 54, 84, 90, 120]


def test_issue_times_and_the_forecast_notice():
    assert (wind.issued(-3), wind.issued(-9), wind.issued(45), wind.issued(81)) == (
        "Sun 21:00", "Sun 15:00", "Tue 21:00", "Thu 09:00")
    assert [n for e, n in zip(STORM.disruptions, STORM.notices, strict=True) if e["type"] == "forecast"][:2] == [
        "Wind forecast issued Sun 21:00 (ECMWF, adjusted to the Dique Sur anemometer), to hour 60: no hour above 25 "
        "kn, peak 12 kn.",
        "Wind forecast issued Tue 03:00 (ECMWF, adjusted to the Dique Sur anemometer), to hour 90: above 25 kn hours "
        "75–85, peak 32 kn; above 30 kn 77–80."]


def test_the_week_as_known_holds_the_latest_forecast_only():
    week = WindWeek(STORM)
    assert wind.known(STORM, []).rules["no_moves"] == []
    to_watch(week, 5)
    view = week.view
    forecasts = [e for e in view.disruptions if e["type"] == "forecast"]
    assert [e["hour"] for e in forecasts] == [84] and wind.forecast(view) is forecasts[0]
    assert view.notices[-1] == wind.notice(forecasts[0]) and len(view.notices) == len(view.disruptions)
    assert view.rules["no_moves"] == forecasts[0]["observed"] + forecasts[0]["windows"]
    assert view.rules["no_moves"] == [
        {"start": 37, "end": 41, "min_length": 300, "reason": "observed: wind above 25 kn"},
        {"start": 74, "end": 84, "min_length": 300, "reason": "observed: wind above 25 kn"},
        {"start": 80, "end": 84, "min_length": 0, "reason": "observed: wind above 30 kn"},
        {"start": 84, "end": 88, "min_length": 300, "reason": "forecast: wind above 25 kn"},
        {"start": 84, "end": 85, "min_length": 0, "reason": "forecast: wind above 30 kn"}]
    assert view.rules | {"no_moves": []} == world.known(STORM, week.revealed).rules
    assert marine.wind(view) == [(37, 41), (74, 84), (84, 88)]


def test_the_wind_section_and_the_viewers_step():
    week = WindWeek(STORM)
    to_watch(week, 5)
    assert wind.section(week.view) == "\n".join([
        "## Wind at the Dique Sur",
        "- Above 25 kn (10-minute mean) ships of 300 m or more may not berth or leave and every movement takes one "
        "more tug; above 30 kn no ship moves. The rules follow the wind that blows; these windows are forecast.",
        "- Barcelona Port Control, issued Thu 09:00 (ECMWF, adjusted to this anemometer), to hour 144: above 25 kn "
        "hours 84–88, peak 31 kn; above 30 kn 84–85.",
        "- kn every 3 h from hour 90: 19 " + " ".join(["12"] * 18),
        "- Observed since hour 0: above 25 kn hours 37–41, 74–84; above 30 kn 80–84."])
    situation = week.situation()["situation"]
    assert situation.endswith("\n\n" + marine.section(week.view) + "\n\n" + wind.section(week.view))
    assert wind.step_wind(week.view) == {
        "hour": 84, "from": "Barcelona Port Control", "issued": 81, "time": "Thu 09:00", "horizon": 144,
        "kn": [[84, 31], [87, 27], [90, 19]] + [[h, 12] for h in range(93, 145, 3)],
        "windows": [{"start": 84, "end": 88, "min_length": 300}, {"start": 84, "end": 85, "min_length": 0}],
        "observed": [{"start": 37, "end": 41, "min_length": 300}, {"start": 74, "end": 84, "min_length": 300},
                     {"start": 80, "end": 84, "min_length": 0}]}
    week = WindWeek(STORM)
    assert wind.section(week.view).endswith(
        "to hour 60: no hour above 25 kn, peak 12 kn.\n- kn every 3 h from hour 6: " + " ".join(["12"] * 19)
        + "\n- Observed since hour 0: no hour above 25 kn.")


def perturbed(weather: dict, hour: int) -> dict:
    """The weather with the wind from ``hour`` on and every run published after ``hour`` changed."""
    blew = {m: {h for o in weather["observed"] if o["min_length"] == m for h in range(o["start"], o["end"]) if h < hour}
            for m in (300, 0)}
    blew[300] |= set(range(hour, hour + 12))
    blew[0] |= set(range(hour + 2, hour + 5))
    runs = []
    for r in weather["runs"]:
        if r["issued"] > hour:
            kn = [x + 15 for x in r["kn"]]
            r = r | {"kn": kn, "windows": wind.run_windows(r["init"], kn)}
        runs.append(r)
    return weather | {"windows": [w | {"unvalidated": 0.5} for w in wind.merge(blew[300], 300)
                                  + wind.merge(blew[0], 0)],
                      "observed": synthetic.spans(blew[300], 300) + synthetic.spans(blew[0], 0), "runs": runs}


@pytest.mark.parametrize("hour", REFS[STORM_TASK]["watch_hours"])
def test_nothing_after_the_watch_reaches_the_week_as_known(hour):
    """The truth from the watch on and every run published after it changed: what the planner sees up to the watch is
    byte for byte the same."""
    v3, plans = V3.get("dock-24B-w07x1-busy-0"), REFS[STORM_TASK]["rolling"]["plans"]
    seen = []
    for weather in (WEEKS["e00"], perturbed(WEEKS["e00"], hour)):
        task = wind.transplant(v3, weather)
        week, out = WindWeek(task), []
        while True:
            out.append(json.dumps([week.situation(), week.check(plans[week.watch]), wind.known(
                task, week.revealed).to_dict(), wind.step_wind(week.view)], ensure_ascii=False))
            if week.hour == hour:
                break
            week.confirm(plans[week.watch])
            to_watch(week, week.watch + 1)
        seen.append((out, task))
    assert seen[0][0] == seen[1][0]
    assert seen[0][1].rules["no_moves"] != seen[1][1].rules["no_moves"]


def test_a_run_delivered_before_it_was_published_stops_the_build():
    weather = json.loads(json.dumps(WEEKS["e00"]))
    run = next(r for r in weather["runs"] if r["init"] == 18)
    run["issued"] = 26
    with pytest.raises(AssertionError):
        wind.transplant(V3.get("dock-24B-w07x1-busy-0"), weather)


@pytest.mark.parametrize("ref", list(REFS.values()), ids=lambda r: r["task_id"])
def test_a_whole_wind_week_played_in_process_on_the_stored_follower_plans(ref):
    task = TASKS[ref["task_id"]]
    plans = ref["rolling"]["plans"]
    week = WindWeek(task)
    while True:
        situation = week.situation()["situation"]
        assert situation.endswith("\n\n" + marine.section(week.view) + "\n\n" + wind.section(week.view))
        target = plan_from_list(plans[week.watch])
        if rows := [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]:
            checked = week.check(rows)
            assert checked["refused"] == [] and checked["entry_problems"] == [] and checked["feasible"]
            assert week.confirm(rows)["refused"] == []
        if week.last:
            assert week.finish("done")["reward"] == ref["rolling"]["reward"]
            break
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")
    assert week.audit() == {"ok": True, "problems": []}
    assert [e["event_id"] for e in week.log if e["kind"] == "forecast"] == [
        n.event_id for w in week.watches for n in w.notices if n.kind == "forecast"]
    assert all(week.scheduled[e["event_id"]].event_hour - e["hour"] == FREEZE_HOURS for e in week.log
               if e["kind"] == "forecast")
    assert week.data()["excused"] == {"problems": [], "cost": []}
    assert {key: week.grade[key] for key in ("cost", "reward", "feasible", "excused_cost")} == {
        key: value for key, value in ref["rolling"].items() if key not in ("config", "plans")}


def test_the_watch_0_forecast_arrives_with_the_load_and_the_rest_with_their_watches():
    week = WindWeek(TASKS[BUST_TASK])
    assert [(e["event_id"], e["via"]) for e in week.log] == [("closure-1", "load"), ("forecast-9", "load")]
    assert week.messages == []
    to_watch(week, 1)
    assert [m["from"] for m in week.drain()][-1] == "Barcelona Port Control"
