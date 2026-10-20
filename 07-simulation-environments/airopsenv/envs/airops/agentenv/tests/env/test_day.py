"""Day in process: the news timetable, the freeze and its refusals, the freeze line, the planning-call budget, changes
that can't be read, the situation, determinism, the grade, the audit, and that nothing reveals the truth the desk
doesn't know yet."""

import json
from dataclasses import replace

import pytest
from airops_core.core.known import CAP_LOOK, FREEZE, LOOK, frozen, view, watch_times
from airops_core.core.model import News, Task, hhmm
from airops_core.core.reward import reward
from airops_core.core.sim import FLOWN, execute
from airops_core.day import ChangeError, Day, programs, read_changes
from airops_core.schedule import timetable, triggers

SMALL = "b6-jfk-20230626-1400-1700"
STORM = "b6-jfk-20230714-1400"


def change(i: int, action: str, not_before: int | None = None) -> dict:
    return {"flight": f"f{i}", "action": action, "not_before": None if not_before is None else hhmm(not_before)}


def advance(day: Day) -> None:
    day.open()
    for n in day.watches[day.watch].news:
        day.notice(n, "trigger")


def test_news_arrives_at_the_first_watch_at_or_after_it_is_issued(days):
    for task in days.values():
        watches = timetable(task)
        assert [w.t for w in watches] == watch_times(task)
        due = {n: w for w in watches for n in w.news}
        for n in task.news:
            if n.at > watches[-1].t:
                assert n.id not in due
                continue
            w = due[n.id]
            assert w.t >= n.at and (w.index == 0 or watches[w.index - 1].t < n.at)
        at = {n.id: n.at for n in task.news}
        assert all(list(w.news) == sorted(w.news, key=lambda n: (at[n], n)) for w in watches)
        assert [t["id"] for t in triggers(watches)] == [f"watch-{w.index}" for w in watches[1:] if w.news]


def test_each_trigger_delivers_its_watchs_news_through_the_hidden_tool_when_advance_reaches_it(days):
    watches = timetable(days[STORM])
    [first] = [t for t in triggers(watches) if t["id"] == "watch-1"]
    assert first == {"id": "watch-1", "when": {"type": "action", "tool": "advance",
                                               "where": {"result.watch": {"equals": 1}}},
                     "barrier": {"at": "provoking_call", "timeout_seconds": 60},
                     "actions": [{"type": "tool", "tool": "airops_notice", "args": {"news_id": n}}
                                 for n in watches[1].news]}


def test_the_first_watchs_news_comes_with_the_briefing_and_later_news_with_the_next_result(days):
    task = days[STORM]
    day = Day(task)
    assert day.revealed == list(day.watches[0].news) and day.drain() == []
    briefing = day.briefing()
    assert all(day.news[n].text in briefing for n in day.watches[0].news)
    assert briefing.endswith("It is the first watch, 13:00. Nothing is frozen yet: set the day up with "
                             "confirm_changes, then call advance.")
    advance(day)
    news = [day.news[n] for n in day.watches[1].news]
    assert day.drain() == [{"at": hhmm(n.at), "from": n.speaker, "kind": n.kind, "text": n.text} for n in news]
    assert day.drain() == []
    metar = max((day.news[n] for n in day.revealed if day.news[n].kind == "metar"), key=lambda n: n.at)
    assert f"Latest METAR ({hhmm(metar.at)}): {metar.text}" in day.text()


def test_nothing_is_frozen_at_the_first_watch_and_the_freeze_is_the_union_over_watches(fixtures):
    task = fixtures[SMALL]
    day = Day(task)
    first = min(task.flights, key=lambda f: (f.sched_dep, f.id))
    assert day.line is None and day.frozen == set()
    assert day.confirm([change(first.id, "cancel")])["confirmed"] == [f"f{first.id}"]
    union = set()
    for _ in range(3):
        plan = dict(day.plan)
        advance(day)
        union |= frozen(task, execute(task, plan, *view(task, day.t)), day.t)
        assert day.frozen == union and day.line == day.t + FREEZE
        assert day.freezes[-1] == {"watch": day.watch, "t": day.t, "line": day.line, "call": 0,
                                   "frozen": sorted(union)}
    assert first.id in day.frozen


def test_changes_to_frozen_flights_are_refused_and_they_stand(fixtures):
    task = fixtures[SMALL]
    day = Day(task)
    advance(day)
    run = day.project(day.plan)
    gone = min(i for i in day.frozen if run.flights[i].status == FLOWN)
    later = min(i for i, r in run.flights.items() if i not in day.frozen and r.status == FLOWN)
    f = task.flight(gone)
    out = day.confirm([change(gone, "cancel"), change(later, "hold", day.line + 30)])
    assert out["refused"] == [{"flight": f"f{gone}", "from": "Duty manager", "reason": (
        f"f{gone} ({f.flight} {f.orig}-{f.dest}) is frozen: your plan has it leaving at {hhmm(run.flights[gone].dep)}, "
        f"before the freeze line at {hhmm(day.line)}; it stands.")}]
    assert out["confirmed"] == [f"f{later}"] and gone not in day.plan
    assert day.plan[later] == {"cancel": False, "not_before": day.line + 30}
    assert day.refusals == [{"watch": 1, **out["refused"][0]}]
    assert day.confirm([change(gone, "hold")])["unchanged"] == [f"f{gone}"]


def test_no_change_reaches_back_past_the_freeze_line(fixtures):
    task = fixtures[SMALL]
    day = Day(task)
    t1 = day.watches[1].t
    naive = execute(task, {}, *view(task, t1)).flights
    soon = task.flight(min(i for i, r in naive.items() if r.status == FLOWN and t1 < r.dep < t1 + FREEZE))
    day.confirm([change(soon.id, "hold", task.start + 5 * FREEZE)])
    advance(day)
    assert soon.id not in day.frozen
    checked = day.check([change(soon.id, "hold")])
    assert checked["held_to_freeze_line"] == [f"f{soon.id}"]
    out = day.confirm([change(soon.id, "hold")])
    assert out["held_to_freeze_line"] == [f"f{soon.id}"] and out["cost"] == checked["cost"]
    assert day.plan[soon.id] == {"cancel": False, "not_before": day.line}
    assert day.changes[-1] | {"call": 0} == {"watch": 1, "t": day.t, "flight": soon.id,
                                             "was": {"cancel": False, "not_before": task.start + 5 * FREEZE},
                                             "now": {"cancel": False, "not_before": day.line}, "by": "planner",
                                             "call": 0}
    run = day.project(day.plan)
    assert all(r.dep >= day.line for i, r in run.flights.items() if r.status == FLOWN and i not in day.frozen)


def test_three_planning_calls_a_watch(fixtures):
    day = Day(fixtures[SMALL])
    assert day.check("{")["planning_calls_left"] == 2
    assert day.check([])["planning_calls_left"] == 1
    assert day.confirm([])["planning_calls_left"] == 0
    assert day.confirm([change(0, "cancel")]) == day.check([]) == {
        "error": "no planning calls left in this watch (3 used): call advance", "planning_calls_left": 0}
    assert day.plan == {} and day.planning_calls == [3]
    advance(day)
    assert day.calls_left == 3 and day.planning_calls == [3, 0]


def test_changes_that_cant_be_read_are_reported_and_the_rest_apply(fixtures):
    task = fixtures[SMALL]
    late = max(task.flights, key=lambda f: f.sched_dep)
    rows, problems = read_changes(task, [
        {"flight": "f999", "action": "cancel", "not_before": None}, {"flight": "f1", "action": "delay"},
        {"flight": "f2", "action": "hold", "not_before": "25:00"}, "f3",
        {"flight": f"f{late.id}", "action": "hold", "not_before": "00:30"},
        {"flight": "4", "action": "hold", "not_before": "18:00"}, {"flight": "f4", "action": "restore"},
        {"flight": "F5", "action": "cancel", "not_before": "16:00"}])
    assert rows == [(4, {"cancel": False, "not_before": None}), (5, {"cancel": True, "not_before": None})]
    assert problems == [
        'unknown flight "f999": use the ids from the flight table (f0, f1, ...)',
        'f1: action must be cancel, restore or hold, not "delay"',
        'f2: not_before "25:00" is not a time: write HH:MM, or HH:MM+1 after midnight',
        'not a change: "f3"',
        f"f{late.id}: not_before 00:30 is {late.sched_dep - 30} minutes before its scheduled departure "
        f"{hhmm(late.sched_dep)}; write 00:30+1 for after midnight",
        "f4 appears more than once: the last change counts"]
    assert read_changes(task, json.dumps({"changes": [change(6, "cancel")]})) == (
        [(6, {"cancel": True, "not_before": None})], [])
    with pytest.raises(ChangeError, match="changes must be a list"):
        read_changes(task, {"flight": "f6"})
    day = Day(task)
    assert day.check("[{") == {"error": 'changes are not valid JSON (Expecting property name enclosed in double '
                                        'quotes): send a list like [{"flight": "f12", "action": "cancel", '
                                        '"not_before": null}, {"flight": "f30", "action": "hold", "not_before": '
                                        '"15:40"}]', "planning_calls_left": 2}


def test_check_confirms_nothing_and_confirm_does_what_it_said(fixtures):
    task = fixtures[SMALL]
    day = Day(task)
    changes = [change(3, "cancel"), change(7, "hold", task.start + 150)]
    checked = day.check(changes)
    assert day.plan == {} and day.changes == []
    assert checked["cost_now"] == execute(task, {}, *view(task, task.start)).cost.total
    plan = {3: {"cancel": True, "not_before": None}, 7: {"cancel": False, "not_before": task.start + 150}}
    run = execute(task, plan, *view(task, task.start))
    assert checked["cost"] == run.cost.to_dict()
    assert checked["auto_cancelled"] == [f"f{i}" for i, r in run.flights.items() if r.status == "auto_cancelled"]
    assert any(row.startswith(f"f3 {task.flight(3).flight} {task.flight(3).orig}-{task.flight(3).dest} cancel "
                              "cancelled - - (was - ") for row in checked["flights"])
    out = day.confirm(changes)
    assert (out["confirmed"], out["cost"], day.plan) == (["f3", "f7"], checked["cost"], plan)


def test_the_situation_reads_the_day_as_known(days):
    task = days[STORM]
    day = Day(task)
    day.confirm([change(5, "cancel"), change(9, "hold", 900)])
    advance(day)
    text, t = day.text(), day.t
    lines = text.splitlines()
    assert lines[0] == ("# JetBlue operations desk, JFK (John F Kennedy International Airport), Friday 2023-07-14; "
                        "times are local (America/New_York).")
    assert lines[2].startswith(f"Now 14:00, watch 1. Freeze line 15:00: flights your plan has leaving before it are "
                               f"frozen ({len(day.frozen)} now)")
    assert lines[lines.index(next(x for x in lines if x.startswith("## Slot caps"))) + 1] == \
        "hour 14 15 16 17 18 19 20 21"
    assert "known through the 18:00 hour" in text and t + CAP_LOOK == 18 * 60
    rows = {line.split()[0]: line.split() for line in lines if line.startswith("f") and line[1].isdigit()}
    run = day.project(day.plan)
    assert len(rows) == len(task.flights)
    for f in task.flights:
        row = rows[f"f{f.id}"]
        assert row[1:4] == [f.flight.split()[1], f"{f.orig}-{f.dest}", f"{hhmm(f.sched_dep)}-{hhmm(f.sched_arr)}"]
        assert row[7] == (hhmm(f.sched_dep + f.lb) if f.sched_dep <= t + LOOK else "?")
        assert row[6] == day.status(f.id, run.flights[f.id])
    assert rows["f5"][6:9] == ["frozen" if 5 in day.frozen else "cancelled", rows["f5"][7], "cancel"]
    assert rows["f9"][8] == "hold@15:00"
    assert {r[6] for r in rows.values()} <= {"flown", "airborne", "frozen", "open", "cancelled"}
    assert "## Aircraft now, as your plan has them" in text and "\nairborne: " in text


def test_programs_follow_the_advisories_received():
    def adv(i: int, at: int, kind: str, element: str = "JFK", **data) -> News:
        return News(f"adv-{i}", at, "advisory", "FAA Command Center", "text",
                    {"advzy": i, "element": element, "kind": kind, "title": "T", **data})

    news = [adv(1, 600, "gs", gs_period=[600, 700]),
            adv(2, 610, "gdp_proposed", "BOS", proposed_period=[700, 1000], program_rate=[30]),
            adv(3, 620, "gdp", "BOS", program_period=[710, 1000], program_rate=[30, 28], avg_delay=60, max_delay=120),
            adv(4, 630, "other", "JFK", title="JFK ARRIVAL DELAYS", event_time=[500, 700]),
            adv(5, 640, "gs", gs_period=[640, 760], probability_of_extension="HIGH", condition="WEATHER / TSTMS"),
            adv(6, 650, "gs_cnx", "LGA", cnx_period=[650, 700])]
    assert programs(news, 645) == [
        "- BOS ground delay program 11:50-16:40, rate 30/28 arrivals/h (all carriers), average delay 60 min, max 120 "
        "[advisory 3, 10:20]",
        "- JFK arrival delays 08:20-11:40 [advisory 4, 10:30]",
        "- JFK ground stop 10:40-12:40, probability of extension high (weather / tstms) [advisory 5, 10:40]"]
    assert programs([*news, adv(7, 700, "gs_cnx", cnx_period=[700, 720]), adv(8, 710, "gdp_cnx", "BOS")], 710) == []


def test_the_day_is_deterministic(days):
    def played():
        day = Day(days[STORM])
        day.confirm([change(5, "cancel"), change(9, "hold", 900)])
        advance(day)
        day.check([change(20, "cancel")])
        day.confirm([change(30, "hold", 1000)])
        advance(day)
        return day, day.text(), json.dumps(day.situation()), day.briefing()
    a, b = played(), played()
    assert a[1:] == b[1:] and json.dumps(a[0].end()) == json.dumps(b[0].end())
    assert json.dumps(a[0].data()) == json.dumps(b[0].data())


def test_nothing_reveals_the_truth_the_desk_doesnt_know_yet(days, fixtures):
    for task in (days[STORM], fixtures[SMALL]):
        k = 2
        t = timetable(task)[k].t
        other = rewritten(task, t)
        a, b = Day(task), Day(other)
        outs = []
        for day in (a, b):
            got = [day.briefing(), day.confirm([change(5, "cancel"), change(9, "hold", task.start + 120)])]
            for _ in range(k):
                advance(day)
                got.append(day.drain())
            got += [day.situation(), day.check([change(30, "cancel")]), day.confirm([change(31, "cancel")]),
                    day.data()]
            outs.append(json.dumps(got))
        assert outs[0] == outs[1]
        assert a.end() != b.end()


def rewritten(task: Task, t: int) -> Task:
    """``task`` with every fact the desk can't know at watch minute t changed: later delays and caps, the news still
    to come, what the airline did, and the references."""
    flights = tuple(replace(f, lb=f.lb + 45, actual=replace(f.actual, dep=0, cancelled=not f.actual.cancelled))
                    if f.sched_dep > t + LOOK else f for f in task.flights)
    caps = {k: {h: v if 60 * h <= t + CAP_LOOK else v + 3 for h, v in hours.items()} for k, hours in task.caps.items()}
    news = tuple(replace(n, text=n.text + " (changed)") if n.at > t else n for n in task.news)
    return replace(task, flights=flights, caps=caps, news=news, reference={"anchor": 1, "floor": 0})


def test_ending_the_day_runs_the_rest_delivers_every_news_once_and_grades_on_the_truth(days, fixtures):
    task = fixtures[SMALL]
    day = Day(task)
    out = day.end()
    ref = task.reference
    assert out == {"end_reason": "end_day", "watch": len(day.watches) - 1, "cost": ref["naive"],
                   "reward": round(reward(ref["naive"], ref["anchor"], ref["floor"]), 6),
                   "audit": {"ok": True, "problems": []}}
    assert day.grade["executed"] == execute(task, {}).rows() and day.data()["reward"] == out["reward"]
    assert len(day.estimates) == len(day.watches)
    storm = Day(days[STORM])
    advance(storm)
    assert 0.2 <= storm.end()["reward"] <= 1
    assert sorted(e["news_id"] for e in storm.log) == sorted(storm.scheduled)
    assert {e["via"] for e in storm.log} == {"load", "trigger", "env"} and storm.audit()["ok"]


def test_the_audit_catches_news_out_of_its_watch_late_or_twice(days):
    day = Day(days[STORM])
    day.open()
    early = day.watches[2].news[0]
    day.notice(early, "trigger")
    for n in day.watches[1].news:
        day.calls += 1
        day.notice(n, "trigger")
    problems = day.audit()["problems"]
    assert problems[0] == f"watch 1 got {[early, *day.watches[1].news]} instead of {list(day.watches[1].news)}"
    assert problems[1:] == [f"{n} arrived after the planner's next call in watch 1" for n in day.watches[1].news]
    day.end()
    assert f"{early} delivered 2 times" in day.audit()["problems"]
    with pytest.raises(ValueError, match="the day is over"):
        day.notice(early, "trigger")
    with pytest.raises(ValueError, match="unknown news 'nope'"):
        Day(days[STORM]).notice("nope", "trigger")
