from dataclasses import replace

import pytest
from berth_core import evaluate, grade, plan_from_list, plan_to_list

from port_week_core.schedule import label, schedule
from port_week_core.world import Week, excuse, grade_week, known, naive, play, rule


def ids(watches) -> list[str]:
    return [n.event_id for w in watches for n in w.notices]


def to_watch(week: Week, k: int) -> None:
    while week.watch < k:
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")


def test_the_final_known_week_is_the_true_week(one_week):
    assert all(known(t, ids(schedule(t))) == replace(t, reference={}) for t in one_week)


def test_the_week_as_known_at_hour_0(example):
    view = Week(example).view
    assert len(view.ships) == 16 and view.reference == {}
    assert (view.ships[7].arrival, view.ships[10].arrival, [view.ships[s].arrival for s in (12, 13, 14)]) == (
        78, 112, [144, 144, 156])
    assert (view.ships[5].berth_deadline, view.ships[5].deadline_penalty) == (None, 0)
    assert view.rules["crane_outages"] == [] and view.rules["no_moves"] == []
    assert [(b.first, b.last, b.kind) for b in view.blocks] == [(4, 7, "alongside"), (10, 18, "closed")]
    assert (view.notices, view.disruptions) == ([example.notices[2]], [example.disruptions[2]])


@pytest.mark.parametrize("kind", ["late", "bunching", "extra", "priority", "emergency", "crane_outage", "gale"])
def test_each_kind_of_news_changes_the_known_week(one_week, kind):
    task, n = next((t, n) for t in one_week for w in schedule(t)[1:] for n in w.notices if n.kind == kind)
    week = Week(task)
    to_watch(week, next(w.index for w in week.watches if n in w.notices) - 1)
    week.open()
    for m in week.watches[week.watch].notices[:week.watches[week.watch].notices.index(n)]:
        week.notice(m.event_id, m.name, m.text, "trigger")
    week.drain()
    before = week.view
    assert week.notice(n.event_id, n.name, n.text, "trigger") == {"applied": n.event_id, "watch": week.watch}
    after, e = week.view, n.event
    if kind == "late":
        assert (before.ships[e["ship"]].arrival, after.ships[e["ship"]].arrival) == (
            task.ships[e["ship"]].arrival - e["hours"], task.ships[e["ship"]].arrival)
    elif kind == "bunching":
        assert [before.ships[s].arrival for s in e["ships"]] == [task.ships[s].planned_hour for s in e["ships"]]
        assert [after.ships[s].arrival for s in e["ships"]] == [task.ships[s].arrival for s in e["ships"]]
    elif kind == "extra":
        assert (len(before.ships), len(after.ships), after.ships[-1].name) == (e["ship"], e["ship"] + 1, n.name)
    elif kind == "priority":
        assert (before.ships[e["ship"]].weight, after.ships[e["ship"]].weight) == (1, 3)
    elif kind == "emergency":
        assert before.ships[e["ship"]].berth_deadline is None
        assert (after.ships[e["ship"]].berth_deadline, after.ships[e["ship"]].deadline_penalty) == (
            e["deadline"], task.ships[e["ship"]].deadline_penalty)
    elif kind == "crane_outage":
        outage = {"start": e["start"], "end": e["end"], "cranes": e["cranes"]}
        assert outage not in before.rules["crane_outages"] and outage in after.rules["crane_outages"]
    else:
        windows = [(w["start"], w["end"]) for w in after.rules["no_moves"]]
        assert (e["start"], e["end"]) in windows and (e["core_start"], e["core_end"]) in windows
        assert len(windows) == len(before.rules["no_moves"]) + 2
    assert n.text not in before.notices and n.text in after.notices
    assert week.drain() == [{"hour": week.hour, "from": n.name, "text": n.text}]


def test_closures_are_applied_when_the_week_loads(one_week):
    for task in one_week:
        view = Week(task).view
        closed = [(b.first, b.last, b.start, b.end) for b in view.blocks if b.kind == "closed"]
        assert closed == [(b.first, b.last, b.start, b.end) for b in task.blocks if b.kind == "closed"]


def test_with_nothing_excused_the_live_grade_is_v1s(pack):
    for task in pack.tasks:
        for plan in (task.reference["optimal_plan"], task.reference["naive_plan"]):
            live = grade_week(task, plan_from_list(plan), (), {})
            v1 = grade(task, plan_from_list(plan))
            assert (live["reward"], live["feasible"], live["cost"]) == (v1.reward, v1.feasible, v1.cost)
        assert grade_week(task, plan_from_list(task.reference["optimal_plan"]), (), {})["reward"] == 1.0


def test_problem_lines_map_to_their_rules(one_week):
    assert [rule(p) for p in [
        "missing from the plan",
        "berths at hour 116, inside the no-movement window 115-144",
        "gets 0 cranes but can be worked by 1-5",
        "gets -2 cranes but can be worked by 1-5",
        "berths at hour 70 but cannot arrive before hour 78",
        "needs sections 20-25 but the quay has sections 2-22",
        "overlaps closed sections 10-18 during hours 0-39",
        "overlaps ZIM ATLANTIC (alongside) 4-7 during hours 0-1",
        "overlaps ship 12 (RACHEL BORCHARD) in sections 8-9 during hours 164-170",
        "cranes over the pool at hours 108 (e.g. hour 108: 9 in use, 7 available)",
        "leaves at hour 126, when 3 ships move (limit 2 per hour)",
    ]] == ["missing", "no_move", "cranes", "cranes", "arrival", "quay", "closed", "alongside", "ship 12",
           "crane_pool", "moves"]
    lines = {p for t in one_week for r in evaluate(t, naive(Week(t))).ships for p in r.problems}
    assert len(lines) > 100 and all(rule(p) for p in lines)


def carlota_b_at_95(example) -> dict:
    plan = plan_from_list(example.reference["optimal_plan"])
    assert plan[7] == (93, 4, 2)
    return plan | {7: (95, 4, 2)}


def test_a_gale_on_a_frozen_window_is_excused(example):
    plan = carlota_b_at_95(example)
    week = play(example, lambda w: plan)
    assert week.frozen[5] == {"watch": 5, "hour": 90, "before": 96, "call": 0,
                              "ships": sorted(s for s, e in plan.items() if e[0] < 96)}
    assert 7 in week.frozen[5]["ships"]
    assert (week.excused_cost, week.excused_problems, week.refusals) == (
        [{"event_id": "gale-0", "ship": 7, "cost": 20}], [], [])
    assert week.grade == {"reward": 1.0, "feasible": True, "clean_fraction": 1.0, "quality": 1.0, "cost": 226,
                          "raw_cost": 246, "excused_cost": 20, "optimal_cost": 226, "unavoidable_cost": 177,
                          "regret": 0, "violations": [], "excused": []}
    unexcused = grade_week(example, plan, (), {})
    assert (unexcused["cost"], round(unexcused["reward"], 4)) == (246, 0.8116)
    checked = week.check([])
    assert (checked["feasible"], checked["cost"], checked["excused_cost"]) == (True, 226, 20)
    assert checked["delay_cost"] + 5 * checked["moves"] == 226
    assert next(r for r in checked["ships"] if r["ship"] == 7) | {"departure": None} == {
        "ship": 7, "berth_hour": 95, "section": 4, "cranes": 2, "departure": None, "delay_h": 5, "moved": False,
        "cost": 20, "problems": [], "excused": [], "excused_cost": 20}


def test_an_outage_under_frozen_stays_is_excused(example):
    week = Week(example)
    week.confirm([{"ship": 3, "berth_hour": 84, "section": 15, "cranes": 7},
                  {"ship": 8, "berth_hour": 85, "section": 2, "cranes": 2}])
    to_watch(week, 4)
    problem = "cranes over the pool at hours 108 (e.g. hour 108: 9 in use, 7 available)"
    assert week.excused_problems == [{"event_id": "crane_outage-6", "ship": s, "rule": "crane_pool", "problem": problem}
                                     for s in (3, 8)]
    checked = {r["ship"]: r for r in week.check([])["ships"]}
    assert [(checked[s]["problems"], checked[s]["excused"]) for s in (3, 8)] == [([], [problem])] * 2
    week.finish("done")
    assert {"ship": 3, "problem": problem} in week.grade["excused"]
    assert all(v["ship"] not in (3, 8) or "pool" not in v["problem"] for v in week.grade["violations"])


def test_only_what_the_news_added_is_excused(example):
    revealed = ["closure-2", "extra-1", "emergency-7", "late-3", "late-4"]
    before, after = known(example, revealed), known(example, [*revealed, "crane_outage-6"])
    frozen = {3: (84, 15, 7), 8: (85, 2, 2), 1: (2, 4, 2)}
    assert [rule(p) for p in evaluate(before, frozen).ships[1].problems] == ["arrival"]
    problems, cost = excuse(before, after, frozen)
    assert [(s, r) for s, r, _ in problems] == [(3, "crane_pool"), (8, "crane_pool")] and cost == {}
    assert excuse(after, after, frozen) == ([], {})


def test_planning_calls_and_refusals(example):
    week = Week(example)
    assert week.check("{")["error"].startswith("plan is not valid JSON") and week.planning_calls == [1]
    first = plan_to_list(naive(week))
    out = week.confirm(first)
    assert (out["confirmed"], out["refused"], out["planning_calls_left"]) == (list(range(16)), [], 1)
    assert week.confirm(first) == {"confirmed": [], "unchanged": list(range(16)), "refused": [], "entry_problems": [],
                                   "planning_calls_left": 0}
    assert week.check([]) == {"error": "no planning calls left in this watch (3 used): call advance",
                              "planning_calls_left": 0}
    assert week.planning_calls == [3]
    week.open()
    assert (week.watch, week.hour, week.before, week.planning_calls) == (1, 30, 36, [3, 0])
    frozen = week.fixed
    assert frozen == {s: e for s, e in week.plan.items() if e[0] < 36} and 0 in frozen
    h0 = frozen[0][0]
    moved = next(s for s, e in week.plan.items() if e[0] >= 36)
    out = week.confirm([{"ship": 0, "berth_hour": 40, "section": frozen[0][1]},
                        {"ship": moved, "berth_hour": 35, "section": week.plan[moved][1]},
                        {"ship": 16, "berth_hour": 58, "section": 3, "cranes": 2}])
    assert out["refused"] == [
        {"ship": 0, "from": "Harbour master",
         "reason": f"ship 0 ZIM ATLANTIC is frozen: its window starts at hour {h0}, before the freeze line at hour 36; "
                   "it stands."},
        {"ship": moved, "from": "Harbour master",
         "reason": f"ship {moved} {example.ships[moved].name} can't start at hour 35: windows must start at or after "
                   "the freeze line at hour 36."}]
    assert (out["confirmed"], 16 in week.plan, week.plan[0][0]) == ([], False, h0)
    assert out["entry_problems"] == ["unknown ship 16: use the ship ids from the table"]
    assert week.refusals == [{"watch": 1, "ship": r["ship"], "reason": r["reason"]} for r in out["refused"]]


def test_advance_freezes_berths_and_sails_on_the_confirmed_windows(example):
    week = Week(example)
    week.confirm(plan_to_list(naive(week)))
    plan, seen = dict(week.plan), set()
    while not week.last:
        out = week.open()
        hour, before, view = week.hour, week.hour + 6, week.view
        assert (out["watch"], out["hour"], out["time"], out["frozen_before"]) == (week.watch, hour, label(hour), before)
        assert out["note"] == "This watch's bulletin arrives with your next tool result."
        windows = {w["ship"]: w for w in week.windows(view)}
        for sid, (h, _, cranes) in plan.items():
            dep = view.departure(view.ships[sid], h + view.ships[sid].handling_for(cranes))
            status = "open" if h >= before else "departed" if dep <= hour else "berthed" if h <= hour else "frozen"
            assert (windows[sid]["status"], windows[sid]["departure"]) == (status, dep)
            seen.add(status)
        assert out["berthed"] == [s for s, w in windows.items() if w["status"] == "berthed"]
        assert out["departed"] == [s for s, w in windows.items() if w["status"] == "departed"]
        assert week.frozen[-1] == {"watch": week.watch, "hour": hour, "before": before, "call": 0,
                                   "ships": sorted(s for s, e in plan.items() if e[0] < before)}
        assert out["unconfirmed"] == ([] if week.watch == 1 else [16])
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")
        assert week.situation()["unconfirmed"] == [16]
    assert seen == {"open", "frozen", "berthed", "departed"}


def test_the_naive_policy_is_deterministic(one_week, example):
    for task in (t for t in one_week if t.split == "eval"):
        a, b = play(task, naive), play(task, naive)
        assert a.plan == b.plan and a.grade == b.grade and a.data() == b.data()
        assert a.refusals == [] and a.audit() == {"ok": True, "problems": []} and a.end_reason == "done"
    week = play(example, naive)
    assert (week.grade["cost"], week.grade["reward"], week.planning_calls) == (693, 0.201516, [1, 1, 0, 1, 1, 0, 1])


def test_end_runs_the_rest_of_the_week_on_the_confirmed_windows(example):
    week = Week(example)
    week.confirm(plan_to_list(naive(week)))
    out = week.end()
    assert out == {"end_reason": "end_week", "watch": 6, "feasible": week.grade["feasible"],
                   "cost": week.grade["cost"], "reward": week.grade["reward"], "audit": {"ok": True, "problems": []}}
    assert [e["via"] for e in week.log] == ["load"] + ["env"] * 7 and week.watch == 6
    assert week.end() == out
    with pytest.raises(ValueError, match="^the week is over$"):
        week.notice("gale-0", "Harbour master", example.notices[0], "trigger")
    with pytest.raises(ValueError, match="^unknown event 'gale-9'$"):
        week.notice("gale-9", "Harbour master", "", "trigger")


def test_the_audit_names_each_failure(example):
    week = Week(example)
    week.open()
    assert week.audit()["problems"] == ["watch 1 got [] instead of ['extra-1']"]
    week.calls += 1
    week.notice("extra-1", "MAERSK NUBA", "Unscheduled call", "trigger")
    assert week.audit()["problems"] == ["extra-1 arrived after the agent's next call in watch 1",
                                        "extra-1: the message differs from the schedule"]
    week.notice("extra-1", "MAERSK NUBA", example.notices[1], "trigger")
    assert "watch 1 got ['extra-1', 'extra-1'] instead of ['extra-1']" in week.audit()["problems"]
    while not week.last:
        week.open()
    week.notice("late-4", "MAERSK NARMADA", example.notices[4], "trigger")
    assert "late-4: revealed -8 h ahead at hour 120" in week.audit()["problems"]
    week.finish("done")
    problems = week.audit()["problems"]
    assert "extra-1 revealed 2 times" in problems and "gale-0 revealed 0 times" in problems
    assert "late-4 revealed 1 times" not in problems and "closure-2 revealed 1 times" not in problems


def test_a_clean_week_passes_the_audit(example):
    week = play(example, naive)
    assert week.audit() == {"ok": True, "problems": []}
    assert [(e["event_id"], e["watch"], e["via"], e["call"], e["matches"]) for e in week.log] == [
        ("closure-2", 0, "load", 0, True), ("extra-1", 1, "trigger", 0, True), ("emergency-7", 2, "trigger", 0, True),
        ("late-3", 3, "trigger", 0, True), ("late-4", 4, "trigger", 0, True), ("crane_outage-6", 4, "trigger", 0, True),
        ("gale-0", 5, "trigger", 0, True), ("bunching-5", 6, "trigger", 0, True)]
    assert [f["watch"] for f in week.frozen] == list(range(len(week.watches)))
