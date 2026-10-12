"""The lines pack (port_week_core.lines): the wind weeks with each ship's agent, each arrival off its published
schedule by the agent's long-range error and the arrival news given as the agents' ETAs, the ETA model, and the agents'
reports carried by v4's watches with no watch added and no news given early."""

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest
from lines_fixtures import require_lines

from port_week_core import lines, packs, wind
from port_week_core.schedule import schedule

require_lines()

PACK = lines.pack()
WIND = wind.pack()
MANIFEST = json.loads((lines.pack_dir() / "manifest.json").read_text())
E15 = PACK.get("dock-24B-w06x1-busy-0-e15")
WEEKS = [pytest.param(t, id=t.task_id) for t in PACK.tasks]


def reports(watch) -> list:
    return [n for n in watch.notices if n.kind in lines.REPORTS]


def news(task) -> dict[int, list[tuple[int, int]]]:
    """Each late, bunched or unscheduled ship and the (watch, hour) of each v4 notice about it."""
    out: dict[int, list[tuple[int, int]]] = {}
    for w in schedule(task):
        for n in w.notices:
            if n.kind in lines.NEWS_ETA:
                for s in [n.event["ship"]] if n.kind != "bunching" else n.event["ships"]:
                    out.setdefault(s, []).append((w.index, n.hour))
    return out


def reachable(task, sid: int, arrival: int, planned: int) -> int:
    """The first arrival from ``arrival`` on from which the slowest course from hour 0 still makes ``planned``."""
    return next(a for a in range(arrival, arrival + 500) if lines.slowest(task, sid, 0, a) >= planned)


def delayed(task, sid: int, hour: int) -> int:
    """What the ship's agent said at ``hour`` before the news, moved later by the delay, between the next hour and the
    slowest the ship can arrive."""
    before, truth = lines.before_news(task, sid), task.ships[sid].arrival
    eta = lines.estimate(task, sid, hour, before) + truth - before
    return min(max(hour + 1, eta), lines.slowest(task, sid, hour, truth))


def test_the_pack_sits_beside_the_wind_pack_under_the_tasks_root(monkeypatch, tmp_path):
    assert lines.pack_dir() == packs.tasks_root() / packs.LINES_PACK == wind.pack_dir().parent / packs.LINES_PACK
    monkeypatch.setenv("PORT_WEEK_TASKS_DIR", str(tmp_path))
    assert lines.pack_dir() == tmp_path / packs.LINES_PACK
    with pytest.raises(wind.NotBuilt, match="run python tools/build_lines.py"):
        lines.pack()


def test_the_manifest_pins_the_pack_and_the_wind_pack_it_was_made_from():
    body = (lines.pack_dir() / "tasks.jsonl").read_bytes()
    assert {key: MANIFEST[key] for key in ("name", "split", "tasks", "tiers", "quays")} == {
        "name": "dock-v1-lines", "split": "eval", "tasks": 15, "tiers": {"busy": 12, "standard": 3},
        "quays": {"24B": 7, "36A": 8}}
    assert MANIFEST["sha256"] == hashlib.sha256(body).hexdigest()
    assert MANIFEST["wind"] == {"pack": "dock-v1-wind", "sha256": hashlib.sha256(
        (wind.pack_dir() / "tasks.jsonl").read_bytes()).hexdigest()}
    assert list(MANIFEST["lines"]["grounding"]) == ["agents", "reports", "eta_error", "speeds", "queue", "anchor"]
    assert all(list(item) == ["value", "basis"] for item in MANIFEST["lines"]["grounding"].values())


@pytest.mark.parametrize("task", WEEKS)
def test_a_lines_week_is_its_wind_week_with_the_agents_the_arrivals_off_schedule_and_the_news_as_their_etas(task):
    """Only the arrivals and the deadlines that move with them, rules.lines, the reference and the late, bunching,
    unscheduled-call and emergency news differ: each now carries the agent's ETA as of the watch it reaches the
    planner, or the moved deadline."""
    v4 = WIND.get(task.task_id)
    for s, was in zip(task.ships, v4.ships, strict=True):
        assert s.berth_deadline == (None if was.berth_deadline is None else
                                    was.berth_deadline + s.arrival - was.arrival)
    restored = [replace(s, berth_deadline=was.berth_deadline) for s, was in zip(lines.as_v4(task).ships, v4.ships,
                                                                                 strict=True)]
    assert replace(task, ships=restored, rules={}, reference={}, disruptions=[], notices=[]) == replace(
        v4, rules={}, reference={}, disruptions=[], notices=[])
    assert {k: v for k, v in task.rules.items() if k != "lines"} == v4.rules
    firms = task.rules["lines"]["firms"]
    assert list(task.rules["lines"]) == ["firms", "ships"] and list(firms) == sorted(firms)
    assert [list(m) for m in task.rules["lines"]["ships"]] == [
        ["firm", "leg_nm", "bias", "scheduled", "before_news", "v4_arrival"]] * len(task.ships)
    assert {lines.firm(task, s.id) for s in task.ships} == set(firms)
    at = {n.event_id: w.hour for w in schedule(v4) for n in w.notices}
    for i, (e, text, was, was_text) in enumerate(zip(task.disruptions, task.notices, v4.disruptions, v4.notices,
                                                     strict=True)):
        hour = at[f"{was['type']}-{i}"]
        if was["type"] == "late":
            assert e == was | {"eta": delayed(task, was["ship"], hour)}
            assert text == was_text.replace("was delayed leaving", "has been held up at").replace(
                f"it now arrives at hour {v4.ships[was['ship']].arrival}", f"its agent now gives ETA hour {e['eta']}")
        elif was["type"] == "extra":
            assert e == was | {"eta": lines.estimate(task, was["ship"], hour)}
            assert f"has ETA hour {e['eta']} from its agent" in text
        elif was["type"] == "bunching":
            assert e == was | {"etas": [delayed(task, s, hour) for s in was["ships"]]}
            assert text == was_text.replace(
                was_text[was_text.index(" and now arrive together"):was_text.index(" Their planned")],
                "; their agents' ETAs: " + ", ".join(f"ship {s} hour {x}" for s, x in zip(e["ships"], e["etas"],
                                                                                            strict=True)) + ".")
        elif was["type"] == "emergency":
            assert e == was | {"deadline": task.ships[was["ship"]].berth_deadline}
            assert text == was_text.replace(f"docks by hour {was['deadline']}", f"docks by hour {e['deadline']}")
        else:
            assert (e, text) == (was, was_text)


def test_a_delay_notice_moves_the_eta_its_agent_gave_before_the_news_later_by_the_delay():
    """NANTO's agent would have said hour 59 at the news, of the 63 it was due before it; 20 hours late, it says 79.
    ONE FREEDOM's would have said 50 of 48; 6 hours late, 56."""
    seen = []
    for task in PACK.tasks:
        at = {n.event_id: w.hour for w in schedule(lines.as_v4(task)) for n in w.notices}
        for i, e in enumerate(task.disruptions):
            if e["type"] == "late":
                rows = [(e["ship"], e["eta"])]
            elif e["type"] == "bunching":
                rows = list(zip(e["ships"], e["etas"], strict=True))
            else:
                continue
            for sid, eta in rows:
                hour = at[f"{e['type']}-{i}"]
                assert eta == delayed(task, sid, hour) <= lines.slowest(task, sid, hour, task.ships[sid].arrival)
                seen.append((eta, lines.before_news(task, sid)))
    assert (len(seen), sum(eta <= before for eta, before in seen)) == (57, 11)
    assert (lines.estimate(E15, 5, 30, 63), E15.disruptions[3]["eta"]) == (59, 79)
    assert (lines.estimate(E15, 6, 42, 48), E15.disruptions[4]["eta"]) == (50, 56)


def test_each_ships_agent_and_how_far_off_it_is():
    assert E15.rules["lines"]["firms"] == {
        "aperez": "A. Pérez y Cía", "boluda": "Boluda Lines", "cmacgm": "CMA CGM Ibérica",
        "cosco": "COSCO Shipping Lines Spain", "globalliner": "Global Liner Agencias", "hapag": "Hapag-Lloyd Spain",
        "noatum": "Noatum Maritime", "one": "Ocean Network Express Spain"}
    assert lines.meta(E15, 3) == {"firm": "globalliner", "leg_nm": 165, "bias": 0.739, "scheduled": 36,
                                  "before_news": None, "v4_arrival": 36}
    assert (lines.firm(E15, 3), lines.firm_name(E15, "globalliner")) == ("globalliner", "Global Liner Agencias")
    assert lines.leg_hours(E15, 3) == 165 / 17 and lines.departure(E15, 3) == 47 - 165 / 17
    assert lines.departure(E15, 3, 36) == 36 - 165 / 17
    elbspirit = PACK.get("dock-36A-w06x1-busy-0-e07")
    assert (elbspirit.ships[10].from_port, lines.leg_hours(elbspirit, 10), lines.departure(elbspirit, 10)) == (
        "Monrovia", None, None)
    assert Counter((s.from_port, lines.meta(t, s.id)["leg_nm"]) for t in PACK.tasks for s in t.ships
                   if s.from_port in ("Anvers", "Moerdijk", "Al 'Aqabah")) == {
        ("Anvers", 1893): 3, ("Moerdijk", 1893): 6, ("Al 'Aqabah", 1980): 1}
    assert Counter(s.from_port for t in PACK.tasks for s in t.ships if lines.meta(t, s.id)["leg_nm"] is None) == {
        "Colombo": 6, "Monrovia": 3, "Halifax": 2, "Ningbo": 2, "Montreal": 2, "Savannah": 1}
    biases = [m["bias"] for t in PACK.tasks for m in t.rules["lines"]["ships"]]
    assert max(map(abs, biases)) <= 4 and 0.75 <= sum(b > 0 for b in biases) / len(biases) <= 0.85


def test_the_published_schedule_is_the_lines_estimate_from_72_hours_out_not_the_truth():
    """Each ship arrives off its schedule by its agent's error from 72 hours out, but never so early that slowing from
    hour 0 couldn't still make the schedule, so 72 hours out its agent's ETA is the schedule. A delayed or bunched
    ship's whole story moves with it: it arrives after its news and later than it would have without it. An
    unscheduled call keeps v4's arrival, which its notice announces."""
    raised = 0
    for task in PACK.tasks:
        news = lines.first_known(task)
        for s, was in zip(task.ships, WIND.get(task.task_id).ships, strict=True):
            m = lines.meta(task, s.id)
            miss = round(lines.ETA_ERROR * m["bias"] * lines.ETA_HORIZON)
            assert (m["scheduled"], m["v4_arrival"]) == (was.planned_hour, was.arrival)
            if s.id not in news:
                assert (m["before_news"], s.arrival) == (None, reachable(task, s.id, max(1, was.planned_hour + miss),
                                                                          was.planned_hour))
                if s.arrival == was.planned_hour + miss:
                    assert lines.estimate(task, s.id, s.arrival - lines.ETA_HORIZON) == m["scheduled"]
                else:
                    raised += 1
            elif was.planned_hour is None:
                assert (m["before_news"], s.arrival) == (None, was.arrival)
            else:
                before = reachable(task, s.id, max(was.planned_hour + miss, news[s.id] + 2), was.planned_hour)
                assert (m["before_news"], s.arrival) == (before, max(was.arrival + miss, before + 1))
                assert news[s.id] < before < s.arrival
    assert raised == 25
    assert [(lines.scheduled(E15, s), lines.before_news(E15, s), E15.ships[s].arrival)
            for s in (1, 3, 5, 6, 9, 16)] == [
        (7, None, 5), (36, None, 47), (54, 63, 83), (66, 48, 54), (111, None, 97), (None, None, 69)]
    assert (lines.slowest(E15, 1, 0, 4), lines.slowest(E15, 1, 0, 5)) == (6, 8)
    assert (lines.slowest(E15, 9, 0, 96), lines.slowest(E15, 9, 0, 97)) == (110, 111)
    assert (lines.estimate(E15, 3, 47 - 72), lines.estimate(E15, 9, 97 - 72)) == (36, 111)


@pytest.mark.parametrize("task", WEEKS)
def test_an_eta_holds_from_72_hours_out_stays_between_the_next_hour_and_the_slowest_arrival_and_is_exact_from_the_call(
        task):
    for s in task.ships:
        errors = []
        far = s.arrival - round(lines.ETA_ERROR * lines.meta(task, s.id)["bias"] * lines.ETA_HORIZON)
        for hour in range(-120, s.arrival + 1):
            eta, slowest = lines.estimate(task, s.id, hour), lines.slowest(task, s.id, hour, s.arrival)
            assert eta == s.arrival if s.arrival - hour <= lines.CALL_LEAD else hour + 1 <= eta <= slowest
            if s.arrival - hour >= lines.ETA_HORIZON:
                assert eta == min(far, slowest)
            errors.append(abs(eta - s.arrival))
        assert errors == sorted(errors, reverse=True), s.id


def test_etas_are_mostly_early_by_a_share_of_the_hours_left():
    assert [lines.estimate(E15, 3, h) for h in (-25, 0, 23, 35, 45, 46, 47)] == [36, 40, 43, 45, 47, 47, 47]
    assert [lines.estimate(E15, 8, h) for h in (20, 44, 68, 80, 90, 91)] == [97, 96, 94, 93, 92, 92]
    assert lines.meta(E15, 8)["bias"] < 0 < lines.meta(E15, 3)["bias"]
    assert lines.estimate(E15, 3, 0, 36) == 31


@pytest.mark.parametrize("task", WEEKS)
def test_the_timetable_is_v4s_watches_each_with_the_reports_made_since_the_watch_before(task):
    v4, ours = schedule(lines.as_v4(task)), lines.timetable(task)
    hours = [w.hour for w in v4]
    assert [(w.index, w.hour, [n.event_id for n in w.notices]) for w in v4] == [
        (w.index, w.hour, [n.event_id for n in w.notices]) for w in schedule(WIND.get(task.task_id))]
    assert [(w.index, w.hour) for w in ours] == [(w.index, w.hour) for w in v4]
    for w, was in zip(ours, v4, strict=True):
        assert w.notices[:len(was.notices)] == was.notices
        assert {n.kind for n in w.notices[len(was.notices):]} <= set(lines.REPORTS)
        made = reports(w)
        assert made == sorted(made, key=lambda n: (n.hour, n.event_id))
        for n in made:
            e, s = n.event, task.ships[n.event["ship"]]
            planned = n.event_id.endswith("-plan")
            arrival = lines.before_news(task, s.id) if planned else s.arrival
            assert w.index == next(k for k, h in enumerate(hours) if h >= e["hour"])
            assert (e["type"], n.hour, n.event_hour, n.index) == (n.kind, e["hour"], arrival, -1)
            assert e["hour"] == max(arrival - e["lead"], 0)
            assert n.event_id == f"{n.kind}-{s.id}-{e['lead']}" + ("-plan" if planned else "")
            assert e["lead"] == lines.CALL_LEAD if n.kind == "call" else e["lead"] in lines.REPORT_LEADS
            assert not planned or n.kind == "eta"
            assert e["eta"] == (arrival if n.kind == "call" else lines.estimate(task, s.id, e["hour"], arrival))
            assert (n.name, n.text) == (lines.firm_name(task, lines.firm(task, s.id)), lines.report_text(task, e))


@pytest.mark.parametrize("task", WEEKS)
def test_every_ship_due_by_the_last_watch_calls_in_and_the_week_opens_with_the_latest_report_made_before_it(task):
    """A delayed or bunched ship's opening report is from the plan as it stood, when its news comes after hour 0."""
    made = [n for w in lines.timetable(task) for n in reports(w)]
    last = lines.timetable(task)[-1].hour
    assert sorted(n.event["ship"] for n in made if n.kind == "call") == [
        s.id for s in task.ships if s.arrival - lines.CALL_LEAD <= last]
    news = lines.first_known(task)
    for s in task.ships:
        opening = [n for n in made if n.event["ship"] == s.id and n.event_hour - n.event["lead"] <= 0]
        if s.id not in news:
            leads = [lead for lead in (*lines.REPORT_LEADS, lines.CALL_LEAD) if s.arrival - lead <= 0]
        elif (planned := lines.before_news(task, s.id)) is not None and news[s.id] > 0:
            leads = [lead for lead in lines.REPORT_LEADS if planned - lead <= 0]
        else:
            leads = []
        assert [n.event["lead"] for n in opening] == ([min(leads)] if leads else [])


@pytest.mark.parametrize("task", WEEKS)
def test_a_late_bunched_or_unscheduled_ship_is_reported_from_the_truth_only_after_its_news_and_from_the_plan_before(
        task):
    """first_known is the hour of the watch that brings a ship's news, and a report from the plan as it stood comes
    only at an earlier watch."""
    told, hours = news(lines.as_v4(task)), [w.hour for w in lines.timetable(task)]
    assert lines.first_known(task) == {s: hours[max(k for k, _ in notices)] for s, notices in told.items()}
    for w in lines.timetable(task):
        for n in reports(w):
            if (s := n.event["ship"]) in told:
                k = max(k for k, _ in told[s])
                if n.event_id.endswith("-plan"):
                    assert w.index < k and lines.before_news(task, s) is not None
                else:
                    assert n.hour > lines.first_known(task)[s] and w.index > k


def test_the_reports_of_one_week_watch_by_watch():
    watches = lines.timetable(E15)
    assert [(w.hour, [n.event_id for n in reports(w)]) for w in watches] == [
        (0, ["call-0-1", "eta-1-24", "eta-2-48", "eta-3-48", "eta-6-48-plan"]),
        (24, ["eta-2-24", "eta-4-48", "call-1-1", "eta-5-48-plan", "eta-3-24", "call-2-1", "eta-6-24-plan"]),
        (30, ["eta-4-24"]),
        (42, ["eta-5-48", "eta-7-48"]),
        (72, ["eta-8-48", "eta-16-24", "call-3-1", "call-4-1", "eta-9-48", "call-6-1", "eta-5-24", "eta-7-24",
              "call-16-1", "eta-8-24"]),
        (96, ["eta-9-24", "eta-10-48", "call-5-1", "eta-11-48", "call-7-1", "call-8-1", "call-9-1", "eta-12-48"])]
    assert news(lines.as_v4(E15)) == {5: [(2, 30)], 6: [(3, 42)], 16: [(3, 45)]}
    assert lines.first_known(E15) == {5: 30, 6: 42, 16: 42}
    opening = reports(watches[0])
    assert [(n.name, n.text) for n in (opening[0], opening[3])] == [
        ("A. Pérez y Cía", "PERSEUS (ship 0) calls in: at buoy S, arriving hour 1 (Mon 01:00)."),
        ("Global Liner Agencias", "STEEN (ship 3): ETA Tue 16:00 (hour 40), as of Mon 00:00.")]
    assert reports(watches[1])[4].text == "STEEN (ship 3): ETA Tue 19:00 (hour 43), as of Mon 23:00."
    assert (opening[4].event_hour, opening[4].event) == (
        48, {"type": "eta", "ship": 6, "hour": 0, "lead": 48, "eta": 60})
    assert opening[4].text == "ONE FREEDOM (ship 6): ETA Wed 12:00 (hour 60), as of Mon 00:00."
