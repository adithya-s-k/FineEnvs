"""The lines week (port_week_core.lines.LinesWeek): the arrivals as known (published, then each ETA its agent gives,
then the agreed or called-in arrival, never in the past), the agents' answers to the planner's asks and requests from
the plan as it stood until a delayed ship's news, their counter-offers, an agreement the news keeps or undoes, an agreed
ship's reports given as the agreement, check_plan's hours at anchor, and whole weeks played in process keeping their
audit."""

import json
from collections.abc import Iterator
from dataclasses import replace

import pytest
from berth_core import Task, plan_to_list
from lines_fixtures import require_lines

from port_week_core import lines, marine, prompts, wind, world
from port_week_core.lines import LinesWeek
from port_week_core.schedule import label
from port_week_core.wind import WindWeek

require_lines()

PACK = lines.pack()
E15 = PACK.get("dock-24B-w06x1-busy-0-e15")
WEEKS = [pytest.param(t, id=t.task_id) for t in PACK.tasks]


def walk(task: Task, policy: world.Policy | None = None) -> Iterator[LinesWeek]:
    """The week at each of its watches, its news delivered, confirming ``policy``'s plan before moving on."""
    week = LinesWeek(task)
    while True:
        yield week
        if policy:
            target = policy(week)
            if rows := [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]:
                week.confirm(rows)
        if week.last:
            return
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")


def arrival(week: LinesWeek, sid: int) -> int:
    return next(s.arrival for s in week.view.ships if s.id == sid)


def turn(week: LinesWeek, sid: int) -> int:
    return week.view.rules["lines"]["turns"][sid]


def row(week: LinesWeek, sid: int) -> str:
    return next(line for line in lines.section(week).splitlines() if line.startswith(f"- ship {sid} "))


def test_a_late_ship_shows_its_published_arrival_then_each_eta_it_is_given_then_its_arrival_once_it_calls_in():
    """NANTO's 48-hour report before its news is from the plan as it stood, arriving hour 63; from the news on, its
    ETAs are the truth's, hour 83, and no earlier than the notice's 79."""
    nanto, seen = E15.ships[5], []
    for week in walk(E15):
        seen.append((week.hour, arrival(week, 5), turn(week, 5), week.told.get(5), 5 in week.arrived))
    assert (nanto.planned_hour, lines.before_news(E15, 5), nanto.arrival) == (54, 63, 83)
    assert seen == [(0, 54, 54, None, False), (24, 57, 57, (15, 57), False), (30, 79, 79, (30, 79), False),
                    (42, 79, 79, (35, 79), False), (72, 80, 80, (59, 80), False), (96, 83, 83, (82, 83), True)]
    assert next(e for e in E15.disruptions if e.get("ship") == 5) == {"type": "late", "ship": 5, "hours": 20, "eta": 79}


@pytest.mark.parametrize("task", WEEKS)
def test_the_week_as_known_holds_the_agreed_or_called_in_arrival_else_the_latest_eta_or_the_schedule_never_past(task):
    for week in walk(task, world.naive):
        view = week.view
        assert [s.id for s in view.ships] == [s.id for s in wind.known(task, week.revealed).ships]
        for s in view.ships:
            settled = s.id in week.locks or s.id in week.arrived
            eta = week.told[s.id][1] if s.id in week.told else lines.scheduled(task, s.id)
            known = week.actual(s.id) if settled else max(eta, week.hour + 1)
            assert (s.arrival, view.rules["lines"]["turns"][s.id]) == (
                known, task.ships[s.id].arrival if settled else known)
        called = {week.scheduled[e["event_id"]].event["ship"] for e in week.log if e["kind"] == "call"}
        assert week.arrived == called


def test_a_ship_overdue_on_its_eta_reads_as_arriving_the_next_hour_until_it_calls_in():
    """NEXOE MAERSK's agent gave ETA hour 31 at hour 20; at hour 36 it hasn't called in, so it is due at 37 at the
    earliest. It gets in at 44."""
    e09, seen = PACK.get("dock-24B-w07x1-busy-0-e09"), []
    for week in walk(e09):
        if week.hour <= 48:
            seen.append((week.hour, week.told[1], arrival(week, 1), turn(week, 1), 1 in week.arrived))
    assert seen[2:] == [(30, (20, 31), 31, 31, False), (36, (20, 31), 37, 37, False), (42, (20, 31), 43, 43, False),
                        (48, (43, 44), 44, 44, True)]
    assert e09.ships[1].arrival == 44


def test_until_its_news_a_late_ships_true_arrival_never_reaches_the_planner():
    """The late ship arriving two hours later still: what the planner sees is byte for byte the same until the news,
    whose ETA carries the delay."""
    later = replace(E15, ships=[replace(s, arrival=85) if s.id == 5 else s for s in E15.ships])
    seen = []
    for task in (E15, later):
        out = []
        for week in walk(task, world.naive):
            out.append(json.dumps([week.situation(), week.check(plan_to_list(world.naive(week))),
                                   lines.step_lines(week), week.messages], ensure_ascii=False))
        seen.append(out)
    assert [w.hour for w in lines.timetable(E15)] == [0, 24, 30, 42, 72, 96]
    assert seen[0][:2] == seen[1][:2] and seen[0][2] != seen[1][2]


def test_get_situation_ends_with_the_agents_and_the_arrivals_as_known():
    week = LinesWeek(E15)
    situation = week.situation()["situation"]
    view = week.view
    assert situation.endswith("\n\n".join(["", marine.section(view), wind.section(view), lines.section(week)]))
    section = lines.section(week).splitlines()
    assert section[0] == "## Arrivals and the ships' agents" and section[8] == "- Arrivals as known now:"
    assert section[3] == ("- Until the port's news about a delayed ship, its agent works from the plan as it stood. "
                          "The news keeps an earlier agreement only if the ship can still make that hour; otherwise "
                          "the agreement lapses.")
    assert section[2] == (
        "- Message a ship's agent to ask for its ETA or to request an arrival time. Once the ship has sailed from its "
        "previous port or its agent's estimate is within 48 hours, the agent agrees when it can make the hour at 17 kn "
        "or by slowing to as little as 10 kn once it has sailed, and the ship then arrives at that hour; otherwise it "
        "answers with the earliest or the latest hour the ship can make. Before that, the agent can't commit to an "
        "hour and gives its ETA.")
    assert section[5] == (
        "- Ships berth first come, first served: no ship may start berthing while a ship earlier in the queue whose "
        "arrival you agreed waits at anchor, unless the wind holds that ship or the closures leave no stretch of quay "
        "long enough for it. A ship's turn is when it would arrive at cruising speed, so a ship slowed at your request "
        "keeps its turn.")
    rules = prompts.LINES_RULES.splitlines()
    assert section[3:6] == rules[rules.index("The ships' agents:") + 3:rules.index("The ships' agents:") + 6]
    assert section[4] == ("- A ship can't berth before it arrives. If it gets in after its window starts, the port "
                          "berths it at the first hour a free stretch of quay fits it, with the same cranes, around "
                          "every other ship's window and the rules, never ahead of an earlier ship whose arrival you "
                          "agreed still waiting; nothing else moves. A ship that gets in before its window waits at "
                          "anchor for it.")
    assert section[7] == (
        "- Agents, and their ships: A. Pérez y Cía (aperez): ships 0, 1, 12, 15; Boluda Lines (boluda): ships 7; "
        "CMA CGM Ibérica (cmacgm): ships 2, 11; COSCO Shipping Lines Spain (cosco): ships 4; Global Liner Agencias "
        "(globalliner): ships 3; Hapag-Lloyd Spain (hapag): ships 8, 9; Noatum Maritime (noatum): ships 5, 10, 13, "
        "14; Ocean Network Express Spain (one): ships 6.")
    assert [row(week, s) for s in (0, 3, 5, 6)] == [
        "- ship 0 PERSEUS: arrival hour 1, called in, arriving hour 1, turn 1; from Algeciras",
        "- ship 3 STEEN: arrival hour 40, ETA hour 40, as of Mon 00:00; from Valencia",
        "- ship 5 NANTO: arrival hour 54, published schedule; from Alicante",
        "- ship 6 ONE FREEDOM: arrival hour 60, ETA hour 60, as of Mon 00:00; from Damietta"]
    assert len(section) == 9 + len(view.ships) and "16" not in section[7]


def test_a_ship_that_has_called_in_but_not_yet_arrived_reads_as_arriving():
    assert [(w.hour, n.event["ship"]) for w in lines.timetable(E15) for n in w.notices
            if n.kind == "call" and n.event["eta"] > w.hour] == [(0, 0), (24, 2), (96, 9)]
    week = next(w for w in walk(E15) if w.hour == 24)
    ship = E15.ships[2]
    assert 2 in week.arrived and week.actual(2) == ship.arrival == 25
    assert row(week, 2) == ("- ship 2 CMA CGM DALILA: arrival hour 25, called in, arriving hour 25, turn 25; from "
                            "Fos-sur-Mer")
    assert week.status("cmacgm", 2) == {"ship": 2, "name": ship.name, "eta": 25, "as_of": 24}
    assert row(week, 0) == "- ship 0 PERSEUS: arrival hour 1, arrived hour 1, turn 1; from Algeciras"


def test_an_agent_agrees_an_arrival_the_ship_can_make_and_otherwise_offers_the_earliest_or_the_latest_it_can():
    """A counter-offer of the earliest hour is the ship's ETA as the planner knows it, unless the ship is agreed."""
    week = LinesWeek(E15)
    assert (week.reach(3, 0), week.told[3]) == ((47, 53), (0, 40))
    assert week.request("globalliner", 3, 46) == {"ship": 3, "name": "STEEN", "agreed": False,
                                                   "reason": "can't arrive that early", "earliest": 47}
    assert (week.told[3], arrival(week, 3)) == ((0, 47), 47)
    assert week.request("globalliner", 3, 54) == {"ship": 3, "name": "STEEN", "agreed": False,
                                                   "reason": "can't slow down that much", "latest": 53}
    assert week.locks == {} and week.told[3] == (0, 47)
    assert week.request("globalliner", 3, 53) == {"ship": 3, "name": "STEEN", "agreed": True, "arrival": 53,
                                                   "turn": 47, "slowing": True}
    assert week.request("globalliner", 3, 47)["slowing"] is False and week.actual(3) == 47
    assert week.request("globalliner", 3, 53)["agreed"]
    assert week.locks[3] == pytest.approx((47 - 165 / 17, 165 / 17, 53))
    truth = week.truth()
    assert (truth.ships[3].arrival, truth.rules["lines"]["turns"][3], E15.ships[3].arrival) == (53, 47, 47)
    assert (arrival(week, 3), turn(week, 3), week.told[3]) == (53, 47, (0, 53))
    assert row(week, 3) == "- ship 3 STEEN: arrival hour 53, agreed hour 53, turn 47; from Valencia"
    assert week.request("globalliner", 3, 60) == {"ship": 3, "name": "STEEN", "agreed": False,
                                                   "reason": "can't slow down that much", "latest": 53,
                                                   "agreed_arrival": 53}
    assert week.request("globalliner", 3, 46)["earliest"] == 47 and week.told[3] == (0, 53)
    assert week.status("globalliner", 3) == {"ship": 3, "name": "STEEN", "eta": 53, "agreed": True, "turn": 47}


def test_a_ship_still_in_its_previous_port_can_slow_only_by_seven_tenths_of_its_leg():
    """STEEN sails the 165 nm from Valencia in under ten hours: until it leaves, the most it can give is 0.7 of that
    leg; once at sea, 0.7 of the hours left. A ship from beyond the week's reach can stretch all its hours."""
    week, leg = LinesWeek(E15), 165 / 17
    assert lines.departure(E15, 3) == 47 - leg > 37
    assert week.reach(3, 0) == week.reach(3, 37) == (47, int(47 + 0.7 * leg)) == (47, 53)
    assert week.reach(3, 41) == (47, int(41 + 6 * 1.7)) == (47, 51)
    assert week.reach(3, 46) == (47, 47)
    assert week.remaining(3, 0) == pytest.approx((47 - leg, leg)) and week.remaining(3, 41) == (41, 6)
    assert lines.departure(E15, 6) < 0 and week.reach(6, 0, 54) == (54, int(54 * 1.7)) == (54, 91)
    elbspirit = PACK.get("dock-36A-w06x1-busy-0-e07")
    assert LinesWeek(elbspirit).reach(10, 0) == (97, 164)


def test_a_slowed_ship_can_change_pace_again_only_within_what_it_has_left():
    week = LinesWeek(E15)
    week.request("globalliner", 3, 53)
    left = (165 / 17) * (53 - 41) / (53 - (47 - 165 / 17))
    assert week.remaining(3, 41) == pytest.approx((41, left))
    assert week.reach(3, 41) == (49, int(41 + left * 1.7)) == (49, 53)


def test_an_agent_answers_only_for_its_own_ships_that_the_week_shows():
    week = LinesWeek(E15)
    for args, message in [(("aperez", 3), "ship 3 STEEN is not this agent's: Global Liner Agencias handles it"),
                          (("hapag", 16), "no ship 16 in the week"), (("globalliner", "3"), "no ship '3' in the week"),
                          (("aperez", True), "no ship True in the week"), (("aperez", 99), "no ship 99 in the week")]:
        for ask in (week.status, lambda firm, sid: week.request(firm, sid, 40)):
            with pytest.raises(ValueError, match=f"^{message}$"):
                ask(*args)
    for bad in (40.0, "40", True):
        with pytest.raises(ValueError, match="^request_arrival must be an hour"):
            week.request("globalliner", 3, bad)
    assert week.exchanges == [] and week.locks == {}


def test_status_records_the_eta_it_tells_and_an_arrived_ship_reports_its_arrival():
    """VERONICA B is 89 hours out, so its agent's ETA is the published schedule."""
    week = LinesWeek(E15)
    assert (arrival(week, 7), row(week, 7)) == (84, "- ship 7 VERONICA B: arrival hour 84, published schedule; from "
                                                     "Alicante")
    assert week.status("boluda", 7) == {"ship": 7, "name": "VERONICA B", "eta": 84, "as_of": 0}
    assert (week.told[7], arrival(week, 7), turn(week, 7)) == ((0, 84), 84, 84)
    assert row(week, 7) == "- ship 7 VERONICA B: arrival hour 84, ETA hour 84, as of Mon 00:00; from Alicante"
    assert week.exchanges == [{"watch": 0, "hour": 0, "firm": "boluda", "ship": 7, "ask": "eta", "call": 0,
                               "reply": {"ship": 7, "name": "VERONICA B", "eta": 84, "as_of": 0}}]
    week = next(w for w in walk(E15) if w.hour == 24)
    assert week.status("aperez", 0) == {"ship": 0, "name": "PERSEUS", "arrived": 1, "turn": 1}
    assert week.request("aperez", 0, 30) == {"ship": 0, "name": "PERSEUS", "agreed": False,
                                             "reason": "already arrived at hour 1"}
    assert week.exchanges == [
        {"watch": 1, "hour": 24, "firm": "aperez", "ship": 0, "ask": "eta", "call": 0,
         "reply": {"ship": 0, "name": "PERSEUS", "arrived": 1, "turn": 1}},
        {"watch": 1, "hour": 24, "firm": "aperez", "ship": 0, "ask": "request_arrival", "arrival": 30, "call": 0,
         "reply": {"ship": 0, "name": "PERSEUS", "agreed": False, "reason": "already arrived at hour 1"}}]


def test_the_latest_eta_stands_whether_the_planner_asked_for_it_or_the_agent_reported_it():
    week = next(w for w in walk(E15) if w.hour == 42)
    assert week.status("boluda", 7) == {"ship": 7, "name": "VERONICA B", "eta": 85, "as_of": 42}
    week.open()
    report = next(n for n in week.watches[week.watch].notices if n.event_id == "eta-7-24")
    for n in week.watches[week.watch].notices:
        week.notice(n.event_id, n.name, n.text, "trigger")
    assert (report.event["hour"], report.event["eta"], week.told[7]) == (65, 87, (65, 87))
    assert week.status("boluda", 7)["eta"] == 88 and (week.told[7], arrival(week, 7)) == ((72, 88), 88)


@pytest.mark.parametrize("task", WEEKS)
def test_a_whole_week_played_keeps_its_audit_and_reports_its_lines(task):
    week = world.play(task, world.naive, week=LinesWeek)
    assert week.audit() == {"ok": True, "problems": []}
    data = week.data()
    assert data["lines"]["exchanges"] == [] and data["lines"]["agreed"] == []
    assert data["lines"]["arrivals"] == [{"ship": s.id, "arrival": s.arrival, "turn": s.arrival} for s in task.ships]
    assert data["audit"] == week.audit() and data["grade"] == week.grade


def test_the_audit_holds_no_report_to_a_lead_but_every_other_notice_still_to_its_own():
    class Timed(LinesWeek):
        untimed = ()

    problems = world.play(E15, world.naive, week=Timed).audit()["problems"]
    assert problems[:2] == ["eta-2-24: revealed 1 h ahead at hour 24", "call-1-1: revealed -19 h ahead at hour 24"]
    assert {p.split("-")[0] for p in problems} == set(lines.REPORTS)
    assert world.play(E15, world.naive, week=LinesWeek).audit() == {"ok": True, "problems": []}


def test_agreeing_an_arrival_then_playing_on_keeps_the_audit_and_the_agreement():
    """NANTO's agent, 55 hours out with the ship still in Alicante, can't commit to an hour yet."""
    def policy(week: LinesWeek):
        if week.watch == 0:
            week.request("globalliner", 3, 53)
            week.request("noatum", 5, 90)
        return world.naive(week)

    week = world.play(E15, policy, week=LinesWeek)
    assert week.audit() == {"ok": True, "problems": []}
    assert week.plan[3][0] >= 53 and week.truth().ships[3].arrival == 53
    data = week.data()["lines"]
    assert data["agreed"] == [{"ship": 3, "arrival": 53, "turn": 47}]
    assert [(e["watch"], e["firm"], e["ship"], e["arrival"], e["reply"]["agreed"]) for e in data["exchanges"]] == [
        (0, "globalliner", 3, 53, True), (0, "noatum", 5, 90, False)]
    assert data["exchanges"][1]["reply"] == {
        "ship": 5, "name": "NANTO", "agreed": False, "eta": 55,
        "reason": "can't commit yet: it hasn't sailed from Alicante and is more than 48 hours out"}
    assert data["arrivals"][3] == {"ship": 3, "arrival": 53, "turn": 47}
    assert week.truth().rules["lines"]["agreed"] == week.view.rules["lines"]["agreed"] == week.past(
        week.revealed, week.truth()).rules["lines"]["agreed"] == [3]
    assert [a for a in data["arrivals"] if a["ship"] != 3] == [
        {"ship": s.id, "arrival": s.arrival, "turn": s.arrival} for s in E15.ships if s.id != 3]


def agreeing(*requests: tuple[int, str, int, int]) -> world.Policy:
    """Naive, after agreeing ``requests`` (watch, firm, ship, arrival)."""
    def policy(week: LinesWeek):
        for watch, firm, sid, at in requests:
            if week.watch == watch:
                assert week.request(firm, sid, at)["agreed"]
        return world.naive(week)
    return policy


def test_an_agreed_ships_reports_reach_the_planner_as_the_agreement_until_its_agreed_hour():
    """STEEN's 24-hour report comes as the agreement; its call, made at hour 52, would reach the planner at 72, after
    the agreed hour 53, so it isn't delivered. The log keeps the scheduled notices, so the audit holds, with the text
    each delivered, None for none; a call no longer marks an agreed ship arrived."""
    week = world.play(E15, agreeing((0, "globalliner", 3, 53)), week=LinesWeek)
    steen = "STEEN (ship 3): arriving hour 53 (Wed 05:00), as agreed; turn hour 47."
    assert [(m["hour"], m["from"], m["text"]) for m in week.messages if "(ship 3)" in m["text"]] == [
        (24, "Global Liner Agencias", steen)]
    rewritten = [(e["event_id"], e["watch"], e["text"]) for e in week.log if e["kind"] in lines.REPORTS
                 and week.scheduled[e["event_id"]].event["ship"] == 3 and e["via"] != "load"]
    assert rewritten == [("eta-3-24", 1, steen), ("call-3-1", 4, None)]
    assert all(e["matches"] for e in week.log) and week.audit() == {"ok": True, "problems": []}
    assert 3 not in week.arrived and week.told[3] == (0, 53)
    assert row(week, 3) == "- ship 3 STEEN: arrival hour 53, agreed hour 53, turn 47; from Valencia"


def test_a_report_reaching_the_planner_at_the_agreed_hour_is_stale_too():
    """ONE FREEDOM, agreed at hour 0 for 72, has its call, timed by its unslowed arrival, reach the planner at 72, the
    agreed hour itself, so it isn't delivered."""
    week = world.play(E15, agreeing((0, "one", 6, 72)), week=LinesWeek)
    assert [(e["watch"], e["hour"], e["event_id"], e["text"]) for e in week.log if e["kind"] in lines.REPORTS
            and week.scheduled[e["event_id"]].event["ship"] == 6 and e["via"] != "load"] == [
        (1, 24, "eta-6-24-plan", f"ONE FREEDOM (ship 6): arriving hour 72 ({label(72)}), as agreed; turn hour 48."),
        (4, 72, "call-6-1", None)]
    assert week.locks[6][2] == 72 and week.audit() == {"ok": True, "problems": []}


def test_a_message_repeated_in_one_bulletin_is_dropped_and_one_repeated_in_a_later_bulletin_kept():
    """CMA CGM DALILA, agreed for 30, has its 24-hour report and its call in the bulletin at 24, each as the agreement:
    the bulletin carries it once, and the log has no text for the second. VERONICA B, agreed at 42 for 98, has its
    24-hour report at 72 and its call at 96 as the agreement, both delivered."""
    week = world.play(E15, agreeing((0, "cmacgm", 2, 30), (3, "boluda", 7, 98)), week=LinesWeek)

    def about(sid: int) -> list[tuple]:
        return [(e["watch"], e["event_id"], e["text"]) for e in week.log if e["kind"] in lines.REPORTS
                and week.scheduled[e["event_id"]].event["ship"] == sid and e["via"] != "load"]

    assert about(2) == [(1, "eta-2-24", week.agreement(2)), (1, "call-2-1", None)]
    assert about(7) == [(3, "eta-7-48", week.scheduled["eta-7-48"].text), (4, "eta-7-24", week.agreement(7)),
                        (5, "call-7-1", week.agreement(7))]
    assert [m["hour"] for m in week.messages if m["text"] in (week.agreement(2), week.agreement(7))] == [24, 72, 96]


def test_an_earliest_offer_floors_the_ships_etas_until_its_news_which_floors_them_at_its_own_eta():
    """At hour 0 STEEN's agent offers hour 47 at the earliest and ONE FREEDOM's, from the plan as it stood, 48: STEEN's
    next report, 43, reaches the planner as 47, and so does its ETA asked at 24. The news floors a delayed ship's ETAs
    at the notice's ETA, or its true arrival if earlier: NANTO's at 79, so its 48-hour report, 77, reaches the planner
    as 79; ONE FREEDOM's at 54, its true arrival, below the notice's 56. An unscheduled call's notice sets no floor,
    and an earliest offer about an agreed ship sets none."""
    def asking(week: LinesWeek):
        if week.watch == 0:
            week.request("globalliner", 3, 46)
            week.request("one", 6, 47)
        return {}

    weeks = walk(E15, asking)
    week = next(w for w in weeks if w.hour == 24)
    assert (week.floor, week.told[3]) == ({3: 47, 6: 48}, (23, 47))
    [(eta, text)] = [(week.scheduled[e["event_id"]].event["eta"], e["text"]) for e in week.log
                     if e["event_id"] == "eta-3-24"]
    assert (eta, text) == (43, "STEEN (ship 3): ETA Tue 23:00 (hour 47), as of Mon 23:00.")
    assert text in [m["text"] for m in week.messages] and all(e["matches"] for e in week.log)
    assert lines.estimate(E15, 3, 24) == 44 and week.status("globalliner", 3)["eta"] == 47
    week = next(w for w in weeks if w.hour == 42)
    assert (week.floor, week.told[5], week.told[6]) == ({3: 47, 5: 79, 6: 54}, (35, 79), (42, 56))
    assert ([e["text"] for e in week.log if e["event_id"] == "eta-5-48"], week.scheduled["eta-5-48"].event["eta"]) == (
        ["NANTO (ship 5): ETA Thu 07:00 (hour 79), as of Tue 11:00."], 77)
    assert 16 in week.told and 16 not in week.floor
    agreed = LinesWeek(E15)
    agreed.request("globalliner", 3, 53)
    assert agreed.request("globalliner", 3, 46)["earliest"] == 47 and agreed.floor == {}


def test_an_agreement_made_before_a_ships_news_stands_if_it_can_still_make_it_and_otherwise_lapses():
    """ONE FREEDOM's agent, working from the plan as it stood (hour 48), agrees hour 70 at hour 0, and NANTO's (hour
    63) agrees 70 at 24. NANTO's news at hour 30 puts it in at 83 at the earliest: the agreement lapses and the news
    gives that hour, where the ship, not yet sailed, cruises in. ONE FREEDOM's at 42 puts it in at 54 had it
    cruised; it left Damietta when the truth has it leave and has sailed at the pace hour 70 set, so from 42 it can
    arrive between 68 and 84: the agreement stands, with the turn the truth gives, and the news gives the agreed
    hour."""
    seen = []
    for week in walk(E15, agreeing((0, "one", 6, 70), (1, "noatum", 5, 70))):
        seen.append((week.hour, sorted(week.locks), week.told.get(5), week.told.get(6), week.turn(6)))
        if week.hour == 42:
            assert week.locks[6] == pytest.approx((42, 54 - 48 / 70 * 42, 70)) and week.reach(6, 42) == (68, 84)
    assert seen[1:4] == [(24, [6], (15, 57), (0, 70), 48), (30, [6], (30, 83), (0, 70), 48),
                         (42, [6], (35, 83), (0, 70), 54)]
    news = {e["event_id"]: (e["watch"], e["text"]) for e in week.log if e["kind"] == "late"}
    assert news == {
        "late-3": (2, E15.notices[3].replace("ETA hour 79", "ETA hour 83")
                   + " NANTO (ship 5) can no longer arrive at hour 70: the earlier agreement no longer holds."),
        "late-4": (3, E15.notices[4].replace("ETA hour 56", "ETA hour 70")
                   + f" The earlier agreement stands: ONE FREEDOM (ship 6): arriving hour 70 ({label(70)}), as agreed; "
                     "turn hour 54.")}
    assert [m["text"] for m in week.messages if m["from"] in ("NANTO", "ONE FREEDOM")] == [t for _, t in news.values()]
    assert week.data()["lines"]["agreed"] == [{"ship": 6, "arrival": 70, "turn": 54}]
    assert (week.truth().ships[5].arrival, week.audit()) == (83, {"ok": True, "problems": []})


def through(week: LinesWeek, hour: int) -> None:
    """The week's watches, with their notices delivered, up to ``hour``."""
    while week.hour < hour:
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")


def test_an_agreement_made_before_the_news_lapses_if_its_pace_since_leaves_the_ship_short_of_it():
    """ONE FREEDOM agreed for 64 at hour 0 has sailed at 48/64 of cruising speed since; from 42 it can't arrive before
    65, so the agreement lapses, and having slowed, it cruises in at 65; the news gives its agent's own estimate of
    that, 71, as its later ETAs do, and the agent now agrees from 65. For 65 or 81 it can still make it."""
    for at, stands in ((64, False), (65, True), (81, True)):
        week = LinesWeek(E15)
        assert week.request("one", 6, at)["agreed"]
        through(week, 42)
        assert (6 in week.locks) is stands, at
    assert (week.actual(6), week.turn(6)) == (81, 54)
    week = LinesWeek(E15)
    week.request("one", 6, 64)
    through(week, 42)
    assert (week.actual(6), week.turn(6), week.told[6], week.floor[6], week.reach(6, 42)) == (65, 54, (42, 71), 65,
                                                                                              (65, 81))
    assert week.status("one", 6)["eta"] == 71
    [news] = [m["text"] for m in week.messages if m["from"] == "ONE FREEDOM"]
    assert "its agent now gives ETA hour 71 " in news and news.endswith("can no longer arrive at hour 64: the earlier "
                                                                        "agreement no longer holds.")
    assert week.request("one", 6, 66)["agreed"] and week.truth().ships[6].arrival == 66


def test_at_its_news_a_ship_has_slowed_for_every_agreement_since_the_first():
    """ONE FREEDOM agreed for 60 at hour 0, for 70 and 71 at 24 and 70 again at 30 has sailed slower since hour 0, not
    only since 30: from 42 it can arrive between 66 and 82, and 70 stands. Agreeing 70 again changes nothing; agreed
    for 56 only at 30, it slowed for 12 hours and cruises in at 58."""
    def played(*agreements: tuple[int, int]) -> LinesWeek:
        week = LinesWeek(E15)
        for hour, at in agreements:
            through(week, hour)
            assert week.request("one", 6, at)["agreed"]
        through(week, 42)
        return week
    week = played((0, 60), (24, 70), (24, 71), (30, 70))
    assert (week.locks[6][2], week.reach(6, 42)) == (70, (66, 82))
    assert played((0, 70), (30, 70)).reach(6, 42) == played((0, 70)).reach(6, 42) == (68, 84)
    week = played((30, 56))
    assert 6 not in week.locks and (week.actual(6), week.reach(6, 42)) == (58, (58, 69))


def test_advance_leaves_where_the_ships_are_to_the_bulletin():
    week = LinesWeek(E15)
    assert set(week.open()) == {"watch", "hour", "time", "frozen_before", "note"}


def test_other_ships_reports_keep_their_scheduled_text():
    agreed = world.play(E15, agreeing((0, "globalliner", 3, 53)), week=LinesWeek)
    plain = world.play(E15, world.naive, week=LinesWeek)
    assert [m for m in agreed.messages if "(ship 3)" not in m["text"]] == [
        m for m in plain.messages if "(ship 3)" not in m["text"]]
    assert [m["text"] for m in plain.messages if m["text"].startswith("STEEN (ship 3)")] == [
        n.text for n in plain.scheduled.values() if n.event_id in ("eta-3-24", "call-3-1")]


def test_a_bunching_notice_with_one_ship_agreed_still_gives_the_others_news():
    """MARTHA A's agent can commit at hour 24, its ETA within 48 hours, and agrees the hour it would have arrived
    without the Red Sea news, which comes at 30."""
    task = PACK.get("dock-24B-w35x1-busy-0-e00")
    bunching = next(e for e in task.disruptions if e["type"] == "bunching")
    first, *others = bunching["ships"]
    week = world.play(task, agreeing((1, lines.firm(task, first), first, lines.before_news(task, first))),
                      week=LinesWeek)
    [message] = [m for m in week.messages if m["from"] == "Ship agents"]
    assert all(f"ship {s}" in message["text"] for s in others), message
    assert message["text"].endswith(f"{task.ships[first].name} (ship {first}) can no longer arrive at hour "
                                    f"{lines.before_news(task, first)}: the earlier agreement no longer holds.")
    assert first not in week.locks and lines.first_known(task)[first] == 30 == week.watches[2].hour


def test_before_the_news_of_a_delay_its_agent_answers_from_the_plan_as_it_stood():
    """NANTO's agent gives its ETA, its counter-offers and its turn from hour 63, when it would have arrived without
    its delay, until the notice at hour 30; from then on, from the truth, hour 83. At hour 0 it is 55 hours out and
    hasn't sailed, so it can't commit to an hour; by 24 it can, and offers 63. The notice at 30 gives that offer moved
    by the 20-hour delay, 83; its own estimate, 76, is within 48 hours, so it offers 83."""
    assert (lines.first_known(E15)[5], lines.before_news(E15, 5), E15.ships[5].arrival) == (30, 63, 83)
    for week in walk(E15):
        if week.hour == 0:
            assert (arrival(week, 5), week.pending(5), week.basis(5), week.turn(5)) == (54, True, 63, 63)
            assert (week.pending(3), week.basis(3)) == (False, 47)
            assert week.status("noatum", 5) == {"ship": 5, "name": "NANTO", "eta": 55, "as_of": 0}
            assert week.request("noatum", 5, 54) == {
                "ship": 5, "name": "NANTO", "agreed": False, "eta": 55,
                "reason": "can't commit yet: it hasn't sailed from Alicante and is more than 48 hours out"}
            assert (week.told[5], week.floor) == ((0, 55), {})
        elif week.hour == 24:
            assert week.request("noatum", 5, 54) == {"ship": 5, "name": "NANTO", "agreed": False,
                                                     "reason": "can't arrive that early", "earliest": 63}
            assert week.request("noatum", 5, 73)["latest"] == 72
            assert (week.told[5], arrival(week, 5), week.floor) == ((24, 63), 63, {5: 63})
        elif week.hour == 30:
            assert (week.pending(5), week.basis(5), week.turn(5), week.reach(5, 30)) == (False, 83, 83, (83, 92))
            assert (week.told[5], arrival(week, 5), week.floor) == ((30, 83), 83, {5: 83})
            assert week.status("noatum", 5) == {"ship": 5, "name": "NANTO", "eta": 83, "as_of": 30}
            assert week.request("noatum", 5, 63) == {"ship": 5, "name": "NANTO", "agreed": False,
                                                     "reason": "can't arrive that early", "earliest": 83}
            assert week.request("noatum", 5, 83)["agreed"]
            break


def test_an_agent_commits_once_its_ship_has_sailed_or_is_within_48_hours():
    """CAPE CITIUS, due at 92 from Genoa, 21 hours away: at 42 its ETA is 96, 54 hours out; at 47, 95, 48 hours out;
    it sails at 71. SEASPAN OCEANIA's ETA at 0 is 49, an hour too far; ONE FREEDOM left Damietta before the week.
    STEEN's agent agrees the hour it offers, whatever was asked first."""
    assert (lines.estimate(E15, 8, 42), lines.estimate(E15, 8, 47), lines.departure(E15, 8)) == (96, 95, 92 - 356 / 17)
    assert [lines.commits(E15, 8, h, 92) for h in (42, 46, 47)] == [False, False, True]
    assert (lines.estimate(E15, 4, 0), lines.commits(E15, 4, 0, 49), lines.commits(E15, 4, 1, 49)) == (49, False, True)
    assert lines.departure(E15, 6, 48) < 0 and lines.commits(E15, 6, 0, 48)
    steen = LinesWeek(E15)
    assert steen.request("globalliner", 3, 30)["earliest"] == 47 and steen.request("globalliner", 3, 47)["agreed"]
    week = next(w for w in walk(E15) if w.hour == 42)
    assert week.request("hapag", 8, 100) == {
        "ship": 8, "name": "CAPE CITIUS", "agreed": False, "eta": 96,
        "reason": "can't commit yet: it hasn't sailed from Genoa and is more than 48 hours out"}
    assert week.told[8] == (42, 96) and week.locks == {}


def test_check_plan_shows_the_hours_at_anchor_after_the_arrival_as_known_in_the_lines_week_only():
    """STEEN's agent gives ETA hour 40: berthing at 45 it waits 5 hours at anchor, 4 sections each, on top of the 9
    hours past its slot the wind week charges."""
    plan = plan_to_list({3: (45, 3, 1)})
    [ours] = [r for r in LinesWeek(E15).check(plan)["ships"] if r["ship"] == 3]
    [theirs] = [r for r in WindWeek(wind.pack().get(E15.task_id)).check(plan)["ships"] if r["ship"] == 3]
    assert list(ours.items()) == [("ship", 3), ("berth_hour", 45), ("section", 3), ("cranes", 1), ("departure", 55),
                                  ("delay_h", 9), ("anchor_h", 5), ("moved", False), ("cost", 56), ("problems", [])]
    assert theirs == {k: v for k, v in ours.items() if k != "anchor_h"} | {"cost": 36}


def test_the_notices_list_adds_the_agreement_to_an_agreed_late_ships_delay_notice():
    """The situation's notices keep the delay text and add the agreed arrival after it."""
    week = LinesWeek(E15)
    [late] = [i for i, e in enumerate(E15.disruptions) if e["type"] == "late" and e["ship"] == 6]
    reach = week.reach(6, week.hour)
    assert week.request(lines.firm(E15, 6), 6, reach[0] + 2)["agreed"]
    week.revealed.append(f"late-{late}")
    [text] = [t for e, t in zip(week.view.disruptions, week.view.notices, strict=True)
              if e["type"] == "late" and e["ship"] == 6]
    assert text == E15.notices[late] + " Since then: " + week.agreement(6)


def test_confirm_berths_reports_the_confirmed_plans_problems_and_cost_as_check_plan_does():
    """A clash confirmed at watch 0 is reported with the confirmation, as check_plan reports it, for one planning call;
    v4's confirm reports only what it confirmed."""
    plan = [{"ship": 3, "berth_hour": 47, "section": 4, "cranes": 1}, {"ship": 8, "berth_hour": 47, "section": 4}]
    week = LinesWeek(E15)
    checked = week.check(plan)
    confirmed = week.confirm(plan)
    assert list(confirmed) == ["confirmed", "unchanged", "refused", "entry_problems", "feasible", "cost", "delay_cost",
                               "moves", "excused_cost", "ships", "planning_calls_left"]
    assert {k: confirmed[k] for k in checked if k != "planning_calls_left"} == {
        k: v for k, v in checked.items() if k != "planning_calls_left"}
    assert (confirmed["feasible"], confirmed["confirmed"], confirmed["planning_calls_left"]) == (False, [3, 8], 1)
    v4 = WindWeek(wind.pack().get(E15.task_id))
    assert set(v4.confirm(plan)) == {"confirmed", "unchanged", "refused", "entry_problems", "planning_calls_left"}
