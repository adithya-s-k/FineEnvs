"""OccDay by hand on a small storm day: the turn machine and its refusals, the requests and their log, the freeze, the
advisories and the audit, absences, a crew move released in its turn or at the next watch leaving the day as without
it, the projection's history and crew flags, the scripted desks against programs/occ.py's talk-idle mode, and a recorded
day replayed byte for byte."""

from dataclasses import replace

import pytest
from occ_helpers import case, play, scripted, storm

from airops_core.core.sim import CANCELLED, FLOWN
from airops_core.occ import run
from airops_core.occ.day import OccDay, Refused, reward
from airops_core.occ.replay import replay
from airops_core.programs import crewplan, crews, crewsim, occ

END = {"note": None, "stand_down": False}


@pytest.fixture(scope="module")
def world():
    return storm()


def test_the_desks_take_turns_crew_first_and_only_on_their_turn(world):
    d = OccDay(world)
    assert (d.turn["desk"], d.turn["kind"], d.t) == ("crew", "crew", 0)
    assert d.call("ops", "wait_turn", {}) == {"waiting_for": "the crew desk's crew turn at watch 0 (00:00)"}
    assert "not your turn" in d.call("ops", "ops_flights", {"ids": None, "station": None, "status": None,
                                                           "start": None, "end": None})["error"]
    assert "not your turn" in d.call("crew", "crew_reserves", {"base": None})["error"]
    with pytest.raises(Refused, match="not a tool of the crew desk"):
        d.call("crew", "ops_confirm", {"changes": []})
    with pytest.raises(Refused, match="no desk for role 'both'"):
        d.call("both", "wait_turn", {})
    page = d.call("crew", "wait_turn", {})
    assert page["turn"] == "crew" and page["page"].startswith("# Crew scheduler, JetBlue, 2024-01-01, 00:00")
    assert d.call("crew", "wait_turn", {}) == page
    assert d.call("crew", "end_turn", END) == {"ended": "crew", "watch": 0, "opened": None, "done": False,
                                               "next": "the ops desk's ops turn at watch 0 (00:00)"}
    assert d.call("ops", "wait_turn", {})["page"].startswith("# Ops controller")
    out = d.call("ops", "end_turn", END)
    assert out["opened"] == 1 and d.t == 60 and d.turn["desk"] == "crew"
    assert len(d.refusals) == 2 and all(not r["mutating"] for r in d.refusals)


def test_an_idle_day_is_the_crew_systems_and_passes_its_audit(world):
    d = play(world)
    assert d.grade["cost"] == world.refs["idle"] and d.grade["end_reason"] == "done"
    assert d.audit() == {"ok": True, "problems": {}, "refused_off_turn": 0, "acausal_refused": 0}
    assert {t["kind"] for t in d.turns} == {"crew", "ops"} and d.grade["reward"] == round(
        reward(d.grade["cost"], world.refs["best"], world.refs["B0"]["notice"]), 6)
    assert [n["via"] for n in d.notices] == ["load", "trigger"] and d.notices[1]["watch"] == 2


def test_a_cancel_request_ops_accepts_cancels_now_and_changes_nothing_before(world):
    d = OccDay(world)
    page = d.call("crew", "wait_turn", {})["page"]
    assert "now f2 102 PWM-DCA 10:00 pilot: CANCEL: no crew can reach it by its deadline, at 00:00" in page
    assert d.call("crew", "crew_request", {"type": "CANCEL", "flight": "f2", "time": None, "text": None}) == {
        "request": "R1", "to": "ops", "type": "CANCEL"}
    d.call("crew", "crew_request", {"type": "TEXT", "flight": None, "time": None, "text": "f0 is late"})
    d.call("crew", "end_turn", END)
    page = d.call("ops", "wait_turn", {})["page"]
    assert "R1 from crew at 00:00: CANCEL f2" in page and "R2 text from crew: f0 is late" in page
    before = run.history(d.truth, d.t)
    out = d.call("ops", "ops_reply", {"request": "R1", "answer": "accept", "text": None})
    assert out["answer"] == "accept" and out["delta"]["notice"] < 0 and out["delta"]["total"] < 0
    assert d.truth.run.flights[2].status == CANCELLED and d.truth.run.flights[2].decided == 0
    assert run.history(d.truth, d.t) == before and d.changes[-1] | {"seq": 0} == {
        "seq": 0, "turn": 1, "desk": "ops", "watch": 0, "t": 0, "target": 2, "action": "cancel", "detail": None,
        "ref": "R1", "via": "accept"}
    r = d.requests[0]
    assert (r["sender"], r["receiver"], r["type"], r["params"], r["sent"], r["status"], r["replied"]) == (
        "crew", "ops", "CANCEL", {"flight": 2}, 0, "accepted", 0)
    assert r["actions"] == [{"desk": "ops", "action": "cancel", "target": 2, "minute": 0, "link": "accept"}]
    assert "is cancelled" in d.call("ops", "ops_confirm", {"changes": [
        {"flight": "f2", "action": "hold", "time": "11:00", "other": None, "ref": None}]})["refused"][0]["reason"]


def test_a_request_to_crew_gives_it_a_reply_turn_and_a_newer_request_replaces_an_older(world):
    d = OccDay(world)
    d.call("crew", "wait_turn", {})
    d.call("crew", "end_turn", END)
    d.call("ops", "wait_turn", {})
    d.call("ops", "ops_request", {"type": "COVER", "flight": "f3", "time": "09:00", "text": None})
    d.call("ops", "ops_request", {"type": "COVER", "flight": "f3", "time": "10:00", "text": None})
    assert d.call("ops", "end_turn", END)["next"] == "the crew desk's reply turn at watch 0 (00:00)"
    page = d.call("crew", "wait_turn", {})
    assert page["turn"] == "reply" and "R2 from ops at 00:00: COVER f3 by 10:00" in page["page"]
    assert [r["status"] for r in d.requests] == ["superseded", "open"]
    assert d.call("crew", "crew_reply", {"request": "R2", "answer": "accept", "text": None}) == {
        "request": "R2", "answer": "accept"}
    assert d.call("crew", "end_turn", END)["opened"] == 1


def test_ops_holds_no_earlier_than_the_freeze_line_and_never_on_a_committed_flight(world):
    d = OccDay(world)
    for desk in ("crew", "ops"):
        d.call(desk, "wait_turn", {})
        d.call(desk, "end_turn", END)
    d.call("crew", "wait_turn", {})
    d.call("crew", "end_turn", END)
    d.call("ops", "wait_turn", {})
    hold = {"flight": "f3", "action": "hold", "time": "01:30", "other": None, "ref": None}
    assert "no hold before the freeze line, 02:00" in d.call("ops", "ops_check", {"changes": [hold]})["refused"][0][
        "reason"]
    d.committed.add(3)
    assert "is committed" in d.call("ops", "ops_check", {"changes": [hold | {"time": "03:00"}]})["refused"][0]["reason"]
    d.committed.discard(3)
    out = d.call("ops", "ops_confirm", {"changes": [hold | {"time": "11:00"}]})
    assert out["confirmed"] == ["f3 hold 11:00"] and d.truth.run.flights[3].dep >= 660
    assert d.plan[3] == {"holds": [[60, 660]]}
    claim = d.call("ops", "ops_check", {"changes": [hold | {"action": "claim"}]})
    assert "in no program's pool" in claim["refused"][0]["reason"]


def test_crew_moves_are_the_crew_desks_from_their_minute_and_the_system_crews_the_rest(world):
    d = OccDay(world)
    d.call("crew", "wait_turn", {})
    move = {"action": "assign", "unit": "P3", "flight": "f0", "other": None, "kind": None, "time": None, "ref": None}
    out = d.call("crew", "crew_confirm", {"moves": [move, move | {"unit": "C3"}, move | {"unit": "X9"}]})
    assert out["confirmed"] == ["f0 assign P3", "f0 assign C3"] and "unknown unit" in out["refused"][0]["reason"]
    assert d.plan[0]["names"] == [[0, None, "pilot", "P3"], [0, None, "cabin", "C3"]]
    assert d.truth.crews[0] == {"pilot": "P3", "cabin": "C3"}
    back = d.call("crew", "crew_confirm", {"moves": [move | {"action": "release"}]})
    assert back["confirmed"] == ["f0 release P3"] and d.plan[0]["names"] == [[0, None, "cabin", "C3"]]
    assert d.call("crew", "crew_options", {"flight": "f2", "kind": "pilot"})["options"] == []


def mv(action: str, unit: str, flight: str | None = None, other: str | None = None, time: str | None = None) -> dict:
    return {"action": action, "unit": unit, "flight": flight, "other": other, "kind": None, "time": time, "ref": None}


MOVES = {"assign": ([mv("assign", "P3", "f0")], [mv("release", "P3", "f0")]),
         "call_reserve": ([mv("call_reserve", "RP", "f3")], [mv("release", "RP", "f3")]),
         "deadhead": ([mv("deadhead", "P3", "f0")], [mv("release", "P3", "f0")]),
         "rest": ([mv("rest", "C1", time="23:00")], [mv("release", "C1")]),
         "swap": ([mv("swap", "C1", other="C3")], [mv("release", "C1")]),
         "deadhead+assign": ([mv("deadhead", "P3", "f0"), mv("assign", "P3", "f1")],
                             [mv("release", "P3", "f1"), mv("release", "P3", "f0")])}


@pytest.mark.parametrize("move", sorted(MOVES))
def test_a_crew_move_released_in_its_turn_or_at_the_next_watch_leaves_the_day_as_without_it(world, move):
    reserves = tuple(crews.Unit(uid, "pilot", "A32X", crews.size("pilot", "A32X"), "JFK", "JFK", "JFK", 0, None, None,
                                reserve="short", rap=0, until=900, lead=120) for uid in ("RP", "RQ"))
    w = replace(world, units=(*world.units, *reserves))

    def day(moves, releases, at):
        d = OccDay(w)
        while not d.done:
            desk = d.turn["desk"]
            for n in list(d.pending):
                d.notice(n, "trigger")
            d.call(desk, "wait_turn", {})
            if desk == "crew" and d.turn["kind"] == "crew":
                for rows, k in ((moves, 0), (releases, at)):
                    if rows and d.k == k:
                        assert d.call("crew", "crew_confirm", {"moves": rows})["refused"] == []
            d.call(desk, "end_turn", END)
        return d.grade["cost"], d.truth.run.rows(), sorted(d.truth.crews.items()), crewsim.entries(d.plan)
    moves, releases = MOVES[move]
    assert day(moves, releases, 0) == day(moves, releases, 1) == day([], [], 0) != day(moves, [], 0)


def test_a_watch_waits_for_its_advisory_and_the_audit_flags_one_the_trigger_never_brought(world):
    d = OccDay(world)
    while d.k < 2:
        desk = d.turn["desk"]
        d.call(desk, "wait_turn", {})
        d.call(desk, "end_turn", END)
    assert d.pending == ["adv-b"] and d.call("crew", "wait_turn", {}) == {
        "waiting_for": "the crew desk's crew turn at watch 2 (02:00)"}
    d.missed("late")
    assert "page" in d.call("crew", "wait_turn", {})
    assert d.end()["audit"]["problems"]["A1"] == ["adv-b: watch 2's trigger never brought it"]


def test_a_desk_that_leaves_is_skipped_and_one_that_times_out_ends_the_day_unscored(world):
    left = play(world, policies={"crew": lambda d, desk, page: d.call(desk, "resign", {"reason": "turn_limit"})
                                 if d.k == 3 else None})
    assert {t["desk"] for t in left.turns if t["watch"] > 3} == {"ops"} and left.audit()["ok"]
    d = OccDay(world)
    d.call("crew", "wait_turn", {})
    d.time_out("crew")
    assert d.done and d.call("ops", "wait_turn", {})["done"] and d.call("ops", "ops_flights", {
        "ids": None, "station": None, "status": None, "start": None, "end": None}) == {"error": "the day is over",
                                                                                      "done": True}
    assert d.end()["audit"]["problems"] == {"A7": ["crew: timeout"]} and d.grade["cost"] == world.refs["idle"]


def test_the_projection_keeps_the_truth_before_the_watch_and_flags_each_kind(world):
    d = OccDay(world)
    while d.t < 360:
        for n in list(d.pending):
            d.notice(n, "trigger")
        desk = d.turn["desk"]
        d.call(desk, "wait_turn", {})
        d.call(desk, "end_turn", END)
    p = d.project(d.plan)
    assert run.history(p.day, d.t) == run.history(d.truth, d.t)
    flags = run.flags(p, world.task)
    assert flags[2]["pilot"] == ("no crew", None)
    assert flags[3]["pilot"][0] in ("crew", "legal", "times out") and 0 not in d.committed
    assert p.units["P2"]["station"] == "BUF" and p.aircraft["N1"][0] == "JFK"


def test_the_scripted_desks_play_the_talk_idle_day_and_a_recorded_day_replays_exactly(world):
    d = play(world, policies=scripted("crew", "ops"))
    talk = occ.run(case(world), "talk-idle")
    assert d.grade["cost"] == round(crewplan.worth(talk.day, False), 1) and d.truth.run.rows() == talk.day.run.rows()
    assert d.truth.run.flights[2].status == CANCELLED
    assert [r["status"] for r in d.requests if r["type"] == "CANCEL"] == ["accepted"]
    again = replay(world, d.data())
    assert again.exact and again.day.digest() == d.digest() == play(world, policies=scripted("crew", "ops")).digest()


def test_one_desk_runs_both_jobs_with_no_one_to_ask(world):
    d = play(world, "one", scripted("one"))
    assert d.audit()["ok"] and d.truth.run.flights[2].status == CANCELLED and all(
        t["kind"] == "one" for t in d.turns)
    e = OccDay(world, "one")
    e.call("both", "wait_turn", {})
    assert "no one to ask" in e.call("both", "ops_request", {"type": "TEXT", "flight": None, "time": None,
                                                              "text": "x"})["error"]
    assert e.truth.run.flights[0].status == FLOWN
