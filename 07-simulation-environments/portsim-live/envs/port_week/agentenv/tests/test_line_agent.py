"""portsim-line, the ships' agent (portsim-v1's agentenv/line_agent): its replies as sentences, its usage against the
planner's rules, the gateway's /step route it calls, and each request reaching its tool."""

import asyncio
import json

import portsim_line
from lines_fixture import real_lines
from port_week_core.lines import LinesWeek, pack
from port_week_core.prompts import LINES_RULES
from portsim_line import USAGE, sentence, step_url


@real_lines
def test_each_reply_reads_as_a_sentence_a_refusal_with_the_hour_offered_or_the_eta_an_agreed_ships_eta_too():
    week = LinesWeek(pack().get("dock-24B-w06x1-busy-0-e15"))
    assert sentence(week.status("globalliner", 3)) == "STEEN (ship 3): ETA hour 40."
    assert sentence(week.request("globalliner", 3, 46) | {"asked": 46}) == (
        "STEEN (ship 3) can't make hour 46: can't arrive that early; it can arrive at hour 47 at the earliest.")
    assert sentence(week.request("globalliner", 3, 54) | {"asked": 54}) == (
        "STEEN (ship 3) can't make hour 54: can't slow down that much; it can arrive at hour 53 at the latest.")
    assert sentence(week.request("globalliner", 3, 53) | {"asked": 53}) == (
        "STEEN (ship 3) will arrive at hour 53, reducing speed; it keeps its turn, hour 47.")
    assert sentence(week.request("globalliner", 3, 60) | {"asked": 60}) == (
        "STEEN (ship 3) can't make hour 60: can't slow down that much; it can arrive at hour 53 at the latest; it "
        "still arrives at hour 53, as agreed.")
    assert sentence(week.status("globalliner", 3) | {"agreed_before": True}) == (
        "STEEN (ship 3): ETA hour 53, as agreed; its turn is hour 47.")
    assert sentence({"ship": 0, "name": "PERSEUS", "agreed": False, "reason": "already arrived at hour 1",
                     "asked": 5}) == "PERSEUS (ship 0) can't make hour 5: already arrived at hour 1."
    assert sentence(week.request("noatum", 5, 60) | {"asked": 60}) == (
        "NANTO (ship 5) can't commit to hour 60 yet: it hasn't sailed from Alicante and is more than 48 hours out; "
        "ETA hour 55.")


def test_the_usage_shows_the_same_three_key_requests_as_the_planners_rules():
    shown = json.loads(USAGE.removeprefix("send ").split(": one entry")[0])["requests"]
    assert shown == [{"ship": 7, "ask": "eta", "request_arrival": None},
                     {"ship": 9, "ask": None, "request_arrival": 130}]
    assert all(json.dumps(r) in LINES_RULES for r in shown)


def test_the_step_route_sits_beside_the_mcp_route():
    assert step_url("http://gw:8080/envs/portsim/mcp/") == "http://gw:8080/envs/portsim/step"


def test_a_request_and_an_eta_query_each_reach_their_tool_and_both_at_once_is_refused():
    calls = []

    class Desk(portsim_line.Desk):
        async def call(self, name, arguments):
            calls.append((name, arguments))
            return {"ship": arguments["ship"], "eta": 31}

    desk = Desk(None, "", "globalliner")
    replies = [asyncio.run(desk.answer(e)) for e in (
        {"ship": 3, "ask": "eta"}, {"ship": 3, "ask": "eta", "request_arrival": None},
        {"ship": 3, "request_arrival": 42}, {"ship": 3, "ask": None, "request_arrival": 42},
        {"ship": 3, "ask": "eta", "request_arrival": 42}, {"ship": 3, "ask": None, "request_arrival": None})]
    assert calls == [("ship_status", {"ship": 3})] * 2 + [("agree_arrival", {"ship": 3, "arrival": 42})] * 2
    assert [r["asked"] for r in replies[2:4]] == [42, 42]
    assert all("one per request" in r["error"] for r in replies[4:])

