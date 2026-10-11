"""PortSimLiveEnv in-process: its tools through MCP and its middleware, its extensions and data plane over its HTTP
app, and the clock against a fake gateway."""

import httpx
import pytest
from agentenv_protocol import client
from berth_core import plan_to_list, situation
from port_week_core.schedule import schedule, virtual_time
from port_week_core.world import naive, play

from port_week_agentenv.contract import CLOCK_URI, END_WEEK_URI, LIVE_LOAD_URI, triggers
from port_week_agentenv.server import EMPTY

pytestmark = pytest.mark.anyio

NO_WEEK = "no week loaded: load one with urn:portsim:live-load/v1"
OVER = "the week is over"
NOTE = "This watch's bulletin arrives with your next tool result."


async def invoke(http, uri: str, **params) -> httpx.Response:
    card = (await http.get("/.well-known/agent-env.json")).json()
    return await http.post(client.extension_params(card, uri)["endpoint"], json=params)


async def data(http) -> dict:
    response = await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})
    return response.json()["result"]["parts"][0]["data"]


async def deliver(tools, task, watch: int) -> None:
    """What the gateway does when advance answers this watch: run its trigger's actions in order."""
    trigger = next(t for t in triggers(schedule(task)) if t["when"]["where"]["result.watch"]["equals"] == watch)
    for action in trigger["actions"]:
        assert await tools(action["tool"], **action["args"]) == {"applied": action["args"]["event_id"],
                                                                 "watch": watch}


def changes(week) -> list[dict]:
    target = naive(week)
    return [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]


async def test_the_card_and_routes(env, http):
    card = (await http.get("/.well-known/agent-env.json")).json()
    assert card["name"] == "portsim-live" and client.mcp_path(card) == "/mcp"
    assert card["capabilities"]["operations"] == ["data/reset", "data/get"]
    assert {e["uri"] for e in card["capabilities"]["extensions"]} == {LIVE_LOAD_URI, END_WEEK_URI, CLOCK_URI}
    assert sorted(route.path for route in env.mcp.http_app().routes) == [
        "/.well-known/agent-env.json", "/agentenv", "/agentenv/ext/end_week", "/agentenv/ext/live_load",
        "/agentenv/ext/sync_time", "/mcp"]


async def test_the_agents_tools_come_first_and_port_notice_last(tools):
    listed = await tools.client.list_tools()
    assert [t.name for t in listed] == ["get_situation", "check_plan", "confirm_berths", "advance", "port_notice"]
    assert all("\n" not in t.description for t in listed)
    assert listed[1].description == (
        "Check berth windows without confirming them: your entries over your confirmed windows, on the week as known "
        "now. Lists the ships with a rule problem or a cost, the plan's cost, and the entries confirm_berths would "
        "refuse. Problems and cost that news brought to a frozen window are listed as excused and not counted. Uses "
        "one of this watch's 3 planning calls.")


async def test_no_week_loaded(tools, http):
    for name, args in [("get_situation", {}), ("check_plan", {"plan": []}), ("confirm_berths", {"plan": []}),
                       ("advance", {}), ("port_notice", {"event_id": "gale-0", "name": "", "text": ""})]:
        assert await tools.error(name, **args) == f"Error calling tool '{name}': {NO_WEEK}"
    assert await data(http) == EMPTY
    response = await invoke(http, END_WEEK_URI)
    assert (response.status_code, response.json()["error"]["message"]) == (500, NO_WEEK)


async def test_live_load_starts_a_one_week_task(env, http, pack, example):
    unknown = await invoke(http, LIVE_LOAD_URI, task_id="no-such-task")
    assert (unknown.status_code, unknown.json()["error"]["message"]) == (500, "unknown task id 'no-such-task'")
    multi = next(t.task_id for t in pack.tasks if "x2-" in t.task_id)
    refused = await invoke(http, LIVE_LOAD_URI, task_id=multi)
    assert refused.json()["error"]["message"] == f"{multi} is not a one-week task: live weeks are one week"
    assert env.week is None
    assert (await invoke(http, LIVE_LOAD_URI, task_id=example.task_id)).json() == {"task_id": example.task_id,
                                                                                    "watches": 7}
    assert env.week.task is example and env.week.watch == 0


async def test_an_env_plays_one_week_even_after_a_reset(env, live, http):
    """A second load would replay the week with its news already seen, and the new week's audit couldn't tell."""
    again = "a week was already loaded in this env: each live week needs a new deploy"
    reload = await invoke(http, LIVE_LOAD_URI, task_id="dock-36A-w10x1-standard-0")
    assert (reload.status_code, reload.json()["error"]["message"]) == (500, again)
    assert env.week is live
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert (await invoke(http, LIVE_LOAD_URI, task_id=live.task.task_id)).json()["error"]["message"] == again
    assert env.week is None


async def test_sync_time_reads_the_gateway_clock(env, http, gateway):
    url = f"{gateway.url}/clock/time"
    assert (await invoke(http, CLOCK_URI, env_get_time_url=url)).json() == {"env_get_time_url": url,
                                                                            "virtual_time": "2024-02-12T00:00:00Z"}
    assert env.set_time_url == f"{gateway.url}/clock/set-time" and gateway.puts == []


async def test_get_situation(live, tools):
    out = await tools("get_situation")
    assert out == {"watch": 0, "hour": 0, "time": "Mon 00:00", "frozen_before": 0, "situation": situation(live.view),
                   "windows": [], "unconfirmed": list(range(16)), "planning_calls_left": 3, "messages": []}


async def test_check_plan_reports_the_merged_plan_and_stores_nothing(live, tools):
    out = await tools("check_plan", plan=[{"ship": 7, "berth_hour": 95, "section": 4, "cranes": 2},
                                          {"ship": 16, "berth_hour": 58, "section": 3}])
    assert out["feasible"] is False and (out["cost"], out["delay_cost"], out["moves"]) == (None, None, None)
    assert out["ships"][0] == {"ship": 0, "problems": ["missing from the plan"]}
    assert [r["ship"] for r in out["ships"]] == [s for s in range(16) if s != 7]
    assert (out["refused"], out["entry_problems"], out["planning_calls_left"], out["messages"]) == (
        [], ["unknown ship 16: use the ship ids from the table"], 2, [])
    full = await tools("check_plan", plan=plan_to_list(naive(live)))
    assert full["feasible"] and full["cost"] == full["delay_cost"] + 5 * full["moves"]
    assert all(r["cost"] > 0 or r["problems"] for r in full["ships"])
    assert set(full["ships"][0]) == {"ship", "berth_hour", "section", "cranes", "departure", "delay_h", "moved", "cost",
                                     "problems"}
    assert live.plan == {} and live.planning_calls == [2]


async def test_confirm_berths_sets_windows_and_ships_left_out_keep_theirs(live, tools):
    first = plan_to_list(naive(live))
    assert await tools("confirm_berths", plan=first[:10]) == {
        "confirmed": list(range(10)), "unchanged": [], "refused": [], "entry_problems": [], "planning_calls_left": 2,
        "messages": []}
    out = await tools("confirm_berths", plan=[*first[8:], {"ship": 99, "berth_hour": 1, "section": 1}])
    assert (out["confirmed"], out["unchanged"]) == (list(range(10, 16)), [8, 9])
    assert out["entry_problems"] == ["unknown ship 99: use the ship ids from the table"]
    assert plan_to_list(live.plan) == first


async def test_a_window_with_cranes_out_of_range_is_kept_and_reported(live, tools):
    window = {"ship": 0, "berth_hour": 1, "section": 4, "cranes": 0}
    assert (await tools("confirm_berths", plan=[window]))["confirmed"] == [0]
    checked = await tools("check_plan", plan=[])
    assert checked["ships"][0]["problems"] == ["gets 0 cranes but can be worked by 1-5"]
    await tools("advance")
    assert (await tools("get_situation"))["windows"] == [window | {"departure": 24, "status": "departed"}]


async def test_three_planning_calls_a_watch(live, tools):
    assert (await tools("check_plan", plan="{"))["planning_calls_left"] == 2
    assert (await tools.error("check_plan")).startswith("1 validation error for call[check_plan]\nplan\n")
    await tools("confirm_berths", plan=[])
    await tools("check_plan", plan=[])
    assert await tools("confirm_berths", plan=[]) == {
        "error": "no planning calls left in this watch (3 used): call advance", "planning_calls_left": 0,
        "messages": []}
    assert (live.planning_calls, live.calls) == ([3], 5)


async def test_advance_rearms_the_clock_frozen_at_the_next_bulletin(live, tools, gateway, example):
    await tools("confirm_berths", plan=plan_to_list(naive(live)))
    out = await tools("advance")
    assert gateway.puts == [{"virtual_time": "2024-02-13T06:00:00Z", "virtual_seconds_per_real_second": 0}]
    assert gateway.virtual_time == virtual_time(example, 30)
    assert set(out) == {"watch", "hour", "time", "berthed", "departed", "frozen_before", "unconfirmed", "note",
                        "messages"}
    assert (out["watch"], out["hour"], out["time"], out["frozen_before"], out["note"]) == (1, 30, "Tue 06:00", 36, NOTE)
    assert out["berthed"] and out["departed"] and out["messages"] == []
    assert live.frozen[1] == {"watch": 1, "hour": 30, "before": 36, "call": 2,
                              "ships": sorted(s for s, e in live.plan.items() if e[0] < 36)}


async def test_advance_needs_the_synced_clock_except_after_the_last_watch(env, tools):
    env.live_load("dock-36A-w10x1-standard-0")
    assert await tools.error("advance") == (
        "Error calling tool 'advance': the gateway clock is not synced: call urn:agentenv:clock/v1 sync_time first")
    assert env.week.watch == 0
    env.week.open()
    out = await tools("advance")
    assert out == {"done": True, "feasible": False, "cost": None, "reward": 0.0, "messages": []}
    assert (env.week.end_reason, env.week.watch) == ("done", 1)


async def test_both_refusals_come_from_the_harbour_master(live, tools, example):
    await tools("confirm_berths", plan=plan_to_list(naive(live)))
    await tools("advance")
    h0, sec0, _ = live.plan[0]
    later = next(s for s, e in sorted(live.plan.items()) if e[0] >= 36)
    out = await tools("confirm_berths", plan=[{"ship": 0, "berth_hour": h0 + 1, "section": sec0},
                                              {"ship": later, "berth_hour": 35, "section": live.plan[later][1]}])
    assert out["refused"] == [
        {"ship": 0, "from": "Harbour master",
         "reason": f"ship 0 ZIM ATLANTIC is frozen: its window starts at hour {h0}, before the freeze line at hour 36; "
                   "it stands."},
        {"ship": later, "from": "Harbour master",
         "reason": f"ship {later} {example.ships[later].name} can't start at hour 35: windows must start at or after "
                   "the freeze line at hour 36."}]
    checked = await tools("check_plan", plan=[{"ship": 0, "berth_hour": h0 + 1, "section": sec0}])
    assert checked["refused"] == out["refused"][:1] and live.plan[0][0] == h0
    assert [r["ship"] for r in live.refusals] == [0, later]


async def test_port_notice_applies_the_news_and_queues_the_message(live, tools, example):
    await tools("advance")
    assert 16 not in [s.id for s in live.view.ships]
    assert await tools("port_notice", event_id="extra-1", name="MAERSK NUBA", text=example.notices[1]) == {
        "applied": "extra-1", "watch": 1}
    out = await tools("get_situation")
    assert out["messages"] == [{"hour": 30, "from": "MAERSK NUBA", "text": example.notices[1]}]
    assert out["unconfirmed"][-1] == 16 and example.notices[1] in out["situation"]
    assert (await tools("get_situation"))["messages"] == []
    assert live.calls == 3 and live.audit() == {"ok": True, "problems": []}
    assert await tools.error("port_notice", event_id="extra-9", name="", text="") == (
        "Error calling tool 'port_notice': unknown event 'extra-9'")


async def test_a_double_advance_gets_the_first_bulletin_with_the_second(live, tools, gateway, example):
    await tools("confirm_berths", plan=plan_to_list(naive(live)))
    assert (await tools("advance"))["watch"] == 1
    await deliver(tools, example, 1)
    out = await tools("advance")
    assert (out["watch"], out["messages"]) == (2, [{"hour": 30, "from": "MAERSK NUBA", "text": example.notices[1]}])
    await deliver(tools, example, 2)
    assert [p["virtual_time"] for p in gateway.puts] == ["2024-02-13T06:00:00Z", "2024-02-13T18:00:00Z"]
    assert live.audit() == {"ok": True, "problems": []}
    assert [f["call"] for f in live.frozen] == [0, 2, 3]


async def test_a_notice_after_the_agents_next_call_fails_the_audit(live, tools, http, example):
    await tools("advance")
    await tools("get_situation")
    await deliver(tools, example, 1)
    assert (await data(http))["audit"] == {
        "ok": False, "problems": ["extra-1 arrived after the agent's next call in watch 1"]}


async def test_end_week_runs_the_rest_and_the_week_is_over(live, tools, http, gateway):
    await tools("confirm_berths", plan=plan_to_list(naive(live)))
    await tools("advance")
    await deliver(tools, live.task, 1)
    out = (await invoke(http, END_WEEK_URI)).json()
    assert out == {"end_reason": "end_week", "watch": 6, "feasible": live.grade["feasible"], "cost": live.grade["cost"],
                   "reward": live.grade["reward"], "audit": {"ok": True, "problems": []}}
    assert len(gateway.puts) == 1 and [e["via"] for e in live.log] == ["load", "trigger"] + ["env"] * 6
    assert (await invoke(http, END_WEEK_URI)).json() == out
    for name, args in [("check_plan", {"plan": []}), ("confirm_berths", {"plan": []}), ("advance", {}),
                       ("port_notice", {"event_id": "gale-0", "name": "Harbour master", "text": ""})]:
        assert await tools.error(name, **args) == f"Error calling tool '{name}': {OVER}"
    assert (await tools("get_situation"))["watch"] == 6


async def test_data_get_and_reset(env, live, tools, http, gateway):
    await tools("confirm_berths", plan=plan_to_list(naive(live)))
    await tools("advance")
    await deliver(tools, live.task, 1)
    got = await data(http)
    assert set(got) == set(EMPTY) and "reference" not in str(got) and "optimal_plan" not in str(got)
    assert {k: got[k] for k in ("task_id", "watches", "watch", "hour", "done", "end_reason", "calls_used",
                                "planning_calls", "grade", "reward")} == {
        "task_id": "dock-24B-w07x1-busy-0", "watches": 7, "watch": 1, "hour": 30, "done": False, "end_reason": None,
        "calls_used": 2, "planning_calls": [1, 0], "grade": None, "reward": None}
    assert got["plan"] == plan_to_list(live.plan) and got["audit"] == {"ok": True, "problems": []}
    assert [n["event_id"] for n in got["notices"]] == ["closure-2", "extra-1"]
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert await data(http) == EMPTY and env.week is None and env.set_time_url == f"{gateway.url}/clock/set-time"
    assert await tools.error("get_situation") == f"Error calling tool 'get_situation': {NO_WEEK}"


@pytest.mark.parametrize("task_id", ["dock-24B-w07x1-busy-0", "dock-24B-w16x1-busy-0", "dock-36A-w17x1-standard-0"])
async def test_a_whole_week_with_the_naive_policy(env, tools, http, gateway, pack, task_id):
    """The naive policy played through the tools as an agent would, with the triggers' calls in between, scores what
    play() scores; at a watch where it changes nothing it advances twice in a row."""
    task = pack.get(task_id)
    assert (await invoke(http, LIVE_LOAD_URI, task_id=task_id)).status_code == 200
    await invoke(http, CLOCK_URI, env_get_time_url=f"{gateway.url}/clock/time")
    week, calls, doubles = env.week, 0, 0
    expected = play(task, naive)
    while True:
        rows = changes(week)
        if rows or week.watch == 0:
            situation_now = await tools("get_situation")
            assert situation_now["watch"] == week.watch and situation_now["planning_calls_left"] == 3
            calls += 1
        else:
            doubles += 1
        if rows:
            assert (await tools("confirm_berths", plan=rows))["refused"] == []
            calls += 1
        out = await tools("advance")
        calls += 1
        if out.get("done"):
            break
        assert gateway.virtual_time == virtual_time(task, out["hour"]) and out["watch"] == week.watch
        await deliver(tools, task, out["watch"])
    out.pop("messages")
    assert out == {"done": True, "feasible": expected.grade["feasible"], "cost": expected.grade["cost"],
                   "reward": expected.grade["reward"]}
    assert doubles and gateway.puts == [
        {"virtual_time": virtual_time(task, w.hour), "virtual_seconds_per_real_second": 0} for w in week.watches[1:]]
    got = await data(http)
    assert (got["done"], got["end_reason"], got["calls_used"], got["audit"]) == (
        True, "done", calls, {"ok": True, "problems": []})
    assert (got["grade"], got["plan"], got["planning_calls"]) == (
        expected.grade, plan_to_list(expected.plan), expected.planning_calls)
    assert [(n["event_id"], n["watch"], n["via"]) for n in got["notices"]] == [
        (e["event_id"], e["watch"], e["via"]) for e in expected.log]

