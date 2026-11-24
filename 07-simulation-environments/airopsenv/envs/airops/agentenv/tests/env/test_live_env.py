"""AirOpsLiveEnv in process: its card and routes, its tools through MCP and its middleware (strict schemas, the
planner's tools first), its extensions and data plane over its HTTP app, the clock against a fake gateway, the news the
triggers deliver, and the server as the image runs it."""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from agentenv_protocol import client
from airops_core.core.model import hhmm
from airops_core.day import Day
from airops_core.schedule import CLOCK_URI, END_DAY_URI, LIVE_LOAD_URI, virtual_time
from paths import FIXTURES, PACK

from airops_agentenv.live import EMPTY

pytestmark = pytest.mark.anyio

SMALL = "b6-jfk-20230626-1400-1700"
STORM = "b6-jfk-20230714-1400"
NO_DAY = "no day loaded: load one with urn:airops:live-load/v1"
NOTE = "This watch's news arrives with your next tool result."


async def invoke(http, uri: str, **params) -> httpx.Response:
    card = (await http.get("/.well-known/agent-env.json")).json()
    return await http.post(client.extension_params(card, uri)["endpoint"], json=params)


async def data(http) -> dict:
    response = await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})
    return response.json()["result"]["parts"][0]["data"]


def strict(schema: dict, where: str = "") -> list[str]:
    """Where a schema breaks OpenAI's strict mode: an object must close its properties and require every one; an
    optional field is a nullable one."""
    problems = []
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            problems.append(f"{where}: additionalProperties is not false")
        if set(schema.get("required", [])) != set(props):
            problems.append(f"{where}: required {schema.get('required')} is not every property")
        for name, sub in props.items():
            problems += strict(sub, f"{where}.{name}")
    if "default" in schema:
        problems.append(f"{where}: has a default")
    for sub in schema.get("anyOf", []):
        problems += strict(sub, where)
    if "items" in schema:
        problems += strict(schema["items"], f"{where}[]")
    return problems


@pytest.fixture
async def live(env, gateway):
    """The storm day loaded on an env synced to the fake gateway's clock."""
    env.live_load(STORM)
    await env.sync_time(f"{gateway.url}/clock/time")
    return env.day


async def test_the_card_and_routes(env, http):
    card = (await http.get("/.well-known/agent-env.json")).json()
    assert card["name"] == "airops-day" and client.mcp_path(card) == "/mcp"
    assert card["capabilities"]["operations"] == ["data/reset", "data/get"]
    assert {e["uri"] for e in card["capabilities"]["extensions"]} == {LIVE_LOAD_URI, END_DAY_URI, CLOCK_URI}
    assert sorted(route.path for route in env.mcp.http_app().routes) == [
        "/.well-known/agent-env.json", "/agentenv", "/agentenv/ext/end_day", "/agentenv/ext/live_load",
        "/agentenv/ext/sync_time", "/mcp"]


async def test_the_planners_tools_come_first_in_one_line_each_with_strict_schemas(tools):
    listed = await tools.client.list_tools()
    assert [t.name for t in listed] == ["get_situation", "check_plan", "confirm_changes", "advance", "airops_notice"]
    assert all("\n" not in t.description for t in listed)
    assert {t.name: strict(t.inputSchema, t.name) for t in listed} == {t.name: [] for t in listed}
    item = listed[1].inputSchema["properties"]["changes"]["anyOf"][0]["items"]
    assert item["required"] == ["flight", "action", "not_before"]
    assert item["properties"]["action"]["enum"] == ["cancel", "restore", "hold"]
    assert item["properties"]["not_before"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


async def test_no_day_loaded(tools, http):
    for name, args in [("get_situation", {}), ("check_plan", {"changes": []}), ("confirm_changes", {"changes": []}),
                       ("advance", {}), ("airops_notice", {"news_id": "x"})]:
        assert await tools.error(name, **args) == f"Error calling tool '{name}': {NO_DAY}"
    assert await data(http) == EMPTY
    response = await invoke(http, END_DAY_URI)
    assert (response.status_code, response.json()["error"]["message"]) == (500, NO_DAY)


async def test_live_load_starts_a_day_once_per_env(env, http):
    unknown = await invoke(http, LIVE_LOAD_URI, task_id="no-such-day")
    assert (unknown.status_code, unknown.json()["error"]["message"]) == (500, "unknown task id 'no-such-day'")
    assert (await invoke(http, LIVE_LOAD_URI, task_id=STORM)).json() == {"task_id": STORM, "watches": 13}
    assert env.day.task.task_id == STORM and env.day.watch == 0
    again = "a day was already loaded in this env: each live day needs a new deploy"
    assert (await invoke(http, LIVE_LOAD_URI, task_id=SMALL)).json()["error"]["message"] == again
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert env.day is None
    assert (await invoke(http, LIVE_LOAD_URI, task_id=STORM)).json()["error"]["message"] == again


async def test_sync_time_reads_the_gateway_clock(env, http, gateway):
    url = f"{gateway.url}/clock/time"
    assert (await invoke(http, CLOCK_URI, env_get_time_url=url)).json() == {"env_get_time_url": url,
                                                                            "virtual_time": "2023-06-26T17:00:00Z"}
    assert env.set_time_url == f"{gateway.url}/clock/set-time" and gateway.puts == []


async def test_get_situation(live, tools):
    out = await tools("get_situation")
    assert out == {"watch": 0, "time": "13:00", "freeze_line": None, "planning_calls_left": 3,
                   "situation": live.text(), "messages": []}


async def test_changes_go_through_as_a_list_or_a_json_string(live, tools):
    checked = await tools("check_plan", changes='[{"flight": "f5", "action": "cancel", "not_before": null}]')
    assert checked["planning_calls_left"] == 2 and live.plan == {}
    out = await tools("confirm_changes", changes=[{"flight": "f5", "action": "cancel", "not_before": None},
                                                  {"flight": "f9", "action": "hold", "not_before": "15:00"}])
    assert (out["confirmed"], out["refused"], out["problems"], out["planning_calls_left"]) == (["f5", "f9"], [], [], 1)
    assert live.plan == {5: {"cancel": True, "not_before": None}, 9: {"cancel": False, "not_before": 900}}
    assert (await tools("confirm_changes", changes=[{"flight": "f6", "action": "cancel"}]))["confirmed"] == ["f6"]
    assert "validation errors for call[confirm_changes]" in await tools.error("confirm_changes",
                                                                               changes=[{"flight": "f1"}])
    assert live.calls == 4 and live.planning_calls == [3]


async def test_advance_rearms_the_clock_frozen_at_the_next_watch_and_the_news_comes_next(live, tools, gateway, env):
    out = await tools("advance")
    task = live.task
    assert gateway.puts == [{"virtual_time": virtual_time(task, 840), "virtual_seconds_per_real_second": 0}]
    assert gateway.puts[0]["virtual_time"] == "2023-07-14T18:00:00Z"
    assert out == {"watch": 1, "time": "14:00", "freeze_line": "15:00", "frozen": len(live.frozen),
                   "note": NOTE, "messages": []}
    news = live.watches[1].news
    for n in news:
        assert await tools("airops_notice", news_id=n) == {"delivered": n, "watch": 1}
    assert live.calls == 1 and live.freezes[1]["call"] == 1
    situation = await tools("get_situation")
    assert situation["messages"] == [{"at": hhmm(live.news[n].at), "from": live.news[n].speaker,
                                      "kind": live.news[n].kind, "text": live.news[n].text} for n in news]
    assert (await tools("get_situation"))["messages"] == [] and live.audit()["ok"]


async def test_advance_needs_the_synced_clock_except_after_the_last_watch(env, tools):
    env.live_load(SMALL)
    assert await tools.error("advance") == (
        "Error calling tool 'advance': the gateway clock is not synced: call urn:agentenv:clock/v1 sync_time first")
    while not env.day.last:
        env.day.open()
    out = await tools("advance")
    assert out == {"done": True, "cost": env.day.task.reference["naive"], "reward": env.day.grade["reward"],
                   "messages": []}
    assert await tools.error("check_plan", changes=[]) == "Error calling tool 'check_plan': the day is over"


async def test_end_day_runs_the_rest_and_data_get_reports_it(live, http, tools):
    await tools("confirm_changes", changes=[{"flight": "f5", "action": "cancel", "not_before": None}])
    out = (await invoke(http, END_DAY_URI)).json()
    assert out == {"end_reason": "end_day", "watch": 12, "cost": live.grade["cost"]["total"],
                   "reward": live.grade["reward"], "audit": {"ok": True, "problems": []}}
    got = await data(http)
    assert got == live.data() and got["done"] and got["plan"] == [{"flight": 5, "cancel": True, "not_before": None}]
    played = Day(live.task)
    played.confirm([{"flight": "f5", "action": "cancel", "not_before": None}])
    played.end()
    assert got["grade"] == played.grade


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_the_server_runs_as_the_image_runs_it():
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "airops_agentenv.server"],
                            env={**os.environ, "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port),
                                 "AIROPS_TASKS_DIR": f"{FIXTURES}:{PACK}", "ENVIRONMENT_NAME": "airops-day"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                card = httpx.get(f"http://127.0.0.1:{port}/.well-known/agent-env.json").raise_for_status().json()
                break
            except httpx.HTTPError:
                assert proc.poll() is None and time.monotonic() < deadline, "the env server did not start"
                time.sleep(0.1)
        assert card["name"] == "airops-day"
        loaded = httpx.post(f"http://127.0.0.1:{port}{client.extension_params(card, LIVE_LOAD_URI)['endpoint']}",
                            json={"task_id": SMALL}).json()
        assert loaded == {"task_id": SMALL, "watches": 13}
    finally:
        proc.terminate()
        proc.wait(10)


def test_virtual_time_is_the_minute_hub_local_in_utc(fixtures, days):
    task = fixtures[SMALL]
    assert virtual_time(task, 780) == "2023-06-26T17:00:00Z"
    assert virtual_time(task, 1500) == "2023-06-27T05:00:00Z"
    assert virtual_time(days["b6-bos-20250216-1500"], 300) == "2025-02-16T10:00:00Z"
