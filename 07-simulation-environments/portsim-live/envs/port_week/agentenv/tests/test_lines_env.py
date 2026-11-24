"""PortSimLinesEnv in-process: the wind env's tools, extensions and data plane under its own name with the ships'
agents' two tools, their answers the week's, the middleware leaving them out of the planner's calls, and the agents'
exchanges and agreed arrivals in data/get."""

import json

import httpx
import pytest
from agentenv_protocol import client
from fastmcp import Client
from lines_fixture import require_lines
from port_week_core import lines
from port_week_core.lines import LinesWeek

from port_week_agentenv.contract import CLOCK_URI, END_WEEK_URI, LIVE_LOAD_URI
from port_week_agentenv.server import EMPTY, PortSimLinesEnv, PortSimWindEnv

pytestmark = pytest.mark.anyio
require_lines()

TASK_ID = "dock-24B-w06x1-busy-0-e15"
TASK = lines.pack().get(TASK_ID)
NO_WEEK = "no week loaded: load one with urn:portsim:live-load/v1"


class Tools:
    """Calls tools as an MCP client does, through argument validation and the env's middleware; results are JSON."""

    def __init__(self, client: Client):
        self.client = client

    async def __call__(self, tool: str, /, **arguments) -> dict:
        result = await self.client.call_tool_mcp(tool, arguments)
        assert not result.isError, result.content[0].text
        return json.loads(result.content[0].text)

    async def error(self, tool: str, /, **arguments) -> str:
        result = await self.client.call_tool_mcp(tool, arguments)
        assert result.isError
        return result.content[0].text


@pytest.fixture
def env():
    env = PortSimLinesEnv()
    env.create_app()
    return env


@pytest.fixture
async def tools(env):
    async with Client(env.mcp) as mcp:
        yield Tools(mcp)


@pytest.fixture
async def http(env):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()), base_url="http://portsim") as http:
        yield http


async def invoke(http, uri: str, **params) -> httpx.Response:
    card = (await http.get("/.well-known/agent-env.json")).json()
    return await http.post(client.extension_params(card, uri)["endpoint"], json=params)


async def data(http) -> dict:
    response = await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})
    return response.json()["result"]["parts"][0]["data"]


async def card(env) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()), base_url="http://portsim") as http:
        return (await http.get("/.well-known/agent-env.json")).json()


async def test_the_lines_env_is_the_wind_env_with_the_agents_two_tools_and_a_confirm_that_reports(env, tools):
    other = PortSimWindEnv()
    other.create_app()
    ours, theirs = await card(env), await card(other)
    assert (ours["name"], env.mcp.name) == ("portsim-lines", "portsim-lines")
    assert {e["uri"] for e in ours["capabilities"]["extensions"]} == {LIVE_LOAD_URI, END_WEEK_URI, CLOCK_URI}
    assert ours["capabilities"]["operations"] == theirs["capabilities"]["operations"]
    named = {t["name"]: t for t in ours["capabilities"]["tools"]}
    confirm = next(t for t in theirs["capabilities"]["tools"] if t["name"] == "confirm_berths")
    assert named["confirm_berths"] == confirm | {"description": confirm["description"].replace(
        " Uses one", " Returns the confirmed plan's problems and cost, as check_plan does. Uses one")}
    assert [t for t in ours["capabilities"]["tools"] if t["name"] not in ("ship_status", "agree_arrival",
                                                                         "confirm_berths")] == [
        t for t in theirs["capabilities"]["tools"] if t["name"] != "confirm_berths"]
    assert set(named) == {t["name"] for t in theirs["capabilities"]["tools"]} | {"ship_status", "agree_arrival"}
    assert [t.name for t in await tools.client.list_tools()] == [
        "get_situation", "check_plan", "confirm_berths", "advance", "port_notice", "ship_status", "agree_arrival"]
    listed = {t.name: t for t in await tools.client.list_tools()}
    assert set(listed["ship_status"].inputSchema["properties"]) == {"agent", "ship"}
    assert set(listed["agree_arrival"].inputSchema["properties"]) == {"agent", "ship", "arrival"}
    assert listed["agree_arrival"].description.endswith("For the ships' agents.")


async def test_live_load_starts_a_week_of_the_lines_pack(env, http):
    v4 = "dock-24B-w06x1-busy-0"
    refused = await invoke(http, LIVE_LOAD_URI, task_id=v4)
    assert (refused.status_code, refused.json()["error"]["message"]) == (500, f"unknown task id {v4!r}")
    assert (await invoke(http, LIVE_LOAD_URI, task_id=TASK_ID)).json() == {"task_id": TASK_ID, "watches": 6}
    assert type(env.week) is LinesWeek and env.week.task == TASK


async def test_the_agents_tools_need_a_week(tools):
    for name, args in [("ship_status", {"agent": "globalliner", "ship": 3}),
                       ("agree_arrival", {"agent": "globalliner", "ship": 3, "arrival": 42})]:
        assert await tools.error(name, **args) == f"Error calling tool '{name}': {NO_WEEK}"


async def test_the_agents_tools_give_the_weeks_answers_and_are_not_the_planners_calls(env, tools):
    env.live_load(TASK_ID)
    expected = LinesWeek(TASK)
    assert await tools("ship_status", agent="boluda", ship=7) == expected.status("boluda", 7) == {
        "ship": 7, "name": "VERONICA B", "eta": 84, "as_of": 0}
    assert await tools("agree_arrival", agent="globalliner", ship=3, arrival=54) == expected.request(
        "globalliner", 3, 54)
    assert await tools("agree_arrival", agent="globalliner", ship=3, arrival=53) == expected.request(
        "globalliner", 3, 53) == {"ship": 3, "name": "STEEN", "agreed": True, "arrival": 53, "turn": 47,
                                  "slowing": True}
    assert await tools.error("agree_arrival", agent="aperez", ship=3, arrival=40) == (
        "Error calling tool 'agree_arrival': ship 3 STEEN is not this agent's: Global Liner Agencias handles it")
    assert await tools.error("ship_status", agent="hapag", ship=16) == (
        "Error calling tool 'ship_status': no ship 16 in the week")
    week = env.week
    assert (week.calls, week.exchanges, week.locks, week.told) == (0, expected.exchanges, expected.locks, expected.told)
    out = await tools("get_situation")
    assert week.calls == 1 and out["messages"] == []
    assert "## Arrivals and the ships' agents" in out["situation"]
    assert "- ship 3 STEEN: arrival hour 53, agreed hour 53, turn 47; from Valencia" in out["situation"].splitlines()
    assert "- ship 7 VERONICA B: arrival hour 84, ETA hour 84, as of Mon 00:00; from Alicante" in (
        out["situation"].splitlines())
    await tools("ship_status", agent="globalliner", ship=3)
    assert week.calls == 1 and week.exchanges[-1]["call"] == 1


async def test_end_week_grades_the_week_with_the_agreed_arrival_and_data_get_reports_the_agents(env, tools, http):
    env.live_load(TASK_ID)
    await tools("agree_arrival", agent="globalliner", ship=3, arrival=53)
    await tools("ship_status", agent="boluda", ship=7)
    out = (await invoke(http, END_WEEK_URI)).json()
    assert (out["end_reason"], out["watch"], out["audit"]) == ("end_week", 5, {"ok": True, "problems": []})
    got = await data(http)
    assert set(got) == set(EMPTY) | {"lines"} and "optimal_plan" not in str(got)
    assert got["lines"]["agreed"] == [{"ship": 3, "arrival": 53, "turn": 47}]
    assert [(e["firm"], e["ship"], e["ask"]) for e in got["lines"]["exchanges"]] == [
        ("globalliner", 3, "request_arrival"), ("boluda", 7, "eta")]
    assert got["lines"]["arrivals"][3] == {"ship": 3, "arrival": 53, "turn": 47}
    assert got["grade"] == env.week.grade and got["grade"]["unavoidable_cost"] == 308
    assert await tools.error("ship_status", agent="boluda", ship=7) == (
        "Error calling tool 'ship_status': the week is over")
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert await data(http) == EMPTY
