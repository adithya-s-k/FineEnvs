"""PortSimWindEnv in-process on the synthetic wind weeks: the marine env's tools, extensions and data plane under its
own name, the wind after the pilots and tugs in get_situation, check_plan on the forecast and the wind observed, whole
wind weeks played through the tools, and main() serving it by its name with the wind pack in PORT_WEEK_TASKS_DIR."""

import json

import httpx
import pytest
from agentenv_protocol import client
from berth_core import TaskPack, plan_from_list, plan_to_list, situation
from fastmcp import Client
from port_week_core import marine, wind
from port_week_core.schedule import schedule, virtual_time
from port_week_core.wind import WindWeek
from port_week_core.world import naive, play
from serving import served
from wind_fixture import BUST_TASK, PACK_DIR, REFERENCES, STORM_TASK, tasks_dir, use

from port_week_agentenv.contract import CLOCK_URI, END_WEEK_URI, LIVE_LOAD_URI, triggers
from port_week_agentenv.server import EMPTY, PortSimMarineEnv, PortSimWindEnv

pytestmark = pytest.mark.anyio

PACK = TaskPack(PACK_DIR)
TASK = PACK.get(STORM_TASK)
REFS = {r["task_id"]: r for r in map(json.loads, REFERENCES.read_text().splitlines())}
BIG = TASK.ships[8]


@pytest.fixture
def env(monkeypatch):
    use(monkeypatch)
    env = PortSimWindEnv()
    env.create_app()
    return env


@pytest.fixture
async def week(env, gateway):
    """The synthetic storm week loaded on an env synced to the fake gateway's clock."""
    env.live_load(STORM_TASK)
    await env.sync_time(f"{gateway.url}/clock/time")
    return env.week


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


async def card(env) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()), base_url="http://portsim") as http:
        return (await http.get("/.well-known/agent-env.json")).json()


def ending(view) -> str:
    return "\n\n" + marine.section(view) + "\n\n" + wind.section(view)


def ship_row(out: dict, ship: int) -> dict:
    return next(r for r in out["ships"] if r["ship"] == ship)


async def test_the_wind_env_is_the_marine_env_under_its_own_name(env, tools):
    other = PortSimMarineEnv()
    other.create_app()
    ours, theirs = await card(env), await card(other)
    assert (ours["name"], theirs["name"], env.mcp.name) == ("portsim-wind", "portsim-marine", "portsim-wind")
    assert {e["uri"] for e in ours["capabilities"]["extensions"]} == {LIVE_LOAD_URI, END_WEEK_URI, CLOCK_URI}
    assert ours["capabilities"]["tools"] == theirs["capabilities"]["tools"]
    assert ours["capabilities"]["operations"] == ["data/reset", "data/get"]
    assert sorted(route.path for route in env.mcp.http_app().routes) == sorted(
        route.path for route in other.mcp.http_app().routes)
    assert [t.name for t in await tools.client.list_tools()] == [
        "get_situation", "check_plan", "confirm_berths", "advance", "port_notice"]


async def test_live_load_starts_a_week_of_the_wind_pack(env, http):
    v3 = "dock-24B-w07x1-busy-0"
    refused = await invoke(http, LIVE_LOAD_URI, task_id=v3)
    assert (refused.status_code, refused.json()["error"]["message"]) == (500, f"unknown task id {v3!r}")
    assert (await invoke(http, LIVE_LOAD_URI, task_id=STORM_TASK)).json() == {"task_id": STORM_TASK, "watches": 8}
    assert type(env.week) is WindWeek and env.week.task == TASK
    assert env.week.task.reference["optimal_cost"] == REFS[STORM_TASK]["optimal_cost"] == 234
    assert [(w["start"], w["end"], w["min_length"]) for w in env.week.task.rules["no_moves"]] == [
        (37, 41, 300), (74, 88, 300), (80, 84, 0)]
    assert env.week.view.rules["no_moves"] == []


async def test_get_situation_ends_with_the_wind_after_the_pilots_and_tugs(week, tools):
    out = await tools("get_situation")
    assert out["situation"] == situation(week.view) + ending(week.view)
    assert out["messages"] == []
    lines = out["situation"].splitlines()
    assert lines.index("## Pilots and tugs") < lines.index("## Wind at the Dique Sur") == len(lines) - 5
    assert lines[-4:] == [
        "- Above 25 kn (10-minute mean) ships of 300 m or more may not berth or leave and every movement takes one "
        "more tug; above 30 kn no ship moves. The rules follow the wind that blows; these windows are forecast.",
        "- Barcelona Port Control, issued Sun 21:00 (ECMWF, adjusted to this anemometer), to hour 60: no hour above "
        "25 kn, peak 12 kn.",
        "- kn every 3 h from hour 6: " + " ".join(["12"] * 19),
        "- Observed since hour 0: no hour above 25 kn."]
    assert not [line for line in lines if line.endswith("takes one more tug.")]


async def test_the_forecast_arrives_with_its_watch_and_check_plan_names_its_windows(week, tools):
    entry = [{"ship": 8, "berth_hour": 79, "section": BIG.planned_section, "cranes": BIG.std_cranes}]
    assert BIG.length_m >= 300 and BIG.arrival == 79
    assert not [r for r in (await tools("check_plan", plan=entry))["ships"] if r["ship"] == 8]
    assert (await tools("advance"))["watch"] == 1
    await deliver(tools, TASK, 1)
    out = await tools("get_situation")
    assert out["messages"][-1] == {"hour": 30, "from": "Barcelona Port Control", "text": (
        "Wind forecast issued Tue 03:00 (ECMWF, adjusted to the Dique Sur anemometer), to hour 90: above 25 kn hours "
        "75–85, peak 32 kn; above 30 kn 77–80.")}
    assert out["situation"].endswith(ending(week.view))
    lines = out["situation"].splitlines()
    assert "- From hour 75 to hour 85 (wind above 25 kn), each movement takes one more tug." in lines
    assert lines[-2:] == ["- kn every 3 h from hour 36: 12 12 12 12 12 12 12 12 12 12 12 12 16 28 32 27 27 19 12",
                          "- Observed since hour 0: no hour above 25 kn."]
    checked = await tools("check_plan", plan=entry)
    assert ship_row(checked, 8)["problems"] == ["berths at hour 79, inside the no-movement window 75-85"]
    assert checked["feasible"] is False and 8 not in week.plan


async def test_wind_that_no_forecast_showed_is_excused_on_a_frozen_ship_in_check_plan(week, tools):
    plans = REFS[STORM_TASK]["rolling"]["plans"]
    for k in range(2):
        target = plan_from_list(plans[k])
        rows = [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]
        assert (await tools("confirm_berths", plan=rows))["refused"] == []
        assert (await tools("advance"))["watch"] == k + 1
        await deliver(tools, TASK, k + 1)
    out = await tools("get_situation")
    assert out["messages"][-1]["from"] == "Barcelona Port Control"
    assert out["situation"].splitlines()[-1] == "- Observed since hour 0: above 25 kn hours 37–41."
    assert week.plan[3][0] == 39 and 3 in week.fixed
    row = ship_row(await tools("check_plan", plan=[]), 3)
    assert (row["problems"], row["excused"], row["excused_cost"]) == (
        [], ["berths at hour 39, inside the no-movement window 37-41"], 0)
    assert week.data()["excused"] == {"problems": [], "cost": []}


async def test_end_week_grades_on_the_wind_that_blew_and_data_get_reports_it(week, tools, http):
    plan = REFS[STORM_TASK]["rolling"]["plans"][0]
    await tools("confirm_berths", plan=plan)
    await tools("advance")
    await deliver(tools, TASK, 1)
    expected = WindWeek(TASK)
    expected.confirm(plan)
    expected.open()
    for n in expected.watches[1].notices:
        expected.notice(n.event_id, n.name, n.text, "trigger")
    expected.end()
    out = (await invoke(http, END_WEEK_URI)).json()
    assert out == {"end_reason": "end_week", "watch": 7, "feasible": expected.grade["feasible"],
                   "cost": expected.grade["cost"], "reward": expected.grade["reward"],
                   "audit": {"ok": True, "problems": []}}
    got = await data(http)
    assert set(got) == set(EMPTY) and "optimal_plan" not in str(got) and "movements" not in str(got)
    assert got["grade"] == expected.grade and got["grade"]["optimal_cost"] == 234
    assert got["excused"] == {"problems": [], "cost": []}
    assert [n["event_id"] for n in got["notices"]] == [
        "closure-1", "forecast-9", "extra-0", "tug_outage-7", "pilot_shortage-8", "forecast-10", "emergency-6",
        "forecast-11", "late-2", "forecast-12", "forecast-13", "late-3", "crane_outage-5", "forecast-14",
        "forecast-15", "bunching-4", "forecast-16"]
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert await data(http) == EMPTY


@pytest.mark.parametrize(("task_id", "policy"), [(STORM_TASK, "rolling"), (STORM_TASK, "naive"),
                                                 (BUST_TASK, "rolling")])
async def test_a_whole_wind_week_played_through_the_tools_scores_its_stored_reference(
        env, tools, http, gateway, task_id, policy):
    """The stored rolling plans, or the naive policy, confirmed through the tools watch by watch with the triggers'
    calls in between, score what the wind references hold; each watch's situation ends with the wind as known then."""
    task = PACK.get(task_id)
    reference = REFS[task_id]
    plans = reference["rolling"]["plans"]

    def target(week):
        return naive(week) if policy == "naive" else plan_from_list(plans[week.watch])

    assert (await invoke(http, LIVE_LOAD_URI, task_id=task_id)).status_code == 200
    await invoke(http, CLOCK_URI, env_get_time_url=f"{gateway.url}/clock/time")
    week = env.week
    hours = []
    while True:
        plan = target(week)
        if rows := [r for r in plan_to_list(plan) if week.plan.get(r["ship"]) != plan[r["ship"]]]:
            assert (await tools("confirm_berths", plan=rows))["refused"] == []
        situation_now = await tools("get_situation")
        hours.append(situation_now["hour"])
        assert situation_now["situation"].endswith(ending(wind.known(task, week.revealed)))
        assert wind.forecast(week.view)["hour"] == week.hour
        out = await tools("advance")
        if out.get("done"):
            break
        assert gateway.virtual_time == virtual_time(task, out["hour"])
        await deliver(tools, task, out["watch"])
    assert hours == reference["watch_hours"]
    stored = reference[policy]
    assert {k: out[k] for k in ("feasible", "cost", "reward")} == {k: stored[k] for k in ("feasible", "cost", "reward")}
    got = await data(http)
    assert (got["grade"]["excused_cost"], got["audit"]) == (stored["excused_cost"], {"ok": True, "problems": []})
    assert got["grade"] == play(task, target, week=WindWeek).grade


async def test_main_serves_the_wind_env_under_its_name_with_the_wind_pack_in_its_tasks_dir(tmp_path):
    view = WindWeek(TASK).view
    with served("portsim-wind", tasks_dir(tmp_path)) as base:
        card_now = httpx.get(f"{base}/.well-known/agent-env.json").json()
        assert (card_now["name"], {t["name"] for t in card_now["capabilities"]["tools"]}) == (
            "portsim-wind", {"get_situation", "check_plan", "confirm_berths", "advance", "port_notice"})
        await client.invoke_extension(base, card_now, LIVE_LOAD_URI, {"task_id": STORM_TASK})
        async with Client(f"{base}/mcp") as mcp:
            out = json.loads((await mcp.call_tool_mcp("get_situation", {})).content[0].text)
    assert out["situation"] == situation(view) + ending(view)
