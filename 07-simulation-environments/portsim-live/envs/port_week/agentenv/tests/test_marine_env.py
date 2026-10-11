"""PortSimMarineEnv in-process: the live env's tools, extensions and data plane on the marine weeks, the pilots and tugs
in get_situation and check_plan, whole marine weeks played through the tools, and main() serving it by its name."""

import json

import httpx
import pytest
from agentenv_protocol import client
from berth_core import load_pack, plan_from_list, plan_to_list, situation
from fastmcp import Client
from port_week_core import marine
from port_week_core.marine import MarineWeek
from port_week_core.schedule import ONE_WEEK, schedule, virtual_time
from port_week_core.world import naive, play
from serving import served

from port_week_agentenv.contract import CLOCK_URI, END_WEEK_URI, LIVE_LOAD_URI, triggers
from port_week_agentenv.server import EMPTY, PortSimLiveEnv, PortSimMarineEnv

pytestmark = pytest.mark.anyio

TASK_ID = "dock-24B-w07x1-busy-0"
PACK = marine.pack()
TASK = PACK.get(TASK_ID)
TASK_VIEW = MarineWeek(TASK).view
V2 = load_pack().get(TASK_ID)
SHORT = "tugs short at hour 54: your ships need 3, 2 free"


@pytest.fixture
def env():
    env = PortSimMarineEnv()
    env.create_app()
    return env


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


def v2_optimum() -> list[dict]:
    """dock-v1-eval's optimum for the ships known at hour 0."""
    return [r for r in V2.reference["optimal_plan"] if r["ship"] < len(TASK.ships) - 1]


async def test_the_marine_env_is_the_live_env_under_its_own_name(env, tools):
    live = PortSimLiveEnv()
    live.create_app()
    ours, theirs = await card(env), await card(live)
    assert (ours["name"], theirs["name"], env.mcp.name) == ("portsim-marine", "portsim-live", "portsim-marine")
    assert {e["uri"] for e in ours["capabilities"]["extensions"]} == {LIVE_LOAD_URI, END_WEEK_URI, CLOCK_URI}
    assert ours["capabilities"]["tools"] == theirs["capabilities"]["tools"]
    assert ours["capabilities"]["operations"] == ["data/reset", "data/get"]
    assert sorted(route.path for route in env.mcp.http_app().routes) == sorted(
        route.path for route in live.mcp.http_app().routes)
    assert [t.name for t in await tools.client.list_tools()] == [
        "get_situation", "check_plan", "confirm_berths", "advance", "port_notice"]


async def test_live_load_starts_a_week_of_the_marine_pack(env, http):
    other = next(t.task_id for t in load_pack().tasks if t.split == "train" and ONE_WEEK.match(t.task_id))
    refused = await invoke(http, LIVE_LOAD_URI, task_id=other)
    assert (refused.status_code, refused.json()["error"]["message"]) == (500, f"unknown task id {other!r}")
    assert (await invoke(http, LIVE_LOAD_URI, task_id=TASK_ID)).json() == {"task_id": TASK_ID, "watches": 7}
    assert type(env.week) is MarineWeek and env.week.task == TASK
    assert env.week.task.rules["marine"]["pilots"] == 7 and env.week.task.reference["optimal_cost"] == 226


async def test_get_situation_ends_with_the_pilots_and_tugs(live, tools):
    out = await tools("get_situation")
    assert out["situation"] == situation(live.view) + "\n\n" + marine.section(live.view)
    lines = out["situation"].splitlines()
    assert lines[lines.index("## Pilots and tugs") + 3] == (
        "- Your ships take 3 tugs: ships 3, 8; 2 tugs: ships 0, 2, 5, 9, 10, 15; 1 tug: ships 1, 4, 6, 7, 11, 12, 13, "
        "14.")
    assert "- Mon: 745565 266666 764766 354136 | 867675 286888 867888 854765" in lines
    assert lines[-1] == "- From hour 264: 7 pilots and 8 tugs free."
    assert not [line for line in lines if " out from hour " in line or "one more tug" in line]


async def test_check_plan_names_the_hours_short_of_pilots(live, tools):
    out = await tools("check_plan", plan=[{"ship": 0, "berth_hour": 7, "section": 2, "cranes": 5},
                                          {"ship": 1, "berth_hour": 21, "section": 4, "cranes": 3}])
    short = "pilots short at hour 21: your ships need 2, 1 free"
    assert [(r["ship"], r["problems"]) for r in out["ships"][:2]] == [(0, [short]), (1, [short])]
    assert out["feasible"] is False and live.plan == {}


async def test_the_tug_company_and_the_pilot_station_cut_the_pools_and_check_plan_names_the_short_hour(live, tools):
    assert (await tools("check_plan", plan=v2_optimum()))["feasible"]
    await tools("confirm_berths", plan=v2_optimum())
    assert (await tools("advance"))["watch"] == 1
    await deliver(tools, TASK, 1)
    out = await tools("get_situation")
    assert [(m["from"], m["text"]) for m in out["messages"]] == [
        ("MAERSK NUBA", TASK.notices[1]),
        ("Tug company", "2 of the port's 8 tugs are out of service from hour 54 to hour 78: 6 tugs in that window."),
        ("Pilot station", "2 of the port's 7 pilots on duty are unavailable from hour 54 to hour 66: 5 pilots in that "
                          "window.")]
    assert out["situation"].endswith(marine.section(live.view))
    assert [line for line in out["situation"].splitlines() if " out from hour " in line] == [
        "- Tug company: 2 of the 8 tugs out from hour 54 to hour 78.",
        "- Pilot station: 2 of the 7 pilots out from hour 54 to hour 66."]
    checked = await tools("check_plan", plan=[])
    assert [(r["ship"], r["problems"]) for r in checked["ships"] if r["problems"]] == [
        (4, [SHORT]), (5, [SHORT]), (16, ["missing from the plan"])]
    assert live.excused_problems == [] and live.audit() == {"ok": True, "problems": []}


async def test_end_week_grades_with_the_pilots_and_tugs_and_data_get_reports_it(live, tools, http):
    await tools("confirm_berths", plan=v2_optimum())
    await tools("advance")
    await deliver(tools, TASK, 1)
    expected = MarineWeek(TASK)
    expected.confirm(v2_optimum())
    expected.open()
    for n in expected.watches[1].notices:
        expected.notice(n.event_id, n.name, n.text, "trigger")
    expected.end()
    out = (await invoke(http, END_WEEK_URI)).json()
    assert out == {"end_reason": "end_week", "watch": 6, "feasible": False, "cost": None,
                   "reward": expected.grade["reward"], "audit": {"ok": True, "problems": []}}
    got = await data(http)
    assert set(got) == set(EMPTY) and "optimal_plan" not in str(got) and "movements" not in str(got)
    assert got["grade"] == expected.grade and got["grade"]["optimal_cost"] == 226
    assert got["grade"]["violations"] == [{"ship": 4, "problem": SHORT}, {"ship": 5, "problem": SHORT},
                                          {"ship": 16, "problem": "missing from the plan"}]
    assert [n["event_id"] for n in got["notices"]] == [
        "closure-2", "extra-1", "tug_outage-8", "pilot_shortage-9", "emergency-7", "late-3", "late-4",
        "crane_outage-6", "gale-0", "bunching-5"]
    await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
    assert await data(http) == EMPTY


@pytest.mark.parametrize(("task_id", "policy"), [(TASK_ID, "rolling"), (TASK_ID, "naive"),
                                                 ("dock-36A-w35x1-standard-0", "rolling")])
async def test_a_whole_marine_week_played_through_the_tools_scores_its_stored_reference(
        env, tools, http, gateway, task_id, policy):
    """The stored rolling plans, or the naive policy, confirmed through the tools watch by watch with the triggers'
    calls in between, score what the marine references hold."""
    task = PACK.get(task_id)
    reference = next(r for r in marine.references() if r["task_id"] == task_id)
    plans = reference["rolling"]["plans"]

    def target(week):
        return naive(week) if policy == "naive" else plan_from_list(plans[week.watch])

    assert (await invoke(http, LIVE_LOAD_URI, task_id=task_id)).status_code == 200
    await invoke(http, CLOCK_URI, env_get_time_url=f"{gateway.url}/clock/time")
    week = env.week
    while True:
        plan = target(week)
        if rows := [r for r in plan_to_list(plan) if week.plan.get(r["ship"]) != plan[r["ship"]]]:
            assert (await tools("confirm_berths", plan=rows))["refused"] == []
        situation_now = await tools("get_situation")
        assert situation_now["situation"].endswith(marine.section(week.view))
        out = await tools("advance")
        if out.get("done"):
            break
        assert gateway.virtual_time == virtual_time(task, out["hour"])
        await deliver(tools, task, out["watch"])
    wind = [line for line in situation_now["situation"].splitlines() if line.endswith("takes one more tug.")]
    assert wind == [f"- From hour {a} to hour {b} (wind above 25 kn), each movement takes one more tug."
                    for a, b in marine.wind(task)]
    stored = reference[policy]
    assert {k: out[k] for k in ("feasible", "cost", "reward")} == {k: stored[k] for k in ("feasible", "cost", "reward")}
    got = await data(http)
    assert (got["grade"]["excused_cost"], got["audit"]) == (stored["excused_cost"], {"ok": True, "problems": []})
    assert got["grade"] == play(task, target, week=MarineWeek).grade


async def test_main_serves_the_marine_env_under_its_name():
    with served("portsim-marine") as base:
        card_now = httpx.get(f"{base}/.well-known/agent-env.json").json()
        assert (card_now["name"], {t["name"] for t in card_now["capabilities"]["tools"]}) == (
            "portsim-marine", {"get_situation", "check_plan", "confirm_berths", "advance", "port_notice"})
        await client.invoke_extension(base, card_now, LIVE_LOAD_URI, {"task_id": TASK_ID})
        async with Client(f"{base}/mcp") as mcp:
            out = json.loads((await mcp.call_tool_mcp("get_situation", {})).content[0].text)
    assert out["situation"] == situation(TASK_VIEW) + "\n\n" + marine.section(TASK_VIEW)
