"""The card, the two extensions and the data plane, over the env's HTTP app in-process."""

import json

import pytest
from agentenv_protocol import client
from berth_core import parse_plan, plan_to_list

pytestmark = pytest.mark.anyio

LOAD_TASK, SUBMIT_PLAN = "urn:portsim:load-task/v1", "urn:portsim:submit-plan/v1"
EMPTY = {"checks_used": 0, "calls_used": 0, "done": False, "end_reason": None, "plan": None, "grade": None,
         "reward": None}


async def invoke(http, uri: str, **params):
    card = (await http.get("/.well-known/agent-env.json")).json()
    return await http.post(client.extension_params(card, uri)["endpoint"], json=params)


async def rpc(http, method: str) -> dict:
    return (await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": method})).json()["result"]


async def data(http) -> dict:
    return (await rpc(http, "data/get"))["parts"][0]["data"]


async def test_card(http):
    card = (await http.get("/.well-known/agent-env.json")).json()
    assert card["name"] == "portsim" and client.mcp_path(card) == "/mcp"
    assert card["capabilities"]["operations"] == ["data/reset", "data/get"]
    assert {e["uri"] for e in card["capabilities"]["extensions"]} == {LOAD_TASK, SUBMIT_PLAN}
    assert {t["name"] for t in card["capabilities"]["tools"]} == {"get_situation", "check_plan", "submit_plan"}


def test_nothing_else_is_served(env):
    assert sorted(route.path for route in env.mcp.http_app().routes) == [
        "/.well-known/agent-env.json", "/agentenv", "/agentenv/ext/load_task", "/agentenv/ext/submit", "/mcp"]


async def test_load_task_starts_a_new_episode(env, tools, http):
    await tools("check_plan", plan=[])
    other = next(t.task_id for t in env.pack.tasks if t is not env.task)
    assert (await invoke(http, LOAD_TASK, task_id=other)).json() == {"task_id": other}
    assert await data(http) == {"task_id": other, **EMPTY}


async def test_load_task_refuses_an_unknown_id(http):
    response = await invoke(http, LOAD_TASK, task_id="no-such-task")
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "unknown task id 'no-such-task'"


async def test_submit_plan_answers_as_the_tool_does(env, tools, http):
    plan = env.task.reference["naive_plan"]
    submitted = (await invoke(http, SUBMIT_PLAN, plan=plan)).json()
    assert (env.done, env.reward, env.calls_used) == (True, submitted["reward"], 0)
    again = await invoke(http, SUBMIT_PLAN, plan=json.dumps(plan))
    assert again.status_code == 500
    assert again.json()["error"]["message"] == "the episode is over: the plan was already submitted"
    env.load_task(env.task.task_id)
    assert json.loads(await tools("submit_plan", plan=plan)) == submitted


async def test_data_get_reports_the_episode(env, tools, http):
    assert await data(http) == {"task_id": env.task.task_id, **EMPTY}
    plan = env.task.reference["naive_plan"]
    await tools("check_plan", plan=[])
    await tools("submit_plan", plan=plan)
    assert await data(http) == {"task_id": env.task.task_id, "checks_used": 1, "calls_used": 2, "done": True,
                                "end_reason": "submitted", "plan": plan_to_list(parse_plan(env.task, plan)[0]),
                                "grade": env.grade, "reward": env.grade["reward"]}


async def test_data_reset_clears_the_episode(env, tools, http):
    await tools("check_plan", plan=[])
    await rpc(http, "data/reset")
    assert await data(http) == {"task_id": None, **EMPTY}
    assert await tools.error("check_plan", plan=[]) == "Error calling tool 'check_plan': no episode: call reset first"
