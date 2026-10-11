import json

import pytest
from berth_core import situation

pytestmark = pytest.mark.anyio

OVER = "the episode is over: the plan was already submitted"


def optimal(env) -> list[dict]:
    return env.task.reference["optimal_plan"]


async def test_tools_are_listed_in_upstream_order(tools):
    assert [t.name for t in await tools.client.list_tools()] == ["get_situation", "check_plan", "submit_plan"]


async def test_get_situation(env, tools):
    assert await tools("get_situation") == situation(env.task)


async def test_check_plan_reports_a_plan_without_grading_it(env, tools):
    out = json.loads(await tools("check_plan", plan=optimal(env)))
    assert out["feasible"] and out["cost"] == env.task.reference["optimal_cost"] and out["checks_left"] == 9
    assert len(out["ships"]) == len(env.task.ships) and "reward" not in out
    assert (env.done, env.reward) == (False, None)


async def test_a_check_is_used_before_the_plan_is_read(tools):
    assert json.loads(await tools("check_plan", plan="{")) == {
        "error": "plan is not valid JSON: Expecting property name enclosed in double quotes at position 1",
        "checks_left": 9}


async def test_entry_problems_make_a_plan_infeasible(env, tools):
    out = json.loads(await tools("check_plan", plan=[*optimal(env), {"ship": 99, "berth_hour": 0, "section": 1}]))
    assert out["feasible"] is False and out["violations"] == []
    assert out["entry_problems"] == ["unknown ship 99: use the ship ids from the table"]


async def test_the_eleventh_check_is_refused(env, tools):
    for left in range(9, -1, -1):
        assert json.loads(await tools("check_plan", plan=[]))["checks_left"] == left
    assert json.loads(await tools("check_plan", plan=[])) == {
        "error": "no checks left (you had 10); submit_plan when ready"}
    assert (env.checks_used, env.calls_used) == (10, 11)


async def test_failed_calls_count_toward_the_limit_but_use_no_check(env, tools):
    assert (await tools.error("check_plan")).startswith("1 validation error for call[check_plan]\nplan\n")
    assert await tools.error("no_such_tool") == "Unknown tool: 'no_such_tool'"
    assert (env.checks_used, env.calls_used) == (0, 2)


async def test_the_24th_call_ends_the_episode_with_reward_0(env, tools):
    for _ in range(23):
        await tools.error("check_plan")
    assert not env.done
    assert json.loads(await tools("check_plan", plan=[]))["checks_left"] == 9
    assert (env.done, env.end_reason, env.reward, env.grade) == (True, "tool_call_limit", 0.0, None)
    assert await tools.error("submit_plan", plan=optimal(env)) == f"Error calling tool 'submit_plan': {OVER}"
    assert await tools("get_situation") == situation(env.task)
    assert (env.reward, env.calls_used) == (0.0, 26)


async def test_submit_is_graded_once(env, tools):
    out = json.loads(await tools("submit_plan", plan=optimal(env)))
    assert out == {"submitted": True, "feasible": True, "reward": 1.0, "cost": env.grade["cost"],
                   "delay_cost": env.grade["delay_cost"], "moves": env.grade["moves"]}
    assert (env.done, env.end_reason, env.reward) == (True, "submitted", 1.0)
    assert await tools.error("submit_plan", plan=[]) == f"Error calling tool 'submit_plan': {OVER}"
    assert await tools.error("check_plan", plan=[]) == f"Error calling tool 'check_plan': {OVER}"
    assert (env.reward, env.grade["reward"], env.calls_used) == (1.0, 1.0, 3)


@pytest.mark.parametrize(("coerce", "reward"), [(lambda e: e | {"ship": float(e["ship"])}, 1.0),
                                                 (lambda e: e | {"cranes": True}, 0.023529)],
                         ids=["float-ship", "bool-cranes"])
async def test_the_reward_kept_is_the_one_submit_plan_returns_for_the_validated_plan(env, tools, coerce, reward):
    plan = [coerce(entry) for entry in optimal(env)]
    assert json.loads(await tools("submit_plan", plan=plan))["reward"] == env.reward == env.grade["reward"] == reward


async def test_an_infeasible_submit_lists_20_violations_at_most(env, tools):
    env.load_task(max(env.pack.tasks, key=lambda t: len(t.ships)).task_id)
    out = json.loads(await tools("submit_plan", plan=[]))
    assert (out["submitted"], out["feasible"], out["reward"], out["entry_problems"]) == (True, False, 0.0, [])
    assert out["violations"] == env.grade["violations"][:20] and len(env.grade["violations"]) > 20


async def test_an_unreadable_submit_scores_0(env, tools):
    out = json.loads(await tools("submit_plan", plan="not json"))
    assert (out["feasible"], out["reward"]) == (False, 0.0)
    assert out["entry_problems"] == ["plan is not valid JSON: Expecting value at position 0"]
    assert (env.done, env.plan) == (True, [])


async def test_no_tool_answers_before_a_task_is_loaded(env, tools):
    env.reset()
    assert await tools.error("get_situation") == "Error calling tool 'get_situation': no episode: call reset first"
