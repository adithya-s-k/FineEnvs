"""The env grades each published final plan to its published reward and grade, and each stored optimal plan to 1.0,
each submitted through submit_plan as an MCP client calls it."""

import json

import pytest
from fastmcp import Client

from berth_agentenv.server import PortSimEnv

pytestmark = pytest.mark.anyio


class Submit:
    def __init__(self, env: PortSimEnv, client: Client):
        self.env, self.client = env, client

    async def __call__(self, task_id: str, plan) -> tuple[float, dict]:
        self.env.load_task(task_id)
        result = await self.client.call_tool_mcp("submit_plan", {"plan": plan})
        assert not result.isError, result.content[0].text
        return self.env.reward, self.env.grade


@pytest.fixture
async def submit():
    env = PortSimEnv()
    async with Client(env.create_app()) as client:
        yield Submit(env, client)


@pytest.mark.network
async def test_published_plans(episodes, submit):
    published, graded = {}, {}
    for episode in episodes:
        if episode["submitted"]:
            key = episode["model"], episode["task_id"]
            published[key] = episode["reward"], json.loads(episode["grade"])
            graded[key] = await submit(episode["task_id"], json.loads(episode["final_plan"]))
    assert len(published) == 202
    assert {k: (graded[k], v) for k, v in published.items() if graded[k] != v} == {}


async def test_optimal_plans(submit):
    rewards = {task.task_id: (await submit(task.task_id, task.reference["optimal_plan"]))[0]
               for task in submit.env.pack.tasks}
    assert len(rewards) == 1100
    assert {task_id: r for task_id, r in rewards.items() if r != 1.0} == {}
