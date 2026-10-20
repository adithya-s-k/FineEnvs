"""The verifier against a stand-in env's data/get, scored the way the env_outcome_verifier step scores it: the day's
reward, and no score at all for a day that never finished, broke the reveal schedule or has no reference."""

import asyncio
import socket
import types
from contextlib import asynccontextmanager

import pytest
import uvicorn
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agentenv_protocol import AgentEnvEnvironment, DataPart, environment_card, get_data
from airops_core.core.model import load_pack
from airops_core.day import Day
from paths import FIXTURES

from airops_agentenv.tasks import VERIFIERS

pytestmark = pytest.mark.anyio


def load_verifier() -> types.ModuleType:
    """The verifier as the step runs it, compiled without leaving bytecode beside it."""
    path = VERIFIERS / "airops-verifier" / "verify.py"
    module = types.ModuleType("airops_verifier")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


verifier = load_verifier()


def ended(**changes) -> dict:
    task = load_pack(FIXTURES)[0]
    day = Day(task)
    day.end()
    return day.data() | changes


@environment_card(name="airops-day")
class Stand(AgentEnvEnvironment):
    def __init__(self, answer: dict):
        self.answer = answer

    @get_data
    async def get(self) -> list[DataPart]:
        return [DataPart(data=self.answer)]


@asynccontextmanager
async def served(env: AgentEnvEnvironment):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(env.create_app().streamable_http_app(), host="127.0.0.1", port=port,
                                           log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await serving


@pytest.mark.parametrize("reward", [1.0, 0.8116, 0.2])
async def test_the_score_is_the_days_reward_and_the_validity_row_weighs_nothing(reward):
    data = ended(reward=reward)
    async with served(Stand(data)) as mcp_url:
        rows = await verifier.verify(mcp_url)
    assert aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE) == reward
    assert [(r["result"], r["score"], r.get("weight", 1)) for r in rows] == [(reward >= 1.0, reward, 1),
                                                                             (True, 1.0, 0)]
    assert rows[0]["episode"] == data


@pytest.mark.parametrize("changes", [
    {"done": False, "end_reason": None, "grade": None, "reward": None},
    {"audit": {"ok": False, "problems": ["adv-1 arrived after the planner's next call in watch 4"]}},
    {"reward": None},
])
async def test_a_day_that_never_finished_broke_the_reveal_schedule_or_has_no_reference_is_not_scored(changes):
    async with served(Stand(ended(**changes))) as mcp_url:
        with pytest.raises(RuntimeError, match="the day can't be scored: done"):
            await verifier.verify(mcp_url)
