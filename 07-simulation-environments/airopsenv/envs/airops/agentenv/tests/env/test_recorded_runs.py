"""The recorded run in fixtures/runs, the dataset's sol-v1 run of b6-bos-20250216-1500 (GPT-6.1 Sol, reward 0.672),
played back through the env over MCP (recorded.py) and through a Day in process (airops_agentenv.replay): every output
byte for byte, and the graded day the one its verifier read. In process, or in a container of the image with
AIROPS_IMAGE set."""

import json

import httpx
import pytest
from airops_core.core.model import load_pack
from fastmcp import Client
from paths import PACK, RUNS
from recorded import episodes, replay
from serving import served

from airops_agentenv.replay import replay as day_replay

pytestmark = pytest.mark.anyio
EPISODES = episodes(RUNS)
DAYS = {t.task_id: t for t in load_pack(PACK)}


def test_the_fixtures_hold_the_featured_v1_run():
    assert [(e.run, e.row["model"], e.row["task_id"], e.row["reward"]) for e in EPISODES] == [
        ("sol-v1", "openai/gpt-6.1-sol", "b6-bos-20250216-1500", 0.672431)]


@pytest.mark.parametrize("episode", EPISODES, ids=[e.row["task_id"] for e in EPISODES])
async def test_a_recorded_day_replays_byte_for_byte_over_mcp(episode, gateway):
    with served("airops_agentenv.server", {"AIROPS_TASKS_DIR": str(PACK)}) as base:
        async with Client(f"{base}/mcp") as mcp, httpx.AsyncClient(base_url=base) as http:
            counts = await replay(mcp, http, DAYS[episode.row["task_id"]], episode, f"{gateway.env_url}/clock/time")
    assert (counts.outputs, counts.notices, counts.skipped, counts.refusals) == (44, 20, 8, 0)


@pytest.mark.parametrize("episode", EPISODES, ids=[e.row["task_id"] for e in EPISODES])
def test_a_recorded_day_replays_through_a_day_to_the_graded_day(episode):
    played = day_replay(DAYS[episode.row["task_id"]], episode.transcript["messages"])
    assert json.loads(json.dumps(played.day.data())) == episode.verified
    assert played.day.grade["reward"] == episode.row["reward"]
