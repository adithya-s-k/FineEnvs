"""The recorded runs in fixtures/runs, copied from the agentenv-portsim plugin at v0.4.5 (tests/viewer), played back
through the env over MCP (recorded.py): two live weeks played by Sonnet and GPT through agent-env, and the scripted
marine and wind runs (the wind one on the synthetic wind weeks). In process, or one container per week with
PORTSIM_LIVE_IMAGE set: an env plays one week."""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from berth_core import TaskPack
from fastmcp import Client
from port_week_core import marine, packs
from recorded import episodes, replay
from serving import IMAGE, served
from wind_fixture import PACK_DIR, tasks_dir, use

from port_week_agentenv.contract import LIVE_ENV, MARINE_ENV, WIND_ENV
from port_week_agentenv.server import ENVS

pytestmark = pytest.mark.anyio

RUNS = Path(__file__).resolve().parent / "fixtures" / "runs"
EPISODES = episodes(RUNS)
PACKS = {LIVE_ENV: packs.live_pack, MARINE_ENV: marine.pack, WIND_ENV: lambda: TaskPack(PACK_DIR)}
REPLAYED = {("live-gpt", "openai/gpt-6.1-sol"): (9, 2, 4, 0),
            ("live-sonnet", "anthropic/claude-sonnet-5-5"): (16, 4, 1, 0),
            ("marine-scripted", "scripted/naive"): (24, 9, 0, 0),
            ("marine-scripted", "scripted/rolling"): (24, 9, 0, 0),
            ("wind-scripted", "scripted/blind"): (26, 15, 0, 0),
            ("wind-scripted", "scripted/rolling"): (28, 15, 0, 0)}
"""Per week: outputs compared, notices delivered, calls skipped (get_time; after the week was done) and refusals."""


@asynccontextmanager
async def env_for(name: str, monkeypatch, tmp_path):
    """The env's MCP client and HTTP client: a fresh env in process, or a fresh container."""
    if IMAGE:
        with served(name, tasks_dir(tmp_path) if name == WIND_ENV else None) as base:
            async with Client(f"{base}/mcp") as mcp, httpx.AsyncClient(base_url=base) as http:
                yield mcp, http
        return
    if name == WIND_ENV:
        use(monkeypatch)
    env = ENVS[name]()
    env.create_app()
    async with Client(env.mcp) as mcp, httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()),
                                                         base_url="http://portsim") as http:
        yield mcp, http


def test_the_fixtures_hold_six_weeks():
    assert [(e.run, e.env, e.row["model"], e.row["task_id"]) for e in EPISODES] == [
        ("live-gpt", LIVE_ENV, "openai/gpt-6.1-sol", "dock-36A-w06x1-standard-0"),
        ("live-sonnet", LIVE_ENV, "anthropic/claude-sonnet-5-5", "dock-24B-w06x1-busy-0"),
        ("marine-scripted", MARINE_ENV, "scripted/naive", "dock-24B-w07x1-busy-0"),
        ("marine-scripted", MARINE_ENV, "scripted/rolling", "dock-24B-w07x1-busy-0"),
        ("wind-scripted", WIND_ENV, "scripted/blind", "dock-24B-w07x1-busy-0-e00"),
        ("wind-scripted", WIND_ENV, "scripted/rolling", "dock-24B-w07x1-busy-0-e00")]


@pytest.mark.parametrize("episode", EPISODES, ids=[f"{e.run}-{e.row['model']}" for e in EPISODES])
async def test_a_recorded_week_replays_byte_for_byte(episode, gateway, monkeypatch, tmp_path):
    task = PACKS[episode.env]().get(episode.row["task_id"])
    async with env_for(episode.env, monkeypatch, tmp_path) as (mcp, http):
        counts = await replay(mcp, http, task, episode, f"{gateway.env_url}/clock/time")
    replayed = (counts.outputs, counts.notices, counts.skipped, counts.refusals)
    assert replayed == REPLAYED[episode.run, episode.row["model"]]
