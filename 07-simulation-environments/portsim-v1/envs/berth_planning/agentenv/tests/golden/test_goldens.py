"""The env answers as upstream's does where the published transcripts never go (upstream.json, recorded by the
agentenv-portsim plugin's scripts/record_goldens.py from ../openenv at b0f4c2f)."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.anyio

GOLDEN = json.loads(Path(__file__).with_name("upstream.json").read_text())


async def test_tools_list(harness):
    async with harness.session() as session:
        tools = (await session.list_tools()).tools
    assert [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools] == GOLDEN["tools"]


@pytest.mark.parametrize("episode", GOLDEN["episodes"], ids=lambda e: e["name"])
async def test_episode(harness, episode):
    await harness.load_task(episode["task_id"])
    async with harness.session() as session:
        for call in episode["calls"]:
            assert await harness.call(session, call["tool"], call["arguments"]) == call["output"]
            assert (await harness.data())["done"] == call["done"]
    data = await harness.data()
    assert {key: data[key] for key in episode["end"]} == episode["end"]
