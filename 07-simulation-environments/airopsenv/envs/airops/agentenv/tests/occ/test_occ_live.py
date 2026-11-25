"""airops-occ served over HTTP: strict schemas on every tool, each role seeing its desk's tools only, and two
airops-desk agents playing the day at once through the long-poll, their desks taken from their role headers, the clock
moved at each watch, the advisories delivered as the trigger delivers them, and data/get holding the same day the
scripts play in process."""

import asyncio
import json

import httpx
import pytest
from agentenv_protocol import client
from fastmcp import Client
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from occ_helpers import desk_agent, play, scripted, serve, storm, strict, write_pack
from starlette.responses import JSONResponse

from airops_agentenv.occ import live
from airops_agentenv.occ.live import CLOCK_URI, LOAD_URI, OccEnv

pytestmark = pytest.mark.anyio


@pytest.fixture
def pack(tmp_path, monkeypatch):
    path = write_pack(tmp_path / "pack", storm())
    monkeypatch.setenv("AIROPS_OCC_PACK", str(path))
    return path


async def test_every_tool_is_strict_and_the_card_has_the_extensions(pack):
    env = OccEnv()
    async with Client(env.create_app()) as c:
        listed = await c.list_tools()
    assert {t.name: strict(t.inputSchema, t.name) for t in listed} == {t.name: [] for t in listed}
    assert strict({"type": "string", "maxLength": 4, "default": "x"}, "t") == ["t: default is not supported",
                                                                               "t: maxLength is not supported"]
    assert len(listed) == 20 and all(len(t.description) < 600 for t in listed)
    move = next(t for t in listed if t.name == "crew_confirm").inputSchema["properties"]["moves"]["items"]
    assert move["required"] == ["action", "unit", "flight", "other", "kind", "time", "ref"]


async def test_two_desks_play_the_day_at_once_through_the_long_poll(pack, monkeypatch):
    monkeypatch.setattr(live, "TICK", 0.05)
    env = OccEnv()
    mcp = env.create_app()
    puts = []

    @mcp.custom_route("/clock/set-time", methods=["PUT"])
    async def set_time(request):
        puts.append((await request.json())["virtual_time"])
        return JSONResponse({"armed": True})

    @mcp.custom_route("/clock/time", methods=["GET"])
    async def get_time(request):
        return JSONResponse({"virtual_time": "2024-01-01T05:00:00Z"})

    world = storm()
    with serve(mcp.http_app()) as url:
        async with httpx.AsyncClient(base_url=url) as http:
            card = (await http.get("/.well-known/agent-env.json")).json()
            load = await http.post(client.extension_params(card, LOAD_URI)["endpoint"],
                                   json={"task_id": world.task.task_id, "mode": "joint"})
            assert load.json() == {"task_id": world.task.task_id, "mode": "joint", "watches": 25, "max_turns": 160}
            await http.post(client.extension_params(card, CLOCK_URI)["endpoint"],
                            json={"env_get_time_url": f"{url}/clock/time"})

            async def trigger():
                while not env.day.done:
                    for n in list(env.day.pending):
                        env.day.notice(n, "trigger")
                        await env.after()
                    await asyncio.sleep(0.02)

            async def desk(role: str):
                episode = desk_agent.Episode(desk_agent.DeskConfig(role=role), {"AIROPS_DESK_POLICY": "script"})
                async with httpx.AsyncClient(headers={"AgentEnv-Role": role}, timeout=60) as h, \
                        streamable_http_client(f"{url}/mcp", http_client=h) as (read, write, _), \
                        ClientSession(read, write) as session:
                    await session.initialize()
                    if role == "ops":
                        assert "crew_confirm" in {t.name for t in (await session.list_tools()).tools}
                    await episode.play(desk_agent.McpDesk(session), "go")
                return episode.result()

            ops, crew, _ = await asyncio.gather(desk("ops"), desk("crew"), trigger())
            data = (await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})).json()
    day = data["result"]["parts"][0]["data"]
    for r in (ops, crew):
        summary = r.parts[1].data["structured_output"]
        assert summary["end_reason"] == "done" and summary["cost_usd"] == 0 and summary["desk_turns"] > 10
    assert day["done"] and day["audit"]["ok"] and day["absent"] == {"crew": {**day["absent"]["crew"], "reason": "done"},
                                                                    "ops": {**day["absent"]["ops"], "reason": "done"}}
    assert len(puts) == day["watch"] and puts[0] == "2024-01-01T06:00:00Z"
    again = await asyncio.to_thread(play, world, "joint", scripted("crew", "ops"))
    assert day["digest"] == again.digest() and json.dumps(day["grade"]["executed"]) == json.dumps(
        again.grade["executed"])
