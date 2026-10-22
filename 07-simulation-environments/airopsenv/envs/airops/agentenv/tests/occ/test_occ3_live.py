"""airops-occ3 served over HTTP: strict schemas on every tool and the customer desk's three, three airops-desk agents
(crew, customer and ops) playing the day with its passengers at once through the long-poll, their desks taken from their
role headers, and data/get holding the same day the scripts play in process; and the customer desk's script."""

import asyncio

import httpx
import pytest
from agentenv_protocol import client
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack3
from fastmcp import Client
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from occ_helpers import desk_agent, done, play, scripted, serve, storm3, strict, write_pack3
from starlette.responses import JSONResponse

from airops_agentenv.occ import live
from airops_agentenv.occ.live import CLOCK_URI, LOAD_URI
from airops_agentenv.occ.live3 import OccEnv3

pytestmark = pytest.mark.anyio


@pytest.fixture
def pack(tmp_path, monkeypatch):
    path = write_pack3(tmp_path / "pack", storm3())
    monkeypatch.setenv("AIROPS_OCC3_PACK", str(path))
    return path


async def test_every_tool_of_the_env_is_strict_and_the_customer_desk_has_three(pack):
    async with Client(OccEnv3().create_app()) as c:
        listed = await c.list_tools()
    assert {t.name: strict(t.inputSchema, t.name) for t in listed} == {t.name: [] for t in listed}
    names = {t.name for t in listed}
    assert len(listed) == 23 and {"cust_flights", "cust_flight", "cust_request"} <= names
    assert next(t for t in listed if t.name == "ops_request").inputSchema["properties"]["to"]["enum"] == ["crew",
                                                                                                          "customer"]


async def test_three_desks_play_the_day_at_once_through_the_long_poll(pack, monkeypatch):
    monkeypatch.setattr(live, "TICK", 0.05)
    env = OccEnv3()
    mcp = env.create_app()

    @mcp.custom_route("/clock/set-time", methods=["PUT"])
    async def set_time(request):
        return JSONResponse({"armed": True})

    @mcp.custom_route("/clock/time", methods=["GET"])
    async def get_time(request):
        return JSONResponse({"virtual_time": "2024-01-01T05:00:00Z"})

    world = Pack3(pack).world(storm3().task.task_id)
    with serve(mcp.http_app()) as url:
        async with httpx.AsyncClient(base_url=url) as http:
            card = (await http.get("/.well-known/agent-env.json")).json()
            assert card["name"] == "airops-occ3"
            load = await http.post(client.extension_params(card, LOAD_URI)["endpoint"],
                                   json={"task_id": world.task.task_id, "mode": "joint"})
            assert load.json()["mode"] == "joint" and load.json()["watches"] == 25
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
                    await episode.play(desk_agent.McpDesk(session), "go")
                return episode.result()

            *desks, _ = await asyncio.gather(desk("crew"), desk("customer"), desk("ops"), trigger())
            data = (await http.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"})).json()
    day = data["result"]["parts"][0]["data"]
    for r in desks:
        summary = r.parts[1].data["structured_output"]
        assert summary["end_reason"] == "done" and summary["cost_usd"] == 0 and summary["desk_turns"] > 10
    assert day["done"] and day["audit"]["ok"] and set(day["absent"]) == {"crew", "customer", "ops"}
    assert {t["kind"] for t in day["turns"]} >= {"crew", "customer", "ops"} and day["passengers"]["outcomes"]
    again = await asyncio.to_thread(play, world, "joint", scripted("crew", "customer", "ops"), True, OccDay3)
    assert day["digest"] == again.digest() and day["grade"]["cost"] == again.grade["cost"]


def test_the_customer_desks_script_holds_a_missed_connection_and_protects_a_stranded_flight():
    script, calls = desk_agent.Script("customer"), []
    page = ("# Customer desk, JetBlue, 2024-07-28, 10:00 (watch 7 of 37, customer turn); freeze line 11:00: holds go\n"
            "## Connections at risk\n"
            "f218 947 JFK-BZN 11:05: f80 MCO in 10:38 -13 12; misses 12, hold to 11:18 makes them, 12 have no later "
            "seat today\n"
            "f220 948 JFK-BUF 11:05: f81 MCO in 10:38 -13 4; misses 4, hold to 11:18 makes them, 4 have no later seat "
            "today\n"
            "f230 523 JFK-LAX 11:00 committed: f82 MCO in 10:08 -2 30; misses 30, hold to 11:02 makes them, 0 have no "
            "later seat today\n"
            "## Flights ops might cancel\n"
            "f32 1434 SJU-MCO 12:00 150(0/16) no crew; cancelled now: 1 same day / 149 later day / 0 refund; no later "
            "seat today 120\n"
            "f33 1435 SJU-MCO 13:00 150(0/16) no crew; cancelled now: 100 same day / 50 later day / 0 refund; no later "
            "seat today 10")

    async def call(name, args):
        calls.append((name, args))
        return {}
    done(script.turn(page, call))
    done(script.turn(page, call))
    assert calls == [("cust_request", {"type": "HOLD", "flight": "f218", "until": "11:18", "note": None}),
                     ("cust_request", {"type": "PROTECT", "flight": "f32", "until": None, "note": None})]
    ops = desk_agent.Script("ops")
    done(ops.turn("R4 from customer at 10:00: HOLD f218 until 11:18 (12 pax)\n"
                  "R5 from customer at 10:00: PROTECT f32 (150 pax)", call))
    assert calls[-1] == ("ops_reply", {"request": "R4", "answer": "accept", "text": None})
    assert desk_agent.Episode(desk_agent.DeskConfig(role="customer"), {}).desk == "customer"
