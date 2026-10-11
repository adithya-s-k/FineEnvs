import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastmcp import Client
from port_week_core import packs
from port_week_core.schedule import ONE_WEEK
from serving import IMAGE
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from port_week_agentenv.server import PortSimLiveEnv

EXAMPLE = "dock-24B-w07x1-busy-0"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def pack():
    return packs.live_pack()


@pytest.fixture(scope="session")
def one_week(pack):
    return [t for t in pack.tasks if ONE_WEEK.match(t.task_id)]


@pytest.fixture(scope="session")
def example(pack):
    return pack.get(EXAMPLE)


class FakeGateway:
    """The gateway's two clock routes the env uses: it records each PUT and reads back the last time set."""

    def __init__(self):
        self.puts: list[dict] = []
        self.virtual_time = "2024-02-12T00:00:00Z"
        self.app = Starlette(routes=[Route("/clock/time", self.time), Route("/clock/set-time", self.set_time,
                                                                             methods=["PUT"])])

    async def time(self, request):
        return JSONResponse({"virtual_time": self.virtual_time})

    async def set_time(self, request):
        body = await request.json()
        self.puts.append(body)
        self.virtual_time = body["virtual_time"]
        return JSONResponse({"armed": True, "t0": body["virtual_time"],
                             "virtual_seconds_per_real_second": body["virtual_seconds_per_real_second"]})


@pytest.fixture(scope="session")
def fake_gateway():
    """On loopback; on every interface when the env runs in a container, which reaches it as host.docker.internal."""
    gateway = FakeGateway()
    sock = socket.socket()
    sock.bind(("0.0.0.0" if IMAGE else "127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(gateway.app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    port = sock.getsockname()[1]
    gateway.url = f"http://127.0.0.1:{port}"
    gateway.env_url = f"http://host.docker.internal:{port}" if IMAGE else gateway.url
    yield gateway
    server.should_exit = True
    thread.join(10)


@pytest.fixture
def gateway(fake_gateway):
    fake_gateway.puts.clear()
    fake_gateway.virtual_time = "2024-02-12T00:00:00Z"
    return fake_gateway


class Tools:
    """Calls tools as an MCP client does, through argument validation and the env's middleware; results are JSON."""

    def __init__(self, client: Client):
        self.client = client

    async def __call__(self, tool: str, /, **arguments) -> dict:
        result = await self.client.call_tool_mcp(tool, arguments)
        assert not result.isError, result.content[0].text
        return json.loads(result.content[0].text)

    async def error(self, tool: str, /, **arguments) -> str:
        result = await self.client.call_tool_mcp(tool, arguments)
        assert result.isError
        return result.content[0].text


@pytest.fixture
def env():
    env = PortSimLiveEnv()
    env.create_app()
    return env


@pytest.fixture
async def tools(env):
    async with Client(env.mcp) as mcp:
        yield Tools(mcp)


@pytest.fixture
async def http(env):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()), base_url="http://portsim") as http:
        yield http


@pytest.fixture
async def live(env, gateway):
    """The example week loaded on an env synced to the fake gateway's clock."""
    env.live_load(EXAMPLE)
    await env.sync_time(f"{gateway.url}/clock/time")
    return env.week
