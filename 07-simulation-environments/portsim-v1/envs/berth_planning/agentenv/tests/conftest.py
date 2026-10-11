import json
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import httpx
import pytest
from agentenv_protocol import client
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@pytest.fixture
def anyio_backend():
    return "asyncio"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server():
    """The env server at PORTSIM_URL (CI points it at the image), else `python -m berth_agentenv.server`, as the
    image runs it, on a free port."""
    if url := os.environ.get("PORTSIM_URL"):
        yield url
        return
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "berth_agentenv.server"],
                            env={**os.environ, "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while True:
        try:
            httpx.get(f"{base}/.well-known/agent-env.json").raise_for_status()
            break
        except httpx.HTTPError:
            assert proc.poll() is None and time.monotonic() < deadline, "the env server did not start"
            time.sleep(0.1)
    yield base
    proc.terminate()
    proc.wait(10)


class Harness:
    """Drives the env server the way upstream's harness drives its env, over agent-env's client and MCP."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.card = httpx.get(f"{base_url}/.well-known/agent-env.json").json()

    async def load_task(self, task_id: str) -> None:
        await client.invoke_extension(self.base_url, self.card, "urn:portsim:load-task/v1", {"task_id": task_id})

    async def data(self) -> dict:
        return (await client.get_data(self.base_url)).parts[0].data

    @asynccontextmanager
    async def session(self):
        async with streamable_http_client(f"{self.base_url}/mcp") as (read, write, _), \
                ClientSession(read, write) as session:
            await session.initialize()
            yield session

    @staticmethod
    async def call(session: ClientSession, name: str, arguments: dict) -> str:
        """The tool's output as upstream's harness hands it to the model: its text, or {"error": text}."""
        result = await session.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return json.dumps({"error": text}) if result.isError else text


@pytest.fixture(scope="session")
def harness(server):
    return Harness(server)
