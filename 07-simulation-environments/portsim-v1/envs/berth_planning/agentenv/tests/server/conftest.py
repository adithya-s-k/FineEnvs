import httpx
import pytest
from fastmcp import Client

from berth_agentenv.server import PortSimEnv

TASK = "dock-24B-w07x1-busy-0"


@pytest.fixture
def env():
    env = PortSimEnv()
    env.create_app()
    env.load_task(TASK)
    return env


class Tools:
    """Calls tools as an MCP client does: through argument validation and the episode middleware."""

    def __init__(self, client: Client):
        self.client = client

    async def __call__(self, name: str, **arguments) -> str:
        result = await self.client.call_tool_mcp(name, arguments)
        assert not result.isError, result.content[0].text
        return result.content[0].text

    async def error(self, name: str, **arguments) -> str:
        result = await self.client.call_tool_mcp(name, arguments)
        assert result.isError
        return result.content[0].text


@pytest.fixture
async def tools(env):
    async with Client(env.mcp) as client:
        yield Tools(client)


@pytest.fixture
async def http(env):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(env.mcp.http_app()), base_url="http://portsim") as http:
        yield http
