"""Score a PortSim episode with the reward the env gave its submitted plan: 0 when nothing was submitted."""

from agentenv_protocol import client


async def verify(mcp_url: str) -> list[dict]:
    episode = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data
    reward = episode["reward"] or 0.0
    return [{"criterion": "the submitted plan's reward, 1.0 at the CP-SAT optimum", "result": reward >= 1.0,
             "score": reward, "episode": episode}]
