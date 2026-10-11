"""Score a live PortSim week with the reward of the week it executed; a week that never finished or broke the reveal
contract raises, so the run is not scored."""

from agentenv_protocol import client


async def verify(mcp_url: str) -> list[dict]:
    week = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data
    if not week["done"] or not week["audit"]["ok"]:
        raise RuntimeError(f"the week can't be scored: done {week['done']}, audit {week['audit']}")
    reward = week["reward"]
    return [{"criterion": "the executed week's reward, 1.0 at the hindsight optimum", "result": reward >= 1.0,
             "score": reward, "episode": week},
            {"criterion": "every watch got its notices, in order, before the agent's next call", "result": True,
             "score": 1.0, "weight": 0}]
