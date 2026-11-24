"""Score a live AirOps day with the reward of its final plan flown on the truth; a day that never finished, broke the
reveal schedule or has no reference raises, so the run is not scored."""

from agentenv_protocol import client


async def verify(mcp_url: str) -> list[dict]:
    day = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data
    if not day["done"] or not day["audit"]["ok"] or day["reward"] is None:
        raise RuntimeError(f"the day can't be scored: done {day['done']}, audit {day['audit']}, reward {day['reward']}")
    reward = day["reward"]
    return [{"criterion": "the final plan's reward on the day as it happened, 1.0 at the hindsight optimum",
             "result": reward >= 1.0, "score": reward, "episode": day},
            {"criterion": "every watch got its news, in order, before the planner's next call", "result": True,
             "score": 1.0, "weight": 0}]
