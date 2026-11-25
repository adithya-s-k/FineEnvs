"""Score an ops control center day with passengers with the reward of its plan on the truth under cost P; a day that
never finished, failed its audit (news, turns, causality, roles, crew and passenger legality, the bound, each desk's
presence) or has no reward raises, so the run is not scored."""

import json

from agentenv_protocol import client


async def verify(mcp_url: str) -> list[dict]:
    day = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data
    if not day["done"] or not day["audit"]["ok"] or day["reward"] is None:
        raise RuntimeError("the day can't be scored: " + json.dumps(
            {"done": day["done"], "problems": day["audit"]["problems"], "reward": day["reward"]}))
    grade = day["grade"]
    out = [{"criterion": "the day's reward on the truth under cost P, 1.0 at the best plan found in hindsight",
            "result": grade["reward"] >= 1.0, "score": grade["reward"], "episode": day}]
    if grade["reward_L"] is not None:
        out.append({"criterion": "the same reward against the capacity-aware lower bound",
                    "result": grade["reward_L"] >= 1.0, "score": grade["reward_L"], "weight": 0})
    return out + [{"criterion": "the audit passed", "result": True, "score": 1.0, "weight": 0}]
