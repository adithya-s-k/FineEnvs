"""Check the remote training API without launching a sandbox or inference request."""
import asyncio
import json


def validate_tools(response, arm):
    tools = response.get("data", {}).get("observation", {}).get("tools", [])
    tool = next((item for item in tools if item.get("name") == "run_rollout"), None)
    if tool is None:
        raise ValueError("Environment did not advertise run_rollout")
    properties = tool.get("input_schema", {}).get("properties", {})
    required = {"sampling", "llm_url", "model"}
    required |= {"agent_timeout_sec", "agent_step_limit"} if arm == "blackbox" else {"require_tokens", "agent_timeout_s"}
    missing = required - properties.keys()
    if missing:
        raise ValueError("Deployed environment lacks training arguments: " + ", ".join(sorted(missing))
                         + ". Deploy the matching environment bundle before starting training.")
    return {"passed": True, "arm": arm, "arguments": sorted(properties)}


async def _probe(url, token, arm):
    from websockets.asyncio.client import connect
    url = url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    headers = {"Authorization": "Bearer " + token} if token else {}
    async with connect(url + "/ws", additional_headers=headers, open_timeout=30) as socket:
        await socket.send(json.dumps({"type": "step", "data": {"type": "list_tools"}}))
        response = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
        return validate_tools(response, arm)


def check(url, token, arm):
    return asyncio.run(_probe(url, token, arm))
