"""A ship's agent in PortSim lines: the firm that deals with the port for a line, answering the planner's messages about
its ships. No model: each answer is the env's (ship_status, agree_arrival), called under the agent's own gateway role,
so a week replays the same way for the same messages. The firm is the agent's deployed name.

The calls go through the gateway's REST /step route, not an MCP session: the gateway (agentenv-framework 0.9.1306)
runs its MCP lifespan per session, so a second client's short session would replace and then close the env sessions
and the trigger driver the planner's session relies on."""

import json
import logging
import os

import httpx
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    a2a_agent,
)

log = logging.getLogger("portsim-line")
ROLE_HEADER = "AgentEnv-Role"
MAX_REQUESTS = 20
COMMIT_REFUSAL = "can't commit yet: "
USAGE = ('send {"requests": [{"ship": 7, "ask": "eta", "request_arrival": null}, {"ship": 9, "ask": null, '
         '"request_arrival": 130}]}: one entry per ship, asking for its ETA or requesting an arrival hour')


def requests_of(request: TaskRequest) -> list[dict]:
    for part in request.parts:
        if part.kind == "data" and isinstance(part.data.get("requests"), list):
            return part.data["requests"]
    for part in request.parts:
        if part.kind == "text":
            try:
                body = json.loads(part.text)
            except json.JSONDecodeError:
                continue
            if isinstance(body, dict) and isinstance(body.get("requests"), list):
                return body["requests"]
            if isinstance(body, list):
                return body
    raise ValueError(f"no requests in the message: {USAGE}")


def sentence(reply: dict) -> str:
    who = f"{reply['name']} (ship {reply['ship']})" if "name" in reply else f"ship {reply.get('ship')}"
    if "error" in reply:
        return f"{who}: {reply['error']}"
    if "arrived" in reply:
        return f"{who} arrived at hour {reply['arrived']}; its turn is hour {reply['turn']}."
    if reply.get("agreed_before"):
        return f"{who}: ETA hour {reply['eta']}, as agreed; its turn is hour {reply['turn']}."
    if reply.get("agreed") is True:
        return (f"{who} will arrive at hour {reply['arrival']}" + (", reducing speed" if reply.get("slowing") else "")
                + f"; it keeps its turn, hour {reply['turn']}.")
    if reply.get("agreed") is False and "eta" in reply:
        return (f"{who} can't commit to hour {reply.get('asked')} yet: "
                f"{reply['reason'].removeprefix(COMMIT_REFUSAL)}; ETA hour {reply['eta']}.")
    if reply.get("agreed") is False:
        offer = (f"; it can arrive at hour {reply['earliest']} at the earliest" if "earliest" in reply else
                 f"; it can arrive at hour {reply['latest']} at the latest" if "latest" in reply else "")
        kept = (f"; it still arrives at hour {reply['agreed_arrival']}, as agreed" if "agreed_arrival" in reply
                else "")
        return f"{who} can't make hour {reply.get('asked')}: {reply['reason']}{offer}{kept}."
    return f"{who}: ETA hour {reply['eta']}."


class Desk:
    """The env tools under the agent's role, through the gateway's /step route."""

    def __init__(self, http: httpx.AsyncClient, url: str, firm: str):
        self.http, self.url, self.firm = http, url, firm

    async def call(self, name: str, arguments: dict) -> dict:
        response = await self.http.post(self.url, json={"action": "call_tool", "tool_name": name,
                                                         "arguments": {"agent": self.firm, **arguments}})
        body = response.json()
        if response.status_code != 200:
            return {"error": body.get("error", f"HTTP {response.status_code}")}
        text = "\n".join(c["text"] for c in body.get("content", []) if c.get("type") == "text")
        return {"error": text} if body.get("isError") else json.loads(text)

    async def answer(self, entry) -> dict:
        if not isinstance(entry, dict) or "ship" not in entry:
            return {"error": f"each request names a ship: {USAGE}"}
        ship, ask, arrival = entry["ship"], entry.get("ask"), entry.get("request_arrival")
        if ask == "eta" and arrival is None:
            out = await self.call("ship_status", {"ship": ship})
            return out | ({"agreed_before": True} if out.get("agreed") else {})
        if ask is None and arrival is not None:
            out = await self.call("agree_arrival", {"ship": ship, "arrival": arrival})
            return out | {"asked": arrival}
        return {"ship": ship, "error": f"ask for the ETA or request an arrival, one per request: {USAGE}"}


def step_url(mcp_url: str) -> str:
    """The gateway's /step route beside its MCP route."""
    return mcp_url.rstrip("/").removesuffix("/mcp") + "/step"


@a2a_agent(
    identity=AgentIdentity(
        name="portsim-line",
        description="A ship's agent at the Port of Barcelona: reports and answers for its line's ships, agrees an "
                    "arrival time when a ship can make it, and otherwise offers the earliest or latest it can. Message "
                    "it with structured requests.",
        version="0.1.0",
    ),
    config=AgentConfig,
    extensions=(MCP_CONFIG_V1,),
)
class PortSimLine(AgentEnvAgent):
    async def run(self, request: TaskRequest[AgentConfig]) -> TaskResult:
        try:
            entries = requests_of(request)
        except ValueError as e:
            return TaskResult.builder().succeeded().add_text(str(e)).add_data({"replies": [], "error": str(e)}).build()
        if len(entries) > MAX_REQUESTS:
            text = f"at most {MAX_REQUESTS} requests a message"
            return TaskResult.builder().succeeded().add_text(text).add_data({"replies": [], "error": text}).build()
        [server] = request.mcp_servers.values()
        headers = {**(server.get("headers") or {}), ROLE_HEADER: request.config.role or "line"}
        async with httpx.AsyncClient(headers=headers, timeout=60) as http:
            desk = Desk(http, step_url(server["url"]), request.config.name or os.environ.get("PORTSIM_AGENT", ""))
            replies = [await desk.answer(entry) for entry in entries]
        return (TaskResult.builder().succeeded().add_text(" ".join(sentence(r) for r in replies))
                .add_data({"replies": replies}).build())


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    PortSimLine().serve()
