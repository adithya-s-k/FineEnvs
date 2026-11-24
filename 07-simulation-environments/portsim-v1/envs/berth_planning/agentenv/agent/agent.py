"""PortSimEnv's harness loop (../../openenv/berth_openenv/agent.py) as an A2A agent: one episode on the
env's MCP tools through agent-env's model endpoint, on the API upstream calls for the model's provider. Cost is usage
times PRICES, or the most a request could cost when its usage is lost (it failed, or the harness cut its stream); when
PORTSIM_MAX_COST_USD is set, an episode stops before a request that could take it past that. The summary also gives the
cost without the failed requests, which may not have been billed, and how many failed. An env that lists advance
is a live week in watches: the episode ends on advance's done, with no tool-call cap, or once a turn would start within
TIME_MARGIN of the time limit, the week then graded on the windows confirmed; Anthropic requests carry one rolling cache
breakpoint."""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import httpx2
import openai
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    PeerAgentsSetRequest,
    TaskRequest,
    TaskResult,
    Usage,
    a2a_agent,
    extension,
)
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client

log = logging.getLogger("portsim-llm")

REQUEST_TIMEOUT = 900.0
NOTES_CHARS = 8000
MAX_TOOL_CALLS = 24
TIME_MARGIN = 1200
HARNESS_CAP = "length (harness cap)"
PRICES = {  # USD per 1M tokens: input, output, cache read. LiteLLM public price map, 2026-10-06 and 2026-10-09.
    "anthropic/claude-opus-5-5": (4.00, 20.00, 0.40),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "anthropic/claude-haiku-5-5": (0.10, 0.50, 0.01),
    "openai/gpt-6-astra": (10.00, 50.00, 1.00),
    "openai/gpt-6.1-sol": (2.00, 10.00, 0.10),
    "openai/gpt-6-luna": (0.10, 0.50, 0.01),
    "fireworks_ai/kimi-k3": (3.00, 15.00, 0.30),
    "fireworks_ai/glm-5p3": (1.40, 4.40, 0.26),
    "fireworks_ai/glm-5p3-flash": (0.15, 0.50, 0.03),
    "fireworks_ai/qwen3p8-2p4t-a95b": (2.00, 6.00, 0.25),  # no Fireworks entry in the map: the other providers' price
    "fireworks_ai/deepseek-v4p1-flash": (0.30, 1.20, 0.006),
    "groq/qwen/qwen3.8-27b": (0.80, 4.00, 0.80),
    # The open models upstream evaluated, through the Hugging Face router (https://router.huggingface.co/v1): its
    # /v1/models prices for each provider, 2026-10-07, with cache reads charged as input.
    "Qwen/Qwen3.8-2.4T-A95B:together": (2.00, 6.00, 2.00),
    "Qwen/Qwen3.8-27B:ovhcloud": (0.47, 3.19, 0.47),
    "zai-org/GLM-5.3-Flash:baseten": (0.15, 0.50, 0.15),
    "zai-org/GLM-5.3:together": (1.40, 4.40, 1.40),
}
MAX_OUTPUT = {"groq/qwen/qwen3.8-27b": 16384}  # the provider's limit on one reply, under the task's max_tokens


@dataclass(frozen=True)
class Mode:
    steps: tuple[str, ...]
    last_turn: str
    no_call: str
    cut_off: str


V1 = Mode(("check_plan", "submit_plan"),
          "This is your last turn: call submit_plan now with your best plan.",
          "No tool was called. Plans only count through the tools: call check_plan to test a draft or submit_plan "
          "with your final plan.",
          "Your last turn ran out of output tokens before calling a tool. Your notes from it are above. Keep reasoning "
          "short now and call check_plan with your current draft, or submit_plan.")
LIVE = Mode(("check_plan", "confirm_berths", "advance"),
            "This is your last turn: confirm_berths now for every ship that still needs a window; the rest of the "
            "week then runs on your confirmed windows.",
            "No tool was called. Windows only count through the tools: confirm_berths to commit windows, advance to "
            "move to the next watch.",
            "Your last turn ran out of output tokens before calling a tool. Your notes from it are above. Keep "
            "reasoning short now and call check_plan with your current draft, or confirm_berths.")


class PortSimConfig(AgentConfig):
    model: str = "anthropic/claude-sonnet-5-5"
    system_prompt: str = ""
    max_turns: int = 12
    model_params: dict[str, Any] = {"max_tokens": 32000}
    effort: str | None = None


class ProviderError(Exception):
    """A Responses stream that reported a failure, or ended without a response."""


@dataclass
class Tool:
    name: str
    description: str | None
    input_schema: dict | None


MESSAGE_TOOL = "message_agent"
MESSAGE_DESCRIPTION = ("Message a ship's agent, the firm that deals with the port for its line: ask for its ships' "
                       "ETAs or request arrival hours, several of its ships in one message. Each request gets the "
                       "agent's reply.")


def message_schema(agents: list[str]) -> dict:
    """Every field required and the optional ones nullable, so a provider that enforces strict schemas (OpenAI's
    Responses API) still lets a request leave out the ETA query, and an ETA query the hour."""
    request = {"type": "object", "additionalProperties": False, "required": ["ship", "ask", "request_arrival"],
               "properties": {
                   "ship": {"type": "integer", "description": "ship id from the table"},
                   "ask": {"type": ["string", "null"], "enum": ["eta", None],
                           "description": '"eta" to ask for the ship\'s ETA; null when requesting an arrival'},
                   "request_arrival": {"type": ["integer", "null"],
                                       "description": "the hour you request the ship arrive at; null when asking "
                                                      "for its ETA"}}}
    return {"type": "object", "additionalProperties": False, "required": ["agent", "requests"], "properties": {
        "agent": {"type": "string", "enum": agents, "description": "the ship's agent, by its id"},
        "requests": {"type": "array", "minItems": 1, "items": request,
                     "description": '[{"ship": 7, "ask": "eta", "request_arrival": null}, '
                                    '{"ship": 9, "ask": null, "request_arrival": 130}, ...]'}}}


class EnvSession:
    """One MCP session on the env server: one episode; with peers, the ships' agents the planner can message."""

    def __init__(self, session: ClientSession, peers: dict[str, dict] | None = None,
                 http: httpx.AsyncClient | None = None):
        self.session, self.peers, self.http = session, peers or {}, http

    async def tools(self) -> list[Tool]:
        tools = [Tool(t.name, t.description, t.inputSchema) for t in (await self.session.list_tools()).tools]
        return tools + ([Tool(MESSAGE_TOOL, MESSAGE_DESCRIPTION, message_schema(sorted(self.peers)))]
                        if self.peers else [])

    async def call(self, name: str, arguments: dict) -> str:
        if name == MESSAGE_TOOL and self.peers:
            return await self.message(arguments)
        result = await self.session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return json.dumps({"error": text}) if result.isError else text

    async def message(self, arguments: dict) -> str:
        """One A2A message/send to the agent, blocking until it answers; its replies, or the error."""
        agent, requests = arguments.get("agent"), arguments.get("requests")
        if agent not in self.peers:
            return json.dumps({"error": f"no agent {agent!r}: message one of {', '.join(sorted(self.peers))}"})
        if not isinstance(requests, list) or not requests:
            return json.dumps({"error": "requests must be a non-empty list"})
        body = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send", "params": {"message": {
            "messageId": str(uuid.uuid4()), "role": "user", "metadata": {"from": "portsim-llm"},
            "parts": [{"kind": "data", "data": {"requests": requests}}]}}}
        try:
            response = await self.http.post(self.peers[agent]["url"].rstrip("/") + "/a2a", json=body, timeout=120)
            response.raise_for_status()
            reply = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"the agent {agent} didn't answer: {type(e).__name__}"})
        if "error" in reply:
            return json.dumps({"error": f"the agent {agent} failed: {reply['error'].get('message', reply['error'])}"})
        parts = reply["result"]["status"]["message"]["parts"]
        data = next((p["data"] for p in parts if p.get("kind") == "data" and "replies" in p.get("data", {})), {})
        text = " ".join(p["text"] for p in parts if p.get("kind") == "text")
        return json.dumps({"agent": agent, "replies": data.get("replies", []), "text": text}, ensure_ascii=False)


# ================================================================ model adapters

@dataclass
class Call:
    id: str
    name: str
    arguments: dict
    error: str | None = None
    raw: str = ""


@dataclass
class Turn:
    text: str
    reasoning: str
    calls: list[Call]
    stop: str
    usage: dict = field(default_factory=dict)


def plan_value(plan: Any) -> Any:
    """A plan as sent (a list, or a JSON string of one) -> the list, for the record the viewer reads."""
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            return plan
    if isinstance(plan, dict) and isinstance(plan.get("plan"), list):
        plan = plan["plan"]
    return plan


def _result(out: str) -> Any:
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"text": out}


def _args(raw: str) -> tuple[dict, str | None]:
    try:
        v = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return {}, f"arguments are not valid JSON: {e.msg}"
    return (v, None) if isinstance(v, dict) else ({}, "arguments must be a JSON object")


class AnthropicAgent:
    route = "messages"

    def __init__(self, model: str, tools, system: str, max_tokens: int, base_url: str, api_key: str,
                 effort: str | None = None, cache: bool = False):
        self.client = anthropic.AsyncAnthropic(base_url=base_url, auth_token=api_key, max_retries=4,
                                               timeout=REQUEST_TIMEOUT)
        self.model, self.system, self.max_tokens, self.cache = model, system, max_tokens, cache
        self.extra = ({"extra_body": {"output_config": {"effort": effort}}} if effort in ("low", "medium", "high")
                      else {})
        self.tools = [{"name": t.name, "description": t.description or "",
                       "input_schema": t.input_schema or {"type": "object", "properties": {}}} for t in tools]
        self.messages: list[dict] = []
        self._pending: list[dict] = []

    def user(self, text: str):
        self._pending.append({"type": "text", "text": text})

    def notes(self, text: str):
        self._pending.append({"type": "text", "text": f"(Your notes from the cut-off turn)\n{text}"})

    def results(self, pairs: list[tuple[Call, str]]):
        blocks = [{"type": "tool_result", "tool_use_id": c.id, "content": [{"type": "text", "text": out}]}
                  for c, out in pairs]
        self._pending = blocks + self._pending

    def request(self) -> dict:
        if self._pending:
            self.messages.append({"role": "user", "content": self._pending})
            self._pending = []
        messages = self.messages
        if self.cache:
            *blocks, newest = messages[-1]["content"]
            messages = [*messages[:-1], {**messages[-1], "content": [
                *blocks, {**newest, "cache_control": {"type": "ephemeral"}}]}]
        return {"model": self.model, "system": self.system, "messages": messages, "tools": self.tools,
                "max_tokens": self.max_tokens, **self.extra}

    async def step(self, request: dict) -> Turn:
        async with self.client.messages.stream(**request) as s:
            msg = await s.get_final_message()
        self.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        text, reasoning, calls = "", "", []
        for b in msg.content:
            if b.type == "text":
                text += b.text
            elif b.type == "thinking":
                reasoning += getattr(b, "thinking", "") or ""
            elif b.type == "tool_use":
                args = b.input if isinstance(b.input, dict) else {}
                calls.append(Call(b.id, b.name, args, None, json.dumps(b.input)))
        u = msg.usage
        cached, written = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        usage = {"input": (u.input_tokens or 0) + cached + written, "output": u.output_tokens or 0, "cached": cached,
                 "cache_write": written}
        return Turn(text, reasoning, calls, msg.stop_reason or "", usage)


class OpenAIResponsesAgent:
    route = "responses"

    def __init__(self, model: str, tools, system: str, max_tokens: int, base_url: str, api_key: str,
                 effort: str = "medium"):
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=REQUEST_TIMEOUT)
        self.model, self.system, self.max_tokens, self.effort = model, system, max_tokens, effort
        self.tools = [{"type": "function", "name": t.name, "description": t.description or "",
                       "parameters": t.input_schema or {"type": "object", "properties": {}}} for t in tools]
        self.input: list = []
        self._texts: list[str] = []

    def user(self, text: str):
        self._texts.append(text)

    def notes(self, text: str):
        self._texts.append(f"(Your notes from the cut-off turn)\n{text}")

    def results(self, pairs):
        for c, out in pairs:
            self.input.append({"type": "function_call_output", "call_id": c.id, "output": out})

    def request(self) -> dict:
        if self._texts:
            self.input.append({"role": "user", "content": [{"type": "input_text", "text": "\n\n".join(self._texts)}]})
            self._texts = []
        return {"model": self.model, "instructions": self.system, "input": self.input, "tools": self.tools,
                "max_output_tokens": self.max_tokens, "store": False, "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": self.effort, "summary": "auto"}, "stream": True}

    async def step(self, request: dict) -> Turn:
        resp = None
        async for event in await self.client.responses.create(**request):
            if event.type in ("response.completed", "response.incomplete"):
                resp = event.response
            elif event.type == "response.failed":
                raise ProviderError(f"response.failed: {event.response.error}")
            elif event.type == "error":
                raise ProviderError(f"error: {event.code}: {event.message}")
        if resp is None:
            raise ProviderError("the response stream ended without response.completed")
        self.input.extend(resp.output)
        text, reasoning, calls = "", "", []
        for item in resp.output:
            if item.type == "message":
                text += "".join(getattr(c, "text", "") for c in item.content)
            elif item.type == "reasoning":
                reasoning += "\n".join(getattr(s, "text", "") for s in (getattr(item, "summary", None) or []))
            elif item.type == "function_call":
                args, err = _args(item.arguments)
                calls.append(Call(item.call_id, item.name, args, err, item.arguments or ""))
        u = resp.usage
        return Turn(text, reasoning, calls, resp.status or "", {"input": u.input_tokens or 0,
                                                                 "output": u.output_tokens or 0,
                                                                 "cached": u.input_tokens_details.cached_tokens or 0,
                                                                 "cache_write": 0})


class ChatAgent:
    """OpenAI-compatible chat completions, streamed (long generations stall behind the HF router otherwise). Eight
    retries, about 40 s of backoff: a provider's tokens-per-minute limit, like Groq's, answers 429 until its minute
    turns. A call whose arguments aren't a JSON object goes back into the history as {}, its result saying why, since
    Fireworks refuses a history that holds one; the record keeps what the model sent."""

    route = "chat"

    def __init__(self, model: str, tools, system: str, max_tokens: int, base_url: str, api_key: str,
                 effort: str | None = None, bill_to: str | None = None):
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=8, timeout=REQUEST_TIMEOUT,
                                         default_headers={"X-HF-Bill-To": bill_to} if bill_to else None)
        self.model, self.max_tokens = model, max_tokens
        self.extra = {"reasoning_effort": effort} if effort in ("low", "medium", "high") else {}
        self.tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": t.input_schema or {"type": "object", "properties": {}}}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self._texts: list[str] = []

    def user(self, text: str):
        self._texts.append(text)

    def notes(self, text: str):
        if self.messages and self.messages[-1]["role"] == "assistant" and not self.messages[-1].get("content"):
            self.messages[-1]["content"] = f"(notes so far, cut off by the output limit)\n{text}"
        else:
            self._texts.append(f"(Your notes from the cut-off turn)\n{text}")

    def results(self, pairs):
        for c, out in pairs:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})

    def request(self) -> dict:
        if self._texts:
            self.messages.append({"role": "user", "content": "\n\n".join(self._texts)})
            self._texts = []
        return {"model": self.model, "messages": self.messages, "tools": self.tools, "max_tokens": self.max_tokens,
                "stream": True, "stream_options": {"include_usage": True}, **self.extra}

    async def step(self, request: dict) -> Turn:
        stream = await self.client.chat.completions.create(**request)
        content, reasoning, calls, stop = [], [], {}, ""
        usage = {"input": 0, "output": 0, "cached": 0, "cache_write": 0}
        budget, seen = self.max_tokens * 5, 0
        async for chunk in stream:
            if seen > budget:
                stop = HARNESS_CAP
                await stream.close()
                break
            if u := getattr(chunk, "usage", None):
                details = u.prompt_tokens_details
                usage = {"input": u.prompt_tokens or 0, "output": u.completion_tokens or 0,
                         "cached": (details.cached_tokens or 0) if details else 0, "cache_write": 0}
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            if ch.finish_reason:
                stop = ch.finish_reason
            if d is None:
                continue
            if d.content:
                content.append(d.content)
                seen += len(d.content)
            extra = getattr(d, "model_extra", None) or {}
            r = getattr(d, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
            if isinstance(r, str):
                reasoning.append(r)
                seen += len(r)
            for tc in d.tool_calls or []:
                e = calls.setdefault(tc.index or 0, {"id": "", "name": "", "args": ""})
                if tc.id:
                    e["id"] = tc.id
                if tc.function and tc.function.name:
                    e["name"] = tc.function.name if not e["name"] else e["name"]
                if tc.function and tc.function.arguments:
                    e["args"] += tc.function.arguments
        text = "".join(content)
        think = "".join(reasoning)
        if not think and "</think>" in text:
            think, text = text.split("</think>", 1)
            think = think.replace("<think>", "")
        out = []
        for i in sorted(calls):
            e = calls[i]
            args, err = _args(e["args"])
            out.append(Call(e["id"] or f"call_{uuid.uuid4().hex[:8]}", e["name"], args, err, e["args"]))
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if out:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": "{}" if c.error else c.raw or "{}"}}
                                 for c in out]
        self.messages.append(msg)
        return Turn(text, think, out, stop, usage)


def route(model: str) -> str:
    return "messages" if model.startswith("anthropic/") else "responses" if model.startswith("openai/") else "chat"


def make_agent(model: str, tools, system: str, max_tokens: int, effort: str | None, root: str, key: str,
               cache: bool = False, bill_to: str | None = None):
    if route(model) == "messages":
        return AnthropicAgent(model, tools, system, max_tokens, root, key, effort, cache)
    if route(model) == "responses":
        return OpenAIResponsesAgent(model, tools, system, max_tokens, f"{root}/v1", key, effort or "medium")
    return ChatAgent(model, tools, system, max_tokens, f"{root}/v1", key, effort, bill_to)


def bound(agent, request: dict, price: tuple[float, float, float]) -> float:
    """The most a request can cost: each byte of its body an input token (a cache write on Messages), and its output
    limit in output tokens."""
    size = len(json.dumps(request, default=lambda o: o.model_dump(mode="json")).encode())
    out = agent.max_tokens * (5 if agent.route == "chat" else 1)
    return (size * price[0] * (1.25 if agent.route == "messages" else 1) + out * price[1]) / 1e6


def cost(usage: dict, price: tuple[float, float, float]) -> float:
    fresh = usage["input"] - usage["cached"] - usage["cache_write"]
    return (fresh * price[0] + usage["cached"] * price[2] + usage["cache_write"] * 1.25 * price[0]
            + usage["output"] * price[1]) / 1e6


PROVIDER_ERRORS = (anthropic.APIError, openai.APIError, httpx.HTTPError, httpx2.HTTPError, ProviderError)
REFUSED = {401, 402, 403}  # the key or the account: nothing billed, and a retry gets the same answer
STREAM_RETRIES = 2


def mid_stream(error: BaseException) -> bool:
    """A reply cut or failed after its stream began: the SDKs' retries end where the stream starts, and no tool has run
    on it, so the same request can be sent again."""
    return not isinstance(error, (anthropic.APIStatusError, openai.APIStatusError, anthropic.APIConnectionError,
                                  openai.APIConnectionError))


# ================================================================ episode

class Episode:
    def __init__(self, config: PortSimConfig, environ: dict[str, str]):
        self.config, self.model = config, config.model
        self.base, self.key = environ.get("LITELLM_BASE_URL", ""), environ.get("LITELLM_API_KEY", "")
        cap = environ.get("PORTSIM_MAX_COST_USD")
        self.max_cost = None if cap is None else float(cap)
        self.bill_to = environ.get("HF_BILL_TO")
        self.started = time.time()
        self.record: dict[str, Any] = {"model": self.model, "messages": [], "steps": [],
                                       "final": {"submitted": False, "plan": None}, "reward": 0.0,
                                       "usage": {"input_tokens": 0, "output_tokens": 0}, "turns": 0,
                                       "end_reason": None, "errors": []}
        self.reward: float | None = None
        self.live = False
        self.tool_calls = self.checks = self.cached = self.written = 0
        self.spent = self.unbilled = 0.0
        self.failed_requests = 0
        self.failure: tuple[str, str] | None = None
        self.error: str | None = None

    def describe(self, error: BaseException) -> str:
        return f"{type(error).__name__}: {str(error).replace(self.key, '<LITELLM_API_KEY>')[:800]}"

    def fail(self, code: str, message: str | None = None, error: BaseException | None = None) -> None:
        if error is not None:
            self.error = self.describe(error)
            self.record["errors"].append(self.error)
        self.failure = (code, message or self.error)
        self.record["end_reason"] = code

    async def run(self, server: dict, opening: str, peers: dict[str, dict] | None = None) -> None:
        if not self.base or not self.key:
            return self.fail("model_endpoint", "no model endpoint: agent-env passes LITELLM_BASE_URL and "
                                               "LITELLM_API_KEY ([model] in .agentenv/config.toml)")
        if self.model not in PRICES:
            return self.fail("unpriced_model", f"no price for {self.model}; priced: {', '.join(PRICES)}")
        try:
            async with httpx.AsyncClient(headers=server.get("headers"), timeout=300) as http, \
                    httpx.AsyncClient(timeout=120) as peer_http, \
                    streamable_http_client(server["url"], http_client=http) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                await self.play(EnvSession(session, peers, peer_http), opening)
        except* (McpError, httpx.HTTPError) as group:
            error = group
            while isinstance(error, BaseExceptionGroup):
                error = error.exceptions[0]
            self.fail("env_error", error=error)

    async def play(self, env: EnvSession, opening: str) -> None:
        config, system = self.config, self.config.system_prompt
        self.record["messages"] += [{"role": "system", "content": system}, {"role": "user", "content": opening}]
        tools = await env.tools()
        self.live = any(t.name == "advance" for t in tools)
        limit = config.model_params["max_tokens"]
        agent = make_agent(self.model, tools, system, min(limit, MAX_OUTPUT.get(self.model, limit)), config.effort,
                           self.base.rstrip("/").removesuffix("/v1"), self.key, self.live, self.bill_to)
        agent.user(opening)
        async with agent.client:
            await self.turns(env, agent)

    async def turns(self, env: EnvSession, agent) -> None:
        config, record, price = self.config, self.record, PRICES[self.model]
        mode = LIVE if self.live else V1
        max_turns = config.max_turns
        empty = 0
        for turn in range(max_turns):
            elapsed = time.time() - self.started
            if self.live and elapsed > config.timeout_seconds - TIME_MARGIN:
                record["end_reason"] = "time_limit"
                break
            if elapsed > config.timeout_seconds:
                return self.fail("wall_clock",
                                 f"turn {turn + 1} would start after the {config.timeout_seconds} s limit")
            last = turn == max_turns - 1
            if last:
                agent.user(mode.last_turn)
                record["messages"].append({"role": "user", "content": mode.last_turn})
            request = agent.request()
            most = bound(agent, request, price)
            if self.max_cost is not None and self.spent + most > self.max_cost:
                return self.fail("cost_cap", f"the next request could cost ${most:.4f}, past the ${self.max_cost:g} "
                                             f"cap with ${self.spent:.4f} spent")
            for resend in range(STREAM_RETRIES + 1):
                try:
                    t = await agent.step(request)
                    break
                except PROVIDER_ERRORS as e:
                    if getattr(e, "status_code", None) in REFUSED:
                        return self.fail("provider_refused", error=e)
                    self.spent += most
                    self.unbilled += most
                    self.failed_requests += 1
                    if resend == STREAM_RETRIES or not mid_stream(e) or (
                            self.max_cost is not None and self.spent + most > self.max_cost):
                        return self.fail("provider_error", error=e)
                    self.record["errors"].append(f"{self.describe(e)} (turn {turn + 1} sent again)")
                    log.warning("turn %d: the reply failed mid-stream, sending it again", turn + 1)
            self.spent += most if t.stop == HARNESS_CAP else cost(t.usage, price)
            self.cached += t.usage["cached"]
            self.written += t.usage["cache_write"]
            record["turns"] += 1
            record["usage"]["input_tokens"] += t.usage.get("input", 0)
            record["usage"]["output_tokens"] += t.usage.get("output", 0)
            record["messages"].append({"role": "assistant", "content": t.text, "reasoning": t.reasoning, "stop": t.stop,
                                       "tool_calls": [{"id": c.id, "name": c.name,
                                                       "arguments": c.raw or json.dumps(c.arguments)}
                                                      for c in t.calls]})
            log.info("turn %d: %d tool calls, stop %s, $%.4f spent", turn + 1, len(t.calls), t.stop, self.spent)
            if not t.calls:
                empty += 1
                if empty >= 2 or last:
                    record["end_reason"] = "no_tool_call"
                    break
                if t.stop.startswith("length") and t.reasoning:
                    agent.notes(t.reasoning[-NOTES_CHARS:])
                    nudge = mode.cut_off
                else:
                    nudge = mode.no_call
                agent.user(nudge)
                record["messages"].append({"role": "user", "content": nudge})
                continue
            empty = 0
            pairs, done = [], False
            for c in t.calls:
                if done:
                    out = json.dumps({"error": "the episode is over"})
                elif c.error:
                    out = json.dumps({"error": c.error})
                else:
                    out = await env.call(c.name, c.arguments)
                    self.tool_calls += 1
                    if c.name == "check_plan":
                        self.checks += 1
                    res = _result(out)
                    if c.name in mode.steps:
                        record["steps"].append({"turn": turn + 1, "tool": c.name,
                                                "plan": plan_value(c.arguments.get("plan")), "result": res})
                    if self.live and res.get("done") is True:
                        done = True
                        self.reward = record["reward"] = float(res["reward"])
                        record["end_reason"] = "done"
                    elif c.name == "submit_plan" and res.get("submitted") is True:
                        done = True
                        self.reward = record["reward"] = float(res["reward"])
                        record["final"] = {"submitted": True, "plan": plan_value(c.arguments.get("plan"))}
                        record["end_reason"] = "submitted"
                    if not (done or self.live) and self.tool_calls >= MAX_TOOL_CALLS:
                        done = True
                        self.reward = record["reward"] = 0.0
                        record["end_reason"] = "tool_call_limit"
                left = max_turns - turn - 1
                text = out if done else f"{out}\n(turns left: {left})"
                pairs.append((c, text))
                record["messages"].append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out})
            agent.results(pairs)
            if done:
                break
        else:
            record["end_reason"] = record["end_reason"] or "turn_limit"

    def result(self) -> TaskResult:
        record = self.record
        record["seconds"] = round(time.time() - self.started, 1)
        usage = record["usage"]
        summary = {"agent": "portsim-llm", "model": self.model, "route": route(self.model),
                   "end_reason": record["end_reason"], "submitted": record["end_reason"] == "submitted",
                   "reward": self.reward, "turns": record["turns"], "tool_calls": self.tool_calls,
                   "checks": self.checks, "input_tokens": usage["input_tokens"],
                   "output_tokens": usage["output_tokens"], "cached_tokens": self.cached,
                   "cache_write_tokens": self.written, "cost_usd": round(self.spent, 6), "max_cost_usd": self.max_cost,
                   "priced_cost_usd": round(self.spent - self.unbilled, 6), "failed_requests": self.failed_requests,
                   "seconds": record["seconds"], "error": self.error}
        text = (f"{record['end_reason']}: reward {json.dumps(self.reward)}, {record['turns']} turns, "
                f"{self.tool_calls} tool calls, ${self.spent:.4f} ({self.model})")
        builder = TaskResult.builder()
        builder = builder.failed(*self.failure, error_type="infra_error") if self.failure else builder.succeeded()
        return (builder.add_text(text).add_structured_output(summary)
                .usage(Usage(tool_call_count=self.tool_calls, input_tokens=usage["input_tokens"],
                             output_tokens=usage["output_tokens"],
                             total_tokens=usage["input_tokens"] + usage["output_tokens"],
                             cost_usd=round(self.spent, 6),
                             provider_details={"cached_tokens": self.cached, "cache_write_tokens": self.written}))
                .native_trajectory(format="portsim-rollout", payload=record).build())


@a2a_agent(
    identity=AgentIdentity(
        name="portsim-llm",
        description="PortSimEnv's harness loop: a model re-plans a disrupted week of berthing at a Port of Barcelona "
                    "quay through the env's tools, with upstream's prompts and limits, on agent-env's model endpoint.",
        version="0.2.0",
    ),
    config=PortSimConfig,
    extensions=(MCP_CONFIG_V1, TRAJECTORY_V1, PEER_AGENTS_V1),
)
class PortSimLLM(AgentEnvAgent):
    def __init__(self):
        super().__init__()
        self.peers: dict[str, dict] = {}

    async def run(self, request: TaskRequest[PortSimConfig]) -> TaskResult:
        [part] = request.parts
        [server] = request.mcp_servers.values()
        episode = Episode(request.config, dict(os.environ))
        await episode.run(server, part.text, self.peers)
        return episode.result()

    @extension(PEER_AGENTS_V1.set)
    async def set_peers(self, request: PeerAgentsSetRequest) -> dict:
        self.peers = {p.name: p.model_dump() for p in request.peers}
        return {"status": "updated", "peers": sorted(self.peers)}

    @extension(PEER_AGENTS_V1.list)
    async def list_peers(self) -> dict:
        return {"peers": list(self.peers.values())}


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    PortSimLLM().serve()
