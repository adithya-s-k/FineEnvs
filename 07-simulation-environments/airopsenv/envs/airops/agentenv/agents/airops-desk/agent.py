"""A desk of the AirOps ops control center as an A2A agent: the ops controller, the crew scheduler, or one desk running
both, in its role on the airops-occ env (ops and both through an MCP session, crew through the gateway's REST /step
under its role header). Its turns come from the env: it reads its page with wait_turn, which it polls again unseen by
the model until the turn is its own, acts through its desk's tools, and ends the turn with end_turn, whose result the
model sees as the next page. With policy "llm" a model plays through agent-env's model endpoint, on the API each
provider's models take (Anthropic Messages, OpenAI Responses, OpenAI-compatible chat), cost as usage times PRICES and
capped by AIROPS_MAX_COST_USD; with policy "script" (AIROPS_DESK_POLICY) a fixed script plays on the pages and tools
alone, with no model. Either way the desk resigns as it stops, with its reason; the summary gives its cost, tokens,
turns and the largest context it sent. A model desk stops once a turn would start within TIME_MARGIN of the time limit;
a time limit that leaves it no turn at all fails it, so the day is unscored. A request with its retries can outlast
TIME_MARGIN, and then the step times out."""

import asyncio
import contextlib
import json
import logging
import os
import re
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
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    Usage,
    a2a_agent,
)
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client

log = logging.getLogger("airops-desk")

REQUEST_TIMEOUT = 900.0
NOTES_CHARS = 8000
TIME_MARGIN = 1200
HARNESS_CAP = "length (harness cap)"
ROLE_HEADER = "AgentEnv-Role"
STEP_TIMEOUT = 600.0
CACHE_SECONDS = 300
PRICES = {  # USD per 1M tokens: input, output, cache read. LiteLLM public price map, 2026-10-06 and 2026-10-09.
    "anthropic/claude-opus-5-5": (4.00, 20.00, 0.40),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "anthropic/claude-haiku-5-5": (0.10, 0.50, 0.01),
    "openai/gpt-6-astra": (10.00, 50.00, 1.00),
    "openai/gpt-6.1-sol": (2.00, 10.00, 0.10),
    "openai/gpt-6-luna": (0.10, 0.50, 0.01),
}
CONTEXT_FULL = re.compile(r"context.{0,20}(length|window|limit)|too long|maximum context|too many tokens", re.I)
NO_CALL = ("No tool was called. Your decisions only count through the tools; end your turn with end_turn when you are "
           "done with it.")
CUT_OFF = ("Your last request ran out of output tokens before calling a tool. Your notes from it are above. Keep "
           "reasoning short now and act, then end your turn.")
LAST_TURN = "This is your last model turn today: commit what you still want now; the day then runs on without you."


class DeskConfig(AgentConfig):
    model: str = "openai/gpt-6.1-sol"
    system_prompt: str = ""
    max_turns: int = 232
    timeout_seconds: int = 21600
    model_params: dict[str, Any] = {"max_tokens": 32000}
    effort: str | None = None


class ProviderError(Exception):
    """A Responses stream that reported a failure, or ended without a response."""


@dataclass
class Tool:
    name: str
    description: str | None
    input_schema: dict | None


class McpDesk:
    """The desk's tools through one MCP session on the env (its role in the session's headers)."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def tools(self) -> list[Tool]:
        return [Tool(t.name, t.description, t.inputSchema) for t in (await self.session.list_tools()).tools]

    async def call(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return json.dumps({"error": text}) if result.isError else text


class StepDesk:
    """The desk's tools through the gateway's REST /step route under its role header."""

    def __init__(self, http: httpx.AsyncClient, url: str):
        self.http, self.url = http, url

    async def tools(self) -> list[Tool]:
        response = await self.http.post(self.url, json={"action": "list_tools"})
        response.raise_for_status()
        return [Tool(t["name"], t.get("description"), t.get("parameters")) for t in response.json()["tools"]]

    async def call(self, name: str, arguments: dict) -> str:
        response = await self.http.post(self.url, json={"action": "call_tool", "tool_name": name,
                                                        "arguments": arguments})
        body = response.json()
        if response.status_code != 200:
            return json.dumps({"error": body.get("error", f"HTTP {response.status_code}")})
        text = "\n".join(c["text"] for c in body.get("content", []) if c.get("type") == "text")
        return json.dumps({"error": text}) if body.get("isError") else text


def step_url(mcp_url: str) -> str:
    return mcp_url.rstrip("/").removesuffix("/mcp") + "/step"


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
    """Messages with one rolling cache breakpoint on the newest block: a day re-sends its whole history every turn."""

    route = "messages"

    def __init__(self, model: str, tools, system: str, max_tokens: int, base_url: str, api_key: str,
                 effort: str | None = None):
        self.client = anthropic.AsyncAnthropic(base_url=base_url, auth_token=api_key, max_retries=4,
                                               timeout=REQUEST_TIMEOUT)
        self.model, self.system, self.max_tokens = model, system, max_tokens
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
        *blocks, newest = self.messages[-1]["content"]
        messages = [*self.messages[:-1], {**self.messages[-1], "content": [
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
        items, stop = list(resp.output), resp.status or ""
        if stop == "incomplete":
            done = [k for k, x in enumerate(items) if x.type in ("message", "function_call") and
                    getattr(x, "status", "completed") == "completed"]
            history = items[:done[-1] + 1] if done else []
            reason = getattr(resp.incomplete_details, "reason", None)
            stop = f"length ({reason})" if reason == "max_output_tokens" else f"incomplete ({reason})"
        else:
            history = items
        self.input.extend(history)
        text, reasoning, calls = "", "", []
        for item in items:
            if item.type == "message":
                text += "".join(getattr(c, "text", "") for c in item.content)
            elif item.type == "reasoning":
                reasoning += "\n".join(getattr(s, "text", "") for s in (getattr(item, "summary", None) or []))
            elif item.type == "function_call" and item in history:
                args, err = _args(item.arguments)
                calls.append(Call(item.call_id, item.name, args, err, item.arguments or ""))
        u = resp.usage
        return Turn(text, reasoning, calls, stop, {"input": u.input_tokens or 0,
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
               bill_to: str | None = None):
    if route(model) == "messages":
        return AnthropicAgent(model, tools, system, max_tokens, root, key, effort)
    if route(model) == "responses":
        return OpenAIResponsesAgent(model, tools, system, max_tokens, f"{root}/v1", key, effort or "medium")
    return ChatAgent(model, tools, system, max_tokens, f"{root}/v1", key, effort, bill_to)


def size_of(request: dict) -> int:
    return len(json.dumps(request, default=lambda o: o.model_dump(mode="json")).encode())


def estimate(agent, size: int, last: tuple[int, int] | None, out: int, price: tuple[float, float, float],
             warm: bool) -> tuple[float, float]:
    """A request's likely cost, input and output: the last response's input tokens read from the cache while it is
    warm (else fresh), the bytes added since at 3.5 a token fresh (a cache write on Messages), and twice the largest
    output yet."""
    cached, sent = last or (0, 0)
    fresh = max(0, size - sent) / 3.5 + (0 if warm else cached)
    return (((cached if warm else 0) * price[2] + fresh * price[0] * (1.25 if agent.route == "messages" else 1)) / 1e6,
            2 * out * price[1] / 1e6)


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




# ================================================================ scripts

NOW = re.compile(r"^now f(\d+) \S+ \S+ \S+ (pilot|cabin): (?:HOLD (\S+)|CANCEL)", re.M)
OPEN = re.compile(r"^(R\d+) from (ops|crew) at \S+: (CANCEL_PAIR|CANCEL|HOLD|COVER) f(\d+)", re.M)
LINE = re.compile(r"freeze line (\S+):")


class Script:
    """policy "script": a fixed desk on its pages and tools alone, acting once on each of the crew system's words (a
    flight's latest on the page). The crew desk forwards them, CANCEL as a CANCEL request, HOLD as a TEXT; the ops desk
    accepts each CANCEL request at its turn; one desk running both cancels the flights the crew system can't crew and
    holds the others to its crew times."""

    def __init__(self, desk: str):
        self.desk = desk
        self.acted: set[str] = set()

    def new(self, page: str) -> list[tuple]:
        """The crew system's words on the page not acted on yet (a word stays on the page while it stands)."""
        out = []
        for m in NOW.finditer(page):
            end = page.find("\n", m.start())
            line = page[m.start():end if end >= 0 else None].split("; fix")[0]
            if line not in self.acted:
                self.acted.add(line)
                out.append(m.groups())
        return out

    async def turn(self, page: str, call) -> None:
        if self.desk == "crew":
            for i, _, hold in self.new(page):
                if hold:
                    await call("crew_request", {"type": "TEXT", "flight": f"f{i}", "time": None,
                                                "text": f"crew system: HOLD f{i} until {hold}"})
                else:
                    await call("crew_request", {"type": "CANCEL", "flight": f"f{i}", "time": None, "text": None})
        elif self.desk == "ops":
            for rid, _, kind, _ in OPEN.findall(page):
                if kind in ("CANCEL", "CANCEL_PAIR"):
                    await call("ops_reply", {"request": rid, "answer": "accept", "text": None})
        else:
            line = LINE.search(page)
            changes = []
            for i, _, hold in self.new(page):
                if not hold:
                    changes.append({"flight": f"f{i}", "action": "cancel", "time": None, "other": None, "ref": None})
                else:
                    at = hold if line is None or minutes(hold) >= minutes(line.group(1)) else line.group(1)
                    changes.append({"flight": f"f{i}", "action": "hold", "time": at, "other": None, "ref": None})
            if changes:
                await call("ops_confirm", {"changes": changes})


def minutes(text: str) -> int:
    clock, _, day = text.partition("+")
    h, m = clock.split(":")
    return int(h) * 60 + int(m) + 1440 * int(day or 0)


# ================================================================ episode

class Episode:
    def __init__(self, config: DeskConfig, environ: dict[str, str]):
        self.config, self.model = config, config.model
        self.desk = {"ops": "ops", "crew": "crew", "both": "one"}.get(config.role or "", "")
        self.policy = environ.get("AIROPS_DESK_POLICY", "llm")
        self.base, self.key = environ.get("LITELLM_BASE_URL", ""), environ.get("LITELLM_API_KEY", "")
        cap = environ.get("AIROPS_MAX_COST_USD")
        self.max_cost = None if cap is None else float(cap)
        self.die_at = int(environ["AIROPS_DESK_DIE_AT"]) if environ.get("AIROPS_DESK_DIE_AT") else None
        sleep = environ.get("AIROPS_DESK_SLEEP_AT", "")
        self.sleep_at = tuple(int(x) for x in sleep.split(":")) if sleep else None
        self.started = time.time()
        self.record: dict[str, Any] = {"model": self.model if self.policy == "llm" else None, "desk": self.desk,
                                       "policy": self.policy, "messages": [], "pages": [], "turns": 0,
                                       "usage": {"input_tokens": 0, "output_tokens": 0}, "end_reason": None,
                                       "errors": []}
        self.tool_calls = self.desk_turns = self.cached = self.written = self.context = 0
        self.spent = 0.0
        self.effort: str | None = None
        self.last: tuple[int, int] | None = None
        self.last_at = 0.0
        self.most_out = 0
        self.failed_requests = 0
        self.failure: tuple[str, str] | None = None
        self.error: str | None = None
        self.reward: float | None = None

    def describe(self, error: BaseException) -> str:
        return f"{type(error).__name__}: {str(error).replace(self.key, '<LITELLM_API_KEY>')[:800]}" if self.key else \
            f"{type(error).__name__}: {str(error)[:800]}"

    def stop(self, reason: str, message: str | None = None, error: BaseException | None = None,
             failed: bool = False) -> None:
        if error is not None:
            self.error = self.describe(error)
            self.record["errors"].append(self.error)
        self.record["end_reason"] = reason
        if failed:
            self.failure = (reason, message or self.error or reason)

    async def run(self, server: dict, opening: str) -> None:
        if not self.desk:
            return self.stop("error", f"role must be ops, crew or both, not {self.config.role!r}", failed=True)
        if self.policy == "llm" and (not self.base or not self.key):
            return self.stop("model_endpoint", "no model endpoint: agent-env passes LITELLM_BASE_URL and "
                                               "LITELLM_API_KEY ([model] in .agentenv/config.toml)", failed=True)
        if self.policy == "llm" and self.model not in PRICES:
            return self.stop("unpriced_model", f"no price for {self.model}; priced: {', '.join(PRICES)}", failed=True)
        if self.policy == "llm" and self.config.timeout_seconds <= TIME_MARGIN:
            return self.stop("timeout", f"timeout_seconds {self.config.timeout_seconds} leaves no turn: the harness "
                                        f"stops {TIME_MARGIN} s before the time limit", failed=True)
        headers = {**(server.get("headers") or {}), ROLE_HEADER: self.config.role}
        try:
            async with httpx.AsyncClient(headers=headers, timeout=STEP_TIMEOUT) as http:
                if self.desk == "crew":
                    await self.play(StepDesk(http, step_url(server["url"])), opening)
                else:
                    async with streamable_http_client(server["url"], http_client=http) as (read, write, _), \
                            ClientSession(read, write) as session:
                        await session.initialize()
                        await self.play(McpDesk(session), opening)
        except* (McpError, httpx.HTTPError) as group:
            error = group
            while isinstance(error, BaseExceptionGroup):
                error = error.exceptions[0]
            self.stop("error", error=error, failed=True)

    async def play(self, env, opening: str) -> None:
        try:
            page = await self.wait(env)
            if page is None:
                self.record["end_reason"] = self.record["end_reason"] or "done"
                return
            if self.policy == "script":
                await self.scripted(env, page)
            else:
                await self.llm(env, opening, page)
        except BaseException:
            self.record["end_reason"] = self.record["end_reason"] or "error"
            raise
        finally:
            reason = self.record["end_reason"] or "done"
            with contextlib.suppress(Exception):
                await env.call("resign", {"reason": reason if reason in RESIGN else "error"})

    async def wait(self, env) -> dict | None:
        """Polls wait_turn until the desk's page comes, unseen by the model; None once the day is over for it (its
        absence as the end, if the env put it off), or on an error, which stops the desk unscored."""
        while True:
            out = _result(await env.call("wait_turn", {}))
            if isinstance(out, dict) and "page" in out:
                self.desk_turns += 1
                self.record["pages"].append({"watch": out["watch"], "turn": out["turn"], "chars": len(out["page"])})
                return out
            if not isinstance(out, dict) or (not out.get("done") and "error" in out):
                self.stop("error", f"wait_turn: {json.dumps(out)[:500]}", failed=True)
                return None
            if out.get("done"):
                self.reward = out.get("reward")
                if out.get("absent"):
                    self.stop(out["absent"])
                return None

    async def scripted(self, env, page: dict) -> None:
        script = Script(self.desk)

        async def call(name: str, args: dict) -> dict:
            self.tool_calls += 1
            return _result(await env.call(name, args))
        while page is not None:
            if self.die_at is not None and page["watch"] >= self.die_at:
                os._exit(17)
            if self.sleep_at is not None and page["watch"] == self.sleep_at[0]:
                await asyncio.sleep(self.sleep_at[1])
            await script.turn(page["page"], call)
            await call("end_turn", {"note": None, "stand_down": False})
            page = await self.wait(env)
        self.record["end_reason"] = self.record["end_reason"] or "done"

    async def llm(self, env, opening: str, page: dict) -> None:
        config, record, price = self.config, self.record, PRICES[self.model]
        system = config.system_prompt
        first = f"{opening}\n\n{page['page']}"
        record["messages"] += [{"role": "system", "content": system}, {"role": "user", "content": first}]
        tools = [t for t in await env.tools() if t.name not in ("wait_turn", "resign")]
        limit = config.model_params["max_tokens"]
        self.effort = config.effort or ("medium" if route(self.model) == "responses" else None)
        agent = make_agent(self.model, tools, system, limit, config.effort, self.base.rstrip("/").removesuffix("/v1"),
                           self.key)
        agent.user(first)
        async with agent.client:
            await self.turns(env, agent, price)

    async def turns(self, env, agent, price) -> None:
        config, record = self.config, self.record
        empty = 0
        for turn in range(config.max_turns):
            if time.time() - self.started > config.timeout_seconds - TIME_MARGIN:
                if turn == 0:
                    return self.stop("timeout", f"the time limit left no turn: the harness stops {TIME_MARGIN} s "
                                                "before it", failed=True)
                return self.stop("time_limit")
            if turn == config.max_turns - 1:
                agent.user(LAST_TURN)
                record["messages"].append({"role": "user", "content": LAST_TURN})
            request = agent.request()
            size = size_of(request)
            ins, outs = estimate(agent, size, self.last, self.most_out, price,
                                 time.monotonic() - self.last_at < CACHE_SECONDS)
            if self.max_cost is not None and self.spent + ins + outs > self.max_cost:
                return self.stop("cost_cap", f"the next request would likely cost ${ins + outs:.4f}, past the "
                                             f"${self.max_cost:g} cap with ${self.spent:.4f} spent")
            for resend in range(STREAM_RETRIES + 1):
                try:
                    t = await agent.step(request)
                    break
                except PROVIDER_ERRORS as e:
                    if CONTEXT_FULL.search(str(e)):
                        return self.stop("context_full", error=e)
                    if getattr(e, "status_code", None) in REFUSED:
                        return self.stop("provider_refused", error=e, failed=True)
                    self.failed_requests += 1
                    if resend == STREAM_RETRIES or not mid_stream(e):
                        return self.stop("provider_error", error=e, failed=True)
                    record["errors"].append(f"{self.describe(e)} (request {turn + 1} sent again)")
            self.spent += ins + agent.max_tokens * price[1] / 1e6 if t.stop == HARNESS_CAP else cost(t.usage, price)
            self.last, self.last_at = (t.usage.get("input") or round(size / 3.5), size), time.monotonic()
            self.most_out = max(self.most_out, t.usage.get("output", 0))
            self.cached += t.usage["cached"]
            self.written += t.usage["cache_write"]
            self.context = max(self.context, t.usage.get("input", 0))
            record["turns"] += 1
            record["usage"]["input_tokens"] += t.usage.get("input", 0)
            record["usage"]["output_tokens"] += t.usage.get("output", 0)
            record["messages"].append({"role": "assistant", "content": t.text, "reasoning": t.reasoning, "stop": t.stop,
                                       "tool_calls": [{"id": c.id, "name": c.name,
                                                       "arguments": c.raw or json.dumps(c.arguments)}
                                                      for c in t.calls]})
            log.info("request %d: %d tool calls, stop %s, $%.4f spent", turn + 1, len(t.calls), t.stop, self.spent)
            if not t.calls:
                empty += 1
                if empty >= 3:
                    return self.stop("done")
                if t.stop.startswith("length") and t.reasoning:
                    agent.notes(t.reasoning[-NOTES_CHARS:])
                    nudge = CUT_OFF
                else:
                    nudge = NO_CALL
                agent.user(nudge)
                record["messages"].append({"role": "user", "content": nudge})
                continue
            empty = 0
            pairs, over, ended = [], False, False
            for c in t.calls:
                if over:
                    out = json.dumps({"error": "the day is over for this desk"})
                elif ended:
                    out = json.dumps({"error": "not run: it came after end_turn in the same reply; your turn had "
                                               "ended"})
                elif c.error:
                    out = json.dumps({"error": c.error})
                else:
                    out = await env.call(c.name, c.arguments)
                    self.tool_calls += 1
                    res = _result(out)
                    if c.name == "end_turn" and "error" not in res:
                        ended = True
                        nxt = await self.wait(env)
                        if nxt is None:
                            over = True
                            out = json.dumps({**res, "done": True, "reward": self.reward})
                        else:
                            out = json.dumps({**res, **nxt})
                    elif isinstance(res, dict) and res.get("done"):
                        over = True
                        self.stop("off" if "off" in str(res.get("error")) else "done")
                pairs.append((c, out))
                record["messages"].append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out})
            agent.results(pairs)
            if over:
                return None if record["end_reason"] else self.stop("done")
        self.stop("turn_limit")

    def result(self) -> TaskResult:
        record = self.record
        record["seconds"] = round(time.time() - self.started, 1)
        usage = record["usage"]
        summary = {"agent": "airops-desk", "desk": self.desk, "role": self.config.role, "policy": self.policy,
                   "model": record["model"], "route": route(self.model) if self.policy == "llm" else None,
                   "effort": self.effort,
                   "end_reason": record["end_reason"], "reward": self.reward, "desk_turns": self.desk_turns,
                   "turns": record["turns"], "tool_calls": self.tool_calls, "input_tokens": usage["input_tokens"],
                   "output_tokens": usage["output_tokens"], "cached_tokens": self.cached,
                   "cache_write_tokens": self.written, "context_max": self.context, "cost_usd": round(self.spent, 6),
                   "max_cost_usd": self.max_cost, "failed_requests": self.failed_requests, "seconds": record["seconds"],
                   "error": self.error}
        text = (f"{self.desk} desk {record['end_reason']}: reward {json.dumps(self.reward)}, {self.desk_turns} desk "
                f"turns, {record['turns']} model turns, {self.tool_calls} tool calls, ${self.spent:.4f}")
        builder = TaskResult.builder()
        builder = builder.failed(*self.failure, error_type="infra_error") if self.failure else builder.succeeded()
        return (builder.add_text(text).add_structured_output(summary)
                .usage(Usage(tool_call_count=self.tool_calls, input_tokens=usage["input_tokens"],
                             output_tokens=usage["output_tokens"],
                             total_tokens=usage["input_tokens"] + usage["output_tokens"],
                             cost_usd=round(self.spent, 6),
                             provider_details={"cached_tokens": self.cached, "cache_write_tokens": self.written}))
                .native_trajectory(format="airops-desk", payload=record).build())


RESIGN = ("done", "cost_cap", "context_full", "turn_limit", "time_limit", "error", "crash")


@a2a_agent(
    identity=AgentIdentity(
        name="airops-desk",
        description="A desk of the AirOps ops control center: the ops controller, the crew scheduler, or both, playing "
                    "a storm day turn by turn through the env's tools under its role.",
        version="0.1.0",
    ),
    config=DeskConfig,
    extensions=(MCP_CONFIG_V1, TRAJECTORY_V1),
)
class AirOpsDesk(AgentEnvAgent):
    async def run(self, request: TaskRequest[DeskConfig]) -> TaskResult:
        [part] = request.parts
        [server] = request.mcp_servers.values()
        episode = Episode(request.config, dict(os.environ))
        try:
            await episode.run(server, part.text)
        except Exception as e:
            episode.stop("crash", error=e, failed=True)
        return episode.result()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    AirOpsDesk().serve()
