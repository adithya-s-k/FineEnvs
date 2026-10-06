"""Tool-calling agent loop against a berth-planning server, for three provider APIs.

    anthropic:<model>[@low]     Anthropic Messages API (streamed; thinking blocks kept in the history)
    openai:<model>[@low|@high]  OpenAI Responses API (reasoning models only accept function tools here)
    hf:<org/model[:provider]>   Hugging Face router, OpenAI-compatible chat completions (streamed)

Every tool call goes through the env server over OpenEnv's WebSocket session, so the reward recorded is the server's
rubric. One episode: the system prompt is the task's rules, the first user message its situation; the model gets
`max_turns` turns (a turn may hold several tool calls); each tool result says how many turns are left; the last
turn asks it to submit. Two turns in a row without a tool call end the episode. An episode without `submit_plan`
scores 0.

`run_episode` returns the rollout record the viewer reads (messages with reasoning, every plan the agent sent with the
server's answer, the final grade, token usage).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

REQUEST_TIMEOUT = 900.0
NOTES_CHARS = 8000
OPENING = ("{situation}\n\nMake the new berth plan. Use check_plan to test drafts, then call submit_plan with your final "
           "plan.")


# ================================================================ env session

class EnvSession:
    """One WebSocket session on a berth-planning server (one episode at a time)."""

    def __init__(self, base_url: str):
        self.client = MCPToolClient(base_url.rstrip("/"), message_timeout_s=300.0).sync()
        self._tools = None

    def reset(self, task_id: str, episode_id: str | None = None) -> dict:
        kw = {"task_id": task_id}
        if episode_id:
            kw["episode_id"] = episode_id
        return dict(self.client.reset(**kw).observation.metadata or {})

    @property
    def tools(self):
        if self._tools is None:
            self._tools = self.client.list_tools()
        return self._tools

    def call(self, name: str, arguments: dict) -> tuple[str, bool, float | None, dict]:
        step = self.client.step(CallToolAction(tool_name=name, arguments=arguments))
        obs = step.observation
        err = getattr(obs, "error", None)
        res = getattr(obs, "result", None)
        text = None
        if isinstance(res, dict):
            blocks = res.get("content") or []
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None
            if text is None and res.get("data") is not None:
                text = res["data"] if isinstance(res["data"], str) else json.dumps(res["data"])
        elif res is not None:
            data = getattr(res, "data", None)
            text = data if isinstance(data, str) else json.dumps(data) if data is not None else str(res)
        if err:
            msg = err.get("message") if isinstance(err, dict) else getattr(err, "message", str(err))
            text = json.dumps({"error": str(msg)})
        return text or "", bool(step.done or obs.done), step.reward if step.reward is not None else obs.reward, \
            dict(obs.metadata or {})

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


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


def _args(raw: str) -> tuple[dict, str | None]:
    try:
        v = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return {}, f"arguments are not valid JSON: {e.msg}"
    return (v, None) if isinstance(v, dict) else ({}, "arguments must be a JSON object")


class AnthropicAgent:
    def __init__(self, model: str, tools, system: str, max_tokens: int, effort: str = "default"):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=4, timeout=REQUEST_TIMEOUT)
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.extra = {"extra_body": {"output_config": {"effort": effort}}} if effort in ("low", "medium", "high") else {}
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

    def step(self) -> Turn:
        if self._pending:
            self.messages.append({"role": "user", "content": self._pending})
            self._pending = []
        with self.client.messages.stream(model=self.model, system=self.system, messages=self.messages,
                                         tools=self.tools, max_tokens=self.max_tokens, **self.extra) as s:
            msg = s.get_final_message()
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
        usage = {"input": (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                 + (getattr(u, "cache_creation_input_tokens", 0) or 0), "output": u.output_tokens or 0}
        return Turn(text, reasoning, calls, msg.stop_reason or "", usage)


class OpenAIResponsesAgent:
    def __init__(self, model: str, tools, system: str, max_tokens: int, effort: str = "medium"):
        from openai import OpenAI

        self.client = OpenAI(max_retries=3, timeout=REQUEST_TIMEOUT)
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

    def step(self) -> Turn:
        if self._texts:
            self.input.append({"role": "user", "content": [{"type": "input_text", "text": "\n\n".join(self._texts)}]})
            self._texts = []
        resp = self.client.responses.create(model=self.model, instructions=self.system, input=self.input,
                                            tools=self.tools, max_output_tokens=self.max_tokens, store=False,
                                            include=["reasoning.encrypted_content"],
                                            reasoning={"effort": self.effort, "summary": "auto"})
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
        return Turn(text, reasoning, calls, resp.status or "", {"input": getattr(u, "input_tokens", 0) or 0,
                                                                 "output": getattr(u, "output_tokens", 0) or 0})


class ChatAgent:
    """OpenAI-compatible chat completions, streamed (long generations stall behind the HF router otherwise)."""

    def __init__(self, model: str, tools, system: str, max_tokens: int, base_url: str, api_key: str,
                 effort: str = "default"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=REQUEST_TIMEOUT)
        # "org/model:prov1|prov2" -> try prov1, fall back to prov2 on a rejected request
        name, _, provs = model.rpartition(":") if ":" in model else (model, "", "")
        chain = [f"{name}:{p}" for p in provs.split("|")] if provs and "|" in provs else [model]
        self.model, self.fallbacks, self.provider_switches = chain[0], chain[1:], []
        self.max_tokens = max_tokens
        self.extra = {"reasoning_effort": effort} if effort in ("low", "medium", "high") else {}
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                        "parameters": t.input_schema or {"type": "object", "properties": {}}}}
                      for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self._texts: list[str] = []

    def _create(self):
        """Stream a completion; on a request the provider rejects or fails (context too long, 4xx, 5xx - not 429), move on
        to the next provider in the spec (`org/model:prov1|prov2`) for the rest of the episode."""
        import openai

        while True:
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=self.tools, max_tokens=self.max_tokens,
                    stream=True, stream_options={"include_usage": True}, **self.extra)
            except (openai.BadRequestError, openai.UnprocessableEntityError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", 0) or 0
                if not self.fallbacks or status == 429:
                    raise
                self.provider_switches.append(f"{self.model} -> {self.fallbacks[0]} ({status}: {str(e)[:120]})")
                self.model = self.fallbacks.pop(0)

    def user(self, text: str):
        self._texts.append(text)

    def notes(self, text: str):
        # The cut-off turn left an empty assistant message; its reasoning becomes that message's content.
        if self.messages and self.messages[-1]["role"] == "assistant" and not self.messages[-1].get("content"):
            self.messages[-1]["content"] = f"(notes so far, cut off by the output limit)\n{text}"
        else:
            self._texts.append(f"(Your notes from the cut-off turn)\n{text}")

    def results(self, pairs):
        for c, out in pairs:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})

    def step(self) -> Turn:
        if self._texts:
            self.messages.append({"role": "user", "content": "\n\n".join(self._texts)})
            self._texts = []
        stream = self._create()
        content, reasoning, calls, stop, usage = [], [], {}, "", {"input": 0, "output": 0}
        # Some providers ignore max_tokens (Novita served 65-75k-token turns for Qwen3.8); cap it here by characters.
        budget, seen = self.max_tokens * 5, 0
        for chunk in stream:
            if seen > budget:
                stop = "length (harness cap)"
                stream.close()
                break
            if getattr(chunk, "usage", None):
                usage = {"input": chunk.usage.prompt_tokens or 0, "output": chunk.usage.completion_tokens or 0}
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
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw or "{}"}}
                                 for c in out]
        self.messages.append(msg)
        return Turn(text, think, out, stop, usage)


def make_agent(spec: str, tools, system: str, max_tokens: int):
    base, _, effort = spec.partition("@")
    provider, _, model = base.partition(":")
    effort = effort or "default"
    if provider == "anthropic":
        return AnthropicAgent(model, tools, system, max_tokens, effort)
    if provider == "openai":
        return OpenAIResponsesAgent(model, tools, system, max_tokens, "medium" if effort == "default" else effort)
    if provider == "hf":
        return ChatAgent(model, tools, system, max_tokens, os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1"),
                         os.environ["HF_TOKEN"], effort)
    if provider == "vllm":
        return ChatAgent(model, tools, system, max_tokens, os.environ["VLLM_BASE_URL"],
                         os.environ.get("VLLM_API_KEY", "none"), effort)
    raise ValueError(f"unknown provider in {spec!r}: use anthropic:, openai:, hf: or vllm:")


# ================================================================ episode

def run_episode(spec: str, base_url: str, task_id: str, max_turns: int = 12, max_tokens: int = 24000,
                wall_clock_s: float = 1800.0) -> dict:
    env = EnvSession(base_url)
    started = time.time()
    episode_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {"model": spec, "task_id": task_id, "episode_id": episode_id,
                              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                              "messages": [], "steps": [], "final": {"submitted": False, "plan": None, "grade": None},
                              "reward": 0.0, "usage": {"input_tokens": 0, "output_tokens": 0}, "turns": 0,
                              "end_reason": None, "errors": []}
    try:
        meta = env.reset(task_id, episode_id)
        system = meta["instructions"]
        opening = OPENING.format(situation=meta["situation"])
        record["messages"] += [{"role": "system", "content": system}, {"role": "user", "content": opening}]
        agent = make_agent(spec, env.tools, system, max_tokens)
        agent.user(opening)
        empty = 0
        for turn in range(max_turns):
            if time.time() - started > wall_clock_s:
                record["end_reason"] = "wall_clock"
                break
            last = turn == max_turns - 1
            if last:
                note = "This is your last turn: call submit_plan now with your best plan."
                agent.user(note)
                record["messages"].append({"role": "user", "content": note})
            t = agent.step()
            record["turns"] += 1
            record["usage"]["input_tokens"] += t.usage.get("input", 0)
            record["usage"]["output_tokens"] += t.usage.get("output", 0)
            record["messages"].append({"role": "assistant", "content": t.text, "reasoning": t.reasoning, "stop": t.stop,
                                       "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.raw or json.dumps(c.arguments)}
                                                      for c in t.calls]})
            if not t.calls:
                empty += 1
                if empty >= 2 or last:
                    record["end_reason"] = "no_tool_call"
                    break
                if t.stop.startswith("length") and t.reasoning:
                    # A turn cut off mid-thought loses its reasoning; hand the tail back as notes so the next turn
                    # continues instead of starting over.
                    agent.notes(t.reasoning[-NOTES_CHARS:])
                    nudge = ("Your last turn ran out of output tokens before calling a tool. Your notes from it are above. "
                             "Keep reasoning short now and call check_plan with your current draft, or submit_plan.")
                else:
                    nudge = ("No tool was called. Plans only count through the tools: call check_plan to test a draft "
                             "or submit_plan with your final plan.")
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
                    out, done, reward, meta = env.call(c.name, c.arguments)
                    if c.name in ("check_plan", "submit_plan"):
                        try:
                            res = json.loads(out)
                        except json.JSONDecodeError:
                            res = {"text": out}
                        record["steps"].append({"turn": turn + 1, "tool": c.name, "plan": plan_value(c.arguments.get("plan")),
                                                "result": res})
                    if done:
                        record["reward"] = float(reward or 0.0)
                        if c.name == "submit_plan":
                            record["final"] = {"submitted": True, "plan": plan_value(c.arguments.get("plan")),
                                               "grade": meta.get("grade"), "rubric": meta.get("rubric")}
                            record["end_reason"] = "submitted"
                        else:
                            record["end_reason"] = meta.get("end_reason", "server_ended")
                left = max_turns - turn - 1
                text = out if done else f"{out}\n(turns left: {left})"
                pairs.append((c, text))
                record["messages"].append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out})
            agent.results(pairs)
            if done:
                break
        else:
            record["end_reason"] = record["end_reason"] or "turn_limit"
        record["end_reason"] = record["end_reason"] or "turn_limit"
    except Exception as exc:  # provider or transport failure: recorded, scored as it stands
        record["errors"].append(f"{type(exc).__name__}: {str(exc)[:800]}")
        record["end_reason"] = record["end_reason"] or "error"
    finally:
        env.close()
        switches = getattr(locals().get("agent"), "provider_switches", None)
        if switches:
            record["provider_switches"] = switches
    record["seconds"] = round(time.time() - started, 1)
    return record
