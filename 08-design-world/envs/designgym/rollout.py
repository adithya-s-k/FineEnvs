"""Roll out real models against the environment, on every surface, and score them.

One harness, three providers, native tool calling everywhere:

    anthropic:<model>   Anthropic Messages API        (default anthropic:claude-sonnet-5)
    openai:<model>      OpenAI Responses API          (default openai:gpt-6-sol)
    openai-chat:<model> OpenAI chat completions, for models without the Responses API
    hf:<model>          Hugging Face router, OpenAI-compatible
                                                     (default hf:Qwen/Qwen3.6-35B-A3B:scaleway)
    vllm:<model>        your own OpenAI-compatible server (vLLM, SGLang) at VLLM_BASE_URL, e.g. an
                        Inference Endpoint's https://<host>/v1; key from VLLM_API_KEY, else HF_TOKEN

Each episode resets the environment over OpenEnv's WebSocket (`MCPToolClient`),
discovers the surface's tools with `list_tools()`, hands them to the model as
functions, and executes every call with `CallToolAction`, so the model drives
exactly what a trainer would. The reference image (reference mode) and any
image a tool returns (view_canvas, view_assets, screenshot) go back to the
model as real image content. On the GUI surface a fresh screenshot follows
every turn, like any computer-use harness.

Keys are read from the environment or the nearest `.env` up the tree
(ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN). Without `--base-url` a local
server is started in-process.

    python rollout.py --tasks 3 --surfaces tools,html,gui
    python rollout.py --models anthropic:claude-sonnet-5 --surfaces gui --tasks 5 --modes reference,description
    python rollout.py --base-url https://<space>.hf.space --models hf:Qwen/Qwen3.6-35B-A3B:scaleway
    python rollout.py --evalset ../../bench/evalsets/recreation-cua-v1.json --subset lite --modes reference,description

With `--evalset` the episodes are the frozen set's tasks (by task id, the same
tasks on every surface and mode), and the summary adds reward per difficulty
tier with 95% bootstrap intervals.

Writes `<out>/trace.jsonl` (every turn of every episode), `<out>/summary.json`,
`<out>/summary.md`, and a final render next to the target for each episode.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import socket
import statistics
import sys
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from designgym.server.episodes import SKINS  # noqa: E402  (every editor layout)
from designgym.cua_format import PLACEHOLDER  # noqa: E402

DEFAULT_MODELS = ["anthropic:claude-sonnet-5", "openai:gpt-6-sol", "hf:Qwen/Qwen3.6-35B-A3B:scaleway"]
# The model is told its action budget, and the environment enforces it; the harness gives it the same
# number of turns, so nothing is cut short without warning (a turn makes one or more actions).
DEFAULT_TURNS = {"tools": 60, "html": 20, "gui": 200}
REQUEST_TIMEOUT = 300.0  # seconds per model request; set by --request-timeout
VERBOSE = False
NUDGE = ("Continue working with the tools. When the design matches, call the submit tool; "
         "do not stop without submitting.")


# ================================================================== env


def load_env_file(explicit: str | None) -> str | None:
    from dotenv import load_dotenv

    candidates = [Path(explicit)] if explicit else []
    for base in (Path.cwd(), _HERE):
        for p in [base, *base.parents]:
            candidates.append(p / ".env")
    for c in candidates:
        if c.is_file():
            load_dotenv(c, override=False)
            return str(c)
    return None


def start_local_server() -> str:
    import uvicorn

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    os.environ["PORT"] = str(port)
    from designgym.server.app import create_server

    server = uvicorn.Server(uvicorn.Config(create_server(), host="127.0.0.1", port=port,
                                           log_level="warning", ws_ping_interval=60,
                                           ws_ping_timeout=300))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    return f"http://127.0.0.1:{port}"


@dataclass
class ToolOutcome:
    text: str
    images: list[tuple[str, str]]  # (media type, base64)
    is_error: bool
    done: bool = False
    reward: float | None = None
    grade: dict | None = None


def run_tool(env, name: str, args: dict) -> ToolOutcome:
    from openenv.core.env_server.mcp_types import CallToolAction

    out = env.step(CallToolAction(tool_name=name, arguments=args))
    obs = out.observation
    if obs.error is not None:
        return ToolOutcome(f"Error: {obs.error.message}", [], True, out.done, out.reward,
                           (obs.metadata or {}).get("grade"))
    res = obs.result or {}
    texts, images = [], []
    for c in res.get("content") or []:
        if c.get("type") == "text":
            texts.append(c.get("text", ""))
        elif c.get("type") == "image":
            images.append((c.get("mimeType") or "image/png", c["data"]))
    return ToolOutcome("\n".join(texts) or ("(image)" if images else "(no output)"), images,
                       bool(res.get("is_error")), out.done, out.reward,
                       (obs.metadata or {}).get("grade"))


# ================================================================== providers


@dataclass
class Call:
    id: str
    name: str
    args: dict
    parse_error: str | None = None


@dataclass
class Turn:
    text: str
    calls: list[Call]
    stop: str
    usage: dict = field(default_factory=dict)


class AnthropicAgent:
    """Messages API with client tools. The whole response content (thinking
    blocks included) is appended back every turn, images travel inside
    tool_result blocks, and top-level cache_control caches the growing prefix."""

    def __init__(self, model: str, tools: list, system: str, max_tokens: int, thinking: str = "default"):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=3, timeout=REQUEST_TIMEOUT)
        self.model = model
        # off: no thinking at all; low: adaptive thinking at low effort; default: the model's own default.
        self.extra = {"off": {"thinking": {"type": "disabled"}},
                      "low": {"extra_body": {"output_config": {"effort": "low"}}}}.get(thinking, {})
        if model.startswith("claude-opus-5-5"):
            # Opus 5.5 can't disable thinking (effort is the only control), and it binds thinking
            # blocks to the exact history. Pruning old screenshots edits that history, which is a 400
            # on accounts created from 2026-08-31, so stale blocks are dropped instead.
            body: dict = {"thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}}
            if thinking in ("off", "low"):
                body["output_config"] = {"effort": "low"}
            self.extra = {"extra_body": body, "extra_headers": {"anthropic-beta": "thinking-binding-controls-2026-08-01"}}
        self.system = system
        self.max_tokens = max_tokens
        self.tools = [{"name": t.name, "description": t.description or "",
                       "input_schema": t.input_schema or {"type": "object", "properties": {}}}
                      for t in tools]
        self.messages: list[dict] = []
        self._pending: list[dict] = []

    def user(self, text: str, images=()):
        content = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": d}}
                   for m, d in images]
        content.append({"type": "text", "text": text})
        self._pending.extend(content)

    def results(self, pairs: list[tuple[Call, ToolOutcome]]):
        trailing: list[dict] = []
        for call, out in pairs:
            content: list[dict] = [{"type": "text", "text": out.text}]
            images = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": d}}
                      for m, d in out.images]
            # The API takes only text in an error result: a screenshot that came with an error follows the results.
            if out.is_error:
                trailing += images
            else:
                content += images
            self._pending.append({"type": "tool_result", "tool_use_id": call.id,
                                  "content": content, "is_error": out.is_error})
        self._pending.extend(trailing)

    def _flush(self):
        if self._pending:
            # tool_result blocks must lead the user turn.
            self._pending.sort(key=lambda b: 0 if b["type"] == "tool_result" else 1)
            self.messages.append({"role": "user", "content": self._pending})
            self._pending = []

    def step(self) -> Turn:
        self._flush()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},
            **self.extra,
        )
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [Call(b.id, b.name, dict(b.input or {})) for b in resp.content if b.type == "tool_use"]
        u = resp.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                 "reasoning_chars": sum(len(getattr(b, "thinking", "") or "") for b in resp.content if b.type == "thinking")}
        return Turn(text, calls, resp.stop_reason or "", usage)

    def prune_images(self, keep: int, max_images: int | None = None):
        _prune(self.messages, keep, anthropic_format=True, max_images=max_images)


class OpenAIAgent:
    """Chat completions with function tools (OpenAI, or any OpenAI-compatible
    router such as Hugging Face's). Tool messages carry text only, so images a
    tool returns follow in a user message right after the tool results."""

    def __init__(self, model: str, tools: list, system: str, max_tokens: int, base_url=None,
                 api_key=None, token_param="max_completion_tokens", thinking: str = "default"):
        from openai import OpenAI

        # Router providers rate-limit bursts (429 "insufficient quota" per minute): back off and retry.
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=6 if base_url else 2,
                             timeout=REQUEST_TIMEOUT)
        self.model = model
        self.extra = thinking_switch(self.client, model, thinking)
        self.max_tokens = max_tokens
        self.token_param = token_param
        self.tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": t.input_schema or {"type": "object", "properties": {}}}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self._images: list[tuple[str, str]] = []
        self._texts: list[str] = []

    def user(self, text: str, images=()):
        self._texts.append(text)
        self._images.extend(images)

    def results(self, pairs):
        for call, out in pairs:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": out.text})
            if out.images:
                self._images.extend(out.images)
                self._texts.append(f"Image returned by {call.name}.")

    def _flush(self):
        if self._texts or self._images:
            parts = [{"type": "image_url", "image_url": {"url": f"data:{m};base64,{d}"}}
                     for m, d in (_compact(m, d) for m, d in self._images)]
            parts.append({"type": "text", "text": "\n".join(self._texts) or "Continue."})
            self.messages.append({"role": "user", "content": parts})
            self._images, self._texts = [], []

    def step(self) -> Turn:
        import openai

        self._flush()
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools,
                **{self.token_param: self.max_tokens}, **self.extra)
        except openai.APIStatusError as exc:
            if exc.status_code not in (400, 413) or "not supported" in str(exc):
                raise
            # 413: the provider caps the request body. A bare 400 deep into an episode is some
            # providers' way of rejecting a long request (Novita gives no reason): either way, keep
            # only the newest image and try once more before giving up on the episode.
            _prune(self.messages, 1, anthropic_format=False, force=True)
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools,
                **{self.token_param: self.max_tokens}, **self.extra)
        choice = resp.choices[0]
        msg = choice.message
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        calls = []
        if msg.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function", "function": {
                "name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in msg.tool_calls]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    err = None if isinstance(args, dict) else "arguments must be a JSON object"
                except json.JSONDecodeError as exc:
                    args, err = {}, f"arguments are not valid JSON: {exc}"
                calls.append(Call(tc.id, tc.function.name, args if isinstance(args, dict) else {}, err))
        self.messages.append(entry)
        u = resp.usage
        usage = {"input": getattr(u, "prompt_tokens", 0) or 0, "output": getattr(u, "completion_tokens", 0) or 0,
                 "reasoning_chars": len(_reasoning_text(msg))}
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None):
            usage["cache_read"] = details.cached_tokens
        return Turn(msg.content or "", calls, choice.finish_reason or "", usage)

    def prune_images(self, keep: int, max_images: int | None = None):
        _prune(self.messages, keep, anthropic_format=False, max_images=max_images)


class OpenAIResponsesAgent:
    """OpenAI Responses API with function tools. Reasoning models such as
    gpt-6-sol only accept function tools here (chat completions rejects tools
    while reasoning is on). Runs stateless (`store=False`) with encrypted
    reasoning carried in the input, so old screenshots can still be pruned."""

    def __init__(self, model: str, tools: list, system: str, max_tokens: int, effort: str = "medium"):
        from openai import OpenAI

        self.client = OpenAI(max_retries=2, timeout=REQUEST_TIMEOUT)
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.effort = effort
        self.tools = [{"type": "function", "name": t.name, "description": t.description or "",
                       "parameters": t.input_schema or {"type": "object", "properties": {}}}
                      for t in tools]
        self.input: list = []
        self._images: list[tuple[str, str]] = []
        self._texts: list[str] = []

    def user(self, text: str, images=()):
        self._texts.append(text)
        self._images.extend(images)

    def results(self, pairs):
        for call, out in pairs:
            self.input.append({"type": "function_call_output", "call_id": call.id, "output": out.text})
            if out.images:
                self._images.extend(out.images)
                self._texts.append(f"Image returned by {call.name}.")

    def _flush(self):
        if self._texts or self._images:
            parts = [{"type": "input_image", "image_url": f"data:{m};base64,{d}"} for m, d in self._images]
            parts.append({"type": "input_text", "text": "\n".join(self._texts) or "Continue."})
            self.input.append({"role": "user", "content": parts})
            self._images, self._texts = [], []

    def step(self) -> Turn:
        self._flush()
        import openai

        try:
            resp = self.client.responses.create(
                model=self.model, instructions=self.system, input=self.input, tools=self.tools,
                max_output_tokens=self.max_tokens, store=False, include=["reasoning.encrypted_content"],
                reasoning={"effort": self.effort})
        except openai.BadRequestError as exc:
            if self.effort != "none" or "'none' is not supported" not in str(exc):
                raise
            self.effort = "low"  # e.g. gpt-6-astra: low is its least reasoning
            return self.step()
        self.input.extend(resp.output)
        text, calls = "", []
        for item in resp.output:
            if item.type == "message":
                text += "".join(getattr(c, "text", "") for c in item.content)
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                    err = None if isinstance(args, dict) else "arguments must be a JSON object"
                except json.JSONDecodeError as exc:
                    args, err = {}, f"arguments are not valid JSON: {exc}"
                calls.append(Call(item.call_id, item.name, args if isinstance(args, dict) else {}, err))
        u = resp.usage
        usage = {"input": getattr(u, "input_tokens", 0) or 0, "output": getattr(u, "output_tokens", 0) or 0}
        out_details = getattr(u, "output_tokens_details", None)
        usage["reasoning_tokens"] = getattr(out_details, "reasoning_tokens", 0) or 0
        details = getattr(u, "input_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None):
            usage["cache_read"] = details.cached_tokens
        stop = resp.status or ""
        if getattr(resp, "incomplete_details", None):
            stop = f"incomplete:{resp.incomplete_details.reason}"
        return Turn(text, calls, stop, usage)

    def prune_images(self, keep: int, max_images: int | None = None):
        # The Responses API names its text parts input_text.
        _prune([m for m in self.input if isinstance(m, dict)], keep, anthropic_format=False,
               text_type="input_text", max_images=max_images)


def _compact(mime: str, data: str, limit: int = 150_000) -> tuple[str, str]:
    """Re-encode a large PNG (a screenshot, a render) as JPEG for OpenAI-compatible routers, some of
    whose providers cap the request body; the image the model sees is the same size."""
    if mime != "image/png" or len(data) < limit:
        return mime, data
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()


def _prune(messages: list, keep: int, anthropic_format: bool, text_type: str = "text", force: bool = False,
           max_images: int | None = None):
    """Keep only the newest `keep` images in the history; older ones become a note.

    Screenshots dominate a GUI episode's tokens and stale ones carry no
    information. Pruning happens in batches so the cached prefix is rewritten
    rarely rather than on every turn. With `max_images` (the `--image-window` mode, as the Tinker adapter's
    ImageWindow), the opening user message (the target) is never touched and up to `max_images` build up before
    a drop, so the prompt prefix up to the first dropped screenshot survives and the server's prompt cache holds.
    """
    first_user = next((k for k, m in enumerate(messages) if m.get("role") == "user"), None)
    slots = []
    for k, m in enumerate(messages):
        if max_images is not None and k == first_user:
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            b = block if isinstance(block, dict) else None
            if b is None:
                continue
            if b.get("type") in ("image", "image_url", "input_image"):
                slots.append((content, i))
            elif anthropic_format and b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                for j, inner in enumerate(b["content"]):
                    if inner.get("type") == "image":
                        slots.append((b["content"], j))
    if len(slots) <= (max_images if max_images is not None else keep + (0 if force else 4)):
        return
    for content, i in slots[: len(slots) - keep]:
        content[i] = {"type": text_type, "text": PLACEHOLDER}  # the same words as the SFT windows and the RL adapter


THINKING = ("default", "off", "low")
_SWITCHES: dict = {}
_SWITCH_LOCK = threading.Lock()
# Ways open-model providers turn thinking down, most specific first. Which one a provider
# honours varies by model *and* provider (novita honours the chat-template flag for
# Qwen3.8 but ignores reasoning_effort; scaleway is the reverse for Qwen3.6), so each is
# tried once per run and the first that yields no reasoning is used.
_OFF_SWITCHES = [{"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                 {"reasoning_effort": "none"}, {"reasoning_effort": "low"}]


def _reasoning_text(msg) -> str:
    text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    if not text and msg.content and "</think>" in msg.content:
        text = msg.content.split("</think>")[0]
    return text if isinstance(text, str) else ""


def thinking_switch(client, model: str, thinking: str) -> dict:
    """Request kwargs that put an OpenAI-compatible model in `thinking` mode (probed once per model)."""
    if thinking == "default":
        return {}
    if thinking == "low":
        return {"reasoning_effort": "low"}
    with _SWITCH_LOCK:
        if model in _SWITCHES:
            return _SWITCHES[model]
        tool = [{"type": "function", "function": {"name": "ping", "description": "Reply to the user.",
                 "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}]
        probe = [{"role": "user", "content": "Call ping with the number of letters in the word canvas."}]
        best, best_chars, tried = {}, None, []
        for kw in _OFF_SWITCHES:
            try:
                r = client.chat.completions.create(model=model, messages=probe, tools=tool, max_tokens=800, **kw)
            except Exception as exc:  # noqa: BLE001 - a provider rejecting a switch is expected
                tried.append(f"{kw}: rejected ({type(exc).__name__})")
                continue
            chars = len(_reasoning_text(r.choices[0].message))
            tried.append(f"{kw}: {chars} reasoning chars")
            if best_chars is None or chars < best_chars:
                best, best_chars = kw, chars
            if chars == 0:
                break
        if best_chars:
            print(f"  warning: could not turn thinking fully off for {model}; using {best} "
                  f"({best_chars} reasoning chars on a probe). Tried: {'; '.join(tried)}", flush=True)
        elif VERBOSE:
            print(f"  thinking off for {model} via {best}", flush=True)
        _SWITCHES[model] = best
        return best


def coordinates_for(spec: str, choice: str = "auto") -> str:
    """GUI coordinate system for a model: open VLMs on the HF router point on a 0-1000 grid
    (Qwen-VL, Gemma, Muse); Claude and GPT point in screenshot pixels."""
    if choice != "auto":
        return choice
    return "normalized" if spec.startswith(("hf:", "vllm:")) else "pixels"


def split_spec(spec: str, default_thinking: str = "default") -> tuple[str, str]:
    """'hf:Qwen/Qwen3.8-27B:novita@off' -> ('hf:Qwen/Qwen3.8-27B:novita', 'off')."""
    base, _, mode = spec.partition("@")
    mode = mode or default_thinking
    if mode not in THINKING:
        raise ValueError(f"unknown thinking mode {mode!r} in {spec!r}; use one of {', '.join(THINKING)}")
    return base, mode


def make_agent(spec: str, tools, system: str, max_tokens: int):
    spec, thinking = split_spec(spec)
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicAgent(model, tools, system, max_tokens, thinking=thinking)
    if provider == "openai":
        return OpenAIResponsesAgent(model, tools, system, max_tokens,
                                    effort={"off": "none", "low": "low"}.get(thinking, "medium"))
    if provider == "openai-chat":
        return OpenAIAgent(model, tools, system, max_tokens, thinking=thinking)
    if provider == "hf":
        return OpenAIAgent(model, tools, system, max_tokens,
                           base_url=os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1"),
                           api_key=os.environ["HF_TOKEN"], token_param="max_tokens", thinking=thinking)
    if provider == "vllm":
        return OpenAIAgent(model, tools, system, max_tokens, base_url=os.environ["VLLM_BASE_URL"],
                           api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("HF_TOKEN") or "none",
                           token_param="max_tokens", thinking=thinking)
    raise ValueError(f"unknown provider in {spec!r}; use anthropic:, openai:, hf: or vllm:")


# ================================================================== episodes


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def save_frame(frames: Path | None, n: int | str, image: tuple[str, str] | bytes | None) -> str | None:
    """Write one step's screen (a GUI screenshot or the canvas render) as frames/<stem>/<n>.<ext>; returns the path
    relative to the run directory, for the viewer to step through."""
    if frames is None or not image:
        return None
    if isinstance(image, tuple):
        media, data = image
        raw, ext = base64.b64decode(data), media.split("/")[-1].replace("jpeg", "jpg")
    else:
        raw, ext = image, "png"
    frames.mkdir(parents=True, exist_ok=True)
    path = frames / (f"{n:03d}.{ext}" if isinstance(n, int) else f"{n}.{ext}")
    path.write_bytes(raw)
    return str(path.relative_to(frames.parents[1]))


def run_episode(base_url: str, spec: str, surface: str, mode: str, split: str, index,
                args, out_dir: Path, entry: dict | None = None, skin: str | None = None,
                episode: dict | None = None, settings: dict | None = None) -> dict:
    """One episode. `index` is a split position, or with `entry` (a frozen-set task) its position in the set.

    With `episode` (a v2 frozen episode: task, editor, seed, look and workspace) and `settings` (the set's run
    settings), the environment is reset to exactly that episode, so every model meets the same one."""
    from openenv.core.env_server.mcp_types import CallToolAction  # noqa: F401
    from openenv.core.mcp_client import MCPToolClient

    t0 = time.time()
    record: dict[str, Any] = {"model": spec, "surface": surface, "mode": mode, "split": split,
                              "index": index, "turns": [], "tool_calls": 0, "tool_errors": 0,
                              "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}}
    skin = skin or SKINS[index % len(SKINS)]
    if entry:
        record.update(tier=entry["tier"], difficulty_score=entry["score"], evalset_task=entry["task_id"])
        where = {"task_id": entry["task_id"]}
    else:
        where = {"split": split, "index": index}
    extra: dict[str, Any] = {}
    if episode:  # a frozen v2 episode: its own task, editor, seed, look and workspace
        settings = settings or {}
        skin = episode["skin"]
        where = {"task_id": episode["task_id"]}
        extra = {k: episode[k] for k in ("seed", "theme", "layout", "file_dialog", "workspace") if episode.get(k) is not None}
        if settings.get("budget_note"):
            extra["budget_note"] = settings["budget_note"]
        record.update(evalset_episode=episode["id"], seed=episode["seed"], theme=episode.get("theme"),
                      layout=episode.get("layout"), file_dialog=episode.get("file_dialog"),
                      workspace=episode.get("workspace"), real_app=episode.get("real", False))
    # a real app's first episode can wait minutes for its desktop to start: give the reset room
    timeout = getattr(args, "message_timeout", 300) if not (episode and episode.get("real")) else max(
        getattr(args, "message_timeout", 300), 900)
    with MCPToolClient(base_url=base_url, message_timeout_s=timeout).sync() as env:
        coords = coordinates_for(spec, args.coordinates) if surface == "gui" else "pixels"
        r = env.reset(**where, surface=surface, mode=mode, skin=skin, coordinates=coords,
                      max_steps=args.max_steps or args.max_turns or DEFAULT_TURNS[surface], **extra)
        meta = r.observation.metadata
        record.update(task_id=meta["task_id"], episode_id=meta["episode_id"], skin=skin,
                      prompt_version=meta.get("prompt_version"), coordinates=meta.get("coordinates", "pixels"))
        tools = [t for t in env.list_tools(use_cache=False) if t.name != "start_task"]
        # the system prompt and tool schemas exactly as the model got them, so a published row is lossless
        record["system"] = meta["instructions"]
        record["tools"] = [{"name": t.name, "description": t.description, "input_schema": getattr(t, "input_schema", None)
                            or getattr(t, "inputSchema", None)} for t in tools]
        agent = make_agent(spec, tools, meta["instructions"], args.max_tokens)
        frames = None if args.no_frames else (out_dir / "frames" / (
            f"{spec.replace(':', '_').replace('/', '_')}__{surface}__{mode}__{meta['task_id']}"
            + (f"__{skin}" if surface == "gui" else "")))
        canvas = lambda: fetch(f"{base_url}/api/episodes/{meta['episode_id']}/render.png?size=768")  # noqa: E731
        opening = []
        if mode == "reference":
            opening.append(("image/jpeg", meta["reference_jpeg_b64"]))
        if surface == "gui":
            shot = run_tool(env, "screenshot", {})
            opening.extend(shot.images)
            record["frame0"] = save_frame(frames, 0, shot.images[-1] if shot.images else None)
        else:
            record["frame0"] = save_frame(frames, 0, canvas())
        intro = {"reference": "This is the target design. Recreate it.",
                 "description": "Build the design from the brief in your instructions."}[mode]
        if surface == "gui":
            intro += (" The second image is the editor as it is now." if mode == "reference"
                      else " The image is the editor as it is now.")
        agent.user(intro, opening)

        done, grade, nudges = False, None, 0
        max_turns = args.max_turns or DEFAULT_TURNS[surface]
        deadline = t0 + args.episode_timeout
        for turn_no in range(max_turns):
            if time.time() > deadline:
                record["error"] = f"episode exceeded {args.episode_timeout}s"
                break
            try:
                t_req = time.time()
                turn = agent.step()
                if VERBOSE:
                    print(f"    [{spec} {surface} {index}] turn {turn_no} {time.time() - t_req:.1f}s "
                          f"calls={[c.name for c in turn.calls]} stop={turn.stop}", flush=True)
            except Exception as exc:  # provider failure: record, stop this episode
                record["error"] = f"{type(exc).__name__}: {exc}"[:800]
                break
            for k, v in turn.usage.items():
                record["usage"][k] = record["usage"].get(k, 0) + (v or 0)
            model_s = round(time.time() - t_req, 2)
            entry = {"turn": turn_no, "text": turn.text, "stop": turn.stop, "calls": [], "model_s": model_s}
            t_env = time.time()
            if not turn.calls:
                record["turns"].append(entry)
                nudges += 1
                if nudges > 2:
                    break
                agent.user(NUDGE)
                continue
            pairs = []
            for call in turn.calls:
                if call.parse_error:
                    out = ToolOutcome(f"Error: {call.parse_error}", [], True)
                else:
                    try:
                        out = run_tool(env, call.name, call.args)
                    except Exception as exc:
                        out = ToolOutcome(f"Error: {exc}", [], True)
                record["tool_calls"] += 1
                record["tool_errors"] += int(out.is_error)
                # full arguments (an HTML page, long copy), so an episode can be replayed and viewed step by step;
                # every image the tool returned (view_canvas, view_reference, screenshot), as the model saw it
                k = len(entry["calls"])
                entry["calls"].append({"name": call.name, "args": call.args, "ok": not out.is_error,
                                       "result": out.text[:400], "images": len(out.images),
                                       "image_paths": [p for j, im in enumerate(out.images)
                                                       if (p := save_frame(frames, f"t{turn_no:03d}c{k}i{j}", im))]})
                pairs.append((call, out))
                if out.done:
                    done, grade = True, out.grade
                    break
            record["turns"].append(entry)
            if done:
                break
            if surface == "gui" and not any(c.name == "screenshot" for c, _ in pairs):
                shot = run_tool(env, "screenshot", {})
                if shot.done:
                    done, grade = True, shot.grade
                    break
                pairs[-1][1].images.extend(shot.images)
            if surface == "gui":
                shots = [im for c, o in pairs for im in o.images]
                entry["frame"] = save_frame(frames, turn_no + 1, shots[-1] if shots else None)
            else:
                entry["frame"] = save_frame(frames, turn_no + 1, canvas())
            agent.results(pairs)
            entry["env_s"] = round(time.time() - t_env, 2)  # tool calls + screenshot/canvas for this turn
            if args.image_window:
                agent.prune_images(args.image_window[1], max_images=args.image_window[0])
            else:
                agent.prune_images(args.keep_images)
        if not done:
            out = run_tool(env, "submit", {})
            grade = out.grade
            record["auto_submitted"] = True
    record["reward"] = grade["total"] if grade else None
    record["grade"] = {k: grade[k] for k in ("total", "structure", "visual", "coverage", "order",
                                              "background", "n_gt", "n_submitted", "n_matched")} if grade else None
    record["seconds"] = round(time.time() - t0, 1)
    record["n_turns"] = len(record["turns"])
    stem = (f"{spec.replace(':', '_').replace('/', '_')}__{surface}__{mode}__{record.get('task_id', index)}"
            + (f"__{skin}" if surface == "gui" else ""))
    # The submitted document itself, so any later reward version can re-score the run without the models.
    info = fetch(f"{base_url}/api/episodes/{record.get('episode_id')}")
    if info:
        try:
            record["final_doc"] = json.loads(info)["doc"]
        except (ValueError, KeyError):
            pass
    render = fetch(f"{base_url}/api/episodes/{record.get('episode_id')}/render.png?size=768")
    if render:
        (out_dir / "renders").mkdir(parents=True, exist_ok=True)
        (out_dir / "renders" / f"{stem}.png").write_bytes(render)
        record["render"] = f"renders/{stem}.png"
    if mode == "reference" and record.get("episode_id"):
        ref = fetch(f"{base_url}/api/episodes/{record['episode_id']}/reference.jpg?size=768")
        if ref:
            (out_dir / "renders" / f"{record['task_id']}__target.jpg").write_bytes(ref)
    return record


# ================================================================== main


def summarise(records: list[dict]) -> tuple[dict, str]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["model"], r["surface"], r["mode"]), []).append(r)
    rows, table = [], ["| model | surface | mode | n | mean reward | min | max | turns | tool calls | errors | failed "
                       "| reasoning / turn |",
                       "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (model, surface, mode), rs in sorted(groups.items()):
        rewards = [r["reward"] if r["reward"] is not None else 0.0 for r in rs]
        failed = sum(1 for r in rs if r.get("error"))
        row = {"model": model, "surface": surface, "mode": mode, "n": len(rs),
               "mean_reward": round(statistics.mean(rewards), 4), "min": round(min(rewards), 4),
               "max": round(max(rewards), 4),
               "mean_turns": round(statistics.mean(r["n_turns"] for r in rs), 1),
               "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rs), 1),
               "tool_errors": sum(r["tool_errors"] for r in rs), "failed": failed,
               "input_tokens": sum(r["usage"].get("input", 0) for r in rs),
               "output_tokens": sum(r["usage"].get("output", 0) for r in rs)}
        # How much the model actually reasoned, so a "@off" run can be checked: tokens where the
        # API reports them (OpenAI), characters of thinking text otherwise (Anthropic, HF router).
        turns = max(1, sum(r["n_turns"] for r in rs))
        rtok = sum(r["usage"].get("reasoning_tokens", 0) for r in rs)
        rchr = sum(r["usage"].get("reasoning_chars", 0) for r in rs)
        row["reasoning_tokens"], row["reasoning_chars"] = rtok, rchr
        reasoning = f"{rtok / turns:.0f} tok" if rtok else f"{rchr / turns:.0f} chars"
        rows.append(row)
        table.append(f"| {model} | {surface} | {mode} | {row['n']} | {row['mean_reward']:.3f} | {row['min']:.3f} | "
                     f"{row['max']:.3f} | {row['mean_turns']} | {row['mean_tool_calls']} | {row['tool_errors']} | {failed} "
                     f"| {reasoning} |")
    tiers = _by_tier(records)
    if tiers:
        table += ["", "| model | surface | mode | tier | n | mean reward | 95% CI |", "|---|---|---|---|---:|---:|---|"]
        table += [f"| {t['model']} | {t['surface']} | {t['mode']} | {t['tier']} | {t['n']} | {t['mean_reward']:.3f} | "
                  f"{t['ci95'][0]:.3f}–{t['ci95'][1]:.3f} |" for t in tiers]
    return {"groups": rows, "tiers": tiers}, "\n".join(table)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    import random

    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)


def _by_tier(records: list[dict]) -> list[dict]:
    from designgym.core.difficulty import TIERS

    groups: dict[tuple, list[float]] = {}
    for r in records:
        if r.get("tier"):
            rw = r["reward"] if r["reward"] is not None else 0.0
            groups.setdefault((r["model"], r["surface"], r["mode"], r["tier"]), []).append(rw)
            groups.setdefault((r["model"], r["surface"], r["mode"], "all"), []).append(rw)
    order = {t: i for i, t in enumerate((*TIERS, "all"))}
    return [{"model": m, "surface": s, "mode": mode, "tier": t, "n": len(v),
             "mean_reward": round(statistics.fmean(v), 4), "ci95": bootstrap_ci(v)}
            for (m, s, mode, t), v in sorted(groups.items(), key=lambda kv: (*kv[0][:3], order[kv[0][3]]))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="running server; default: start one in-process")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma list of provider:model, each optionally suffixed @off, @low or @default")
    ap.add_argument("--coordinates", default="auto", choices=("auto", "pixels", "normalized"),
                    help="GUI coordinate system; auto: 0-1000 grid for hf: models, pixels otherwise")
    ap.add_argument("--thinking", default="default", choices=THINKING,
                    help="reasoning for models without an @ suffix: off (none at all), low, or the model's default")
    ap.add_argument("--surfaces", default="tools,html,gui")
    ap.add_argument("--modes", default="reference")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", type=int, default=3, help="tasks per (model, surface, mode)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=7, help="task index step, to spread the sample")
    ap.add_argument("--evalset", default=None, help="frozen evaluation set JSON; replaces --split/--tasks")
    ap.add_argument("--subset", default="full", help="evalset subset: full, lite, or a named nested subset (e.g. core30)")
    ap.add_argument("--limit", type=int, default=None, help="evalset: only this many tasks, balanced across tiers")
    ap.add_argument("--task-ids", default=None, help="evalset: only these task ids (comma list)")
    ap.add_argument("--tiers", default=None, help="only these difficulty tiers, e.g. hard or easy,medium")
    ap.add_argument("--skins", default=None, help="GUI only: run each task in each of these layouts "
                                                  "(e.g. figma,canva,photoshop,slides,powerpoint,inkscape); "
                                                  "default one per task")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="environment step limit override")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--keep-images", type=int, default=4)
    ap.add_argument("--image-window", default=None, type=lambda v: tuple(int(x) for x in v.split(",")),
                    help="MAX,KEEP: keep the target, let up to MAX screenshots build up, then drop to the newest KEEP "
                         "(cache-friendly, as the Tinker adapter); default: --keep-images behaviour")
    ap.add_argument("--no-frames", action="store_true", help="don't save each step's screenshot / canvas for the viewer")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--request-timeout", type=float, default=300.0, help="seconds per model request")
    ap.add_argument("--episode-timeout", type=float, default=2400.0, help="wall-clock budget per episode")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    global REQUEST_TIMEOUT, VERBOSE
    REQUEST_TIMEOUT, VERBOSE = args.request_timeout, args.verbose
    env_file = load_env_file(args.env_file)
    models = []
    for m in (m.strip() for m in args.models.split(",") if m.strip()):
        base, mode = split_spec(m, args.thinking)
        models.append(base if mode == "default" else f"{base}@{mode}")  # the mode is part of the model's label
    need = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openai-chat": "OPENAI_API_KEY",
            "hf": "HF_TOKEN", "vllm": "VLLM_BASE_URL"}
    missing = [m for m in models if not os.environ.get(need.get(m.split(":")[0], ""), "")]
    if missing:
        print(f"missing API keys for: {', '.join(missing)} (env file: {env_file})", file=sys.stderr)
        return 2
    base_url = args.base_url or start_local_server()
    out = Path(args.out or (_HERE.parents[1] / "results" / "rollouts" / time.strftime("%Y%m%d-%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    entries: dict = {}
    evalset_id = None
    if args.evalset:
        from designgym.core import evalset as ev

        record = ev.load(args.evalset)
        evalset_id = record["evalset_id"]
        chosen = ev.subset_tasks(record, args.subset)
        if args.tiers:
            chosen = [e for e in chosen if e["tier"] in args.tiers.split(",")]
        if args.task_ids:  # re-run particular episodes, e.g. ones a provider failed
            wanted = set(args.task_ids.split(","))
            chosen = [e for e in chosen if e["task_id"] in wanted]
        if args.limit:  # a smaller run that still spans the tiers: take them round-robin
            by_tier: dict[str, list] = {}
            for e in chosen:
                by_tier.setdefault(e["tier"], []).append(e)
            order = [t for t in ("easy", "medium", "hard") if t in by_tier] + sorted(set(by_tier) - {"easy", "medium", "hard"})
            picked = []
            while len(picked) < min(args.limit, len(chosen)):
                for t in order:
                    if by_tier[t] and len(picked) < args.limit:
                        picked.append(by_tier[t].pop(0))
            chosen = picked
        entries = {e["task_id"]: (pos, e) for pos, e in enumerate(chosen)}
        refs = list(entries)
        print(f"evalset {record['name']} ({evalset_id[:12]}), subset {args.subset}: {len(refs)} tasks")
    else:
        refs = [args.start + k * args.stride for k in range(args.tasks)]
    skins = args.skins.split(",") if args.skins else [None]
    jobs = [(m, s, mode, ref, sk)
            for m in models for s in args.surfaces.split(",") for mode in args.modes.split(",")
            for ref in refs for sk in (skins if s == "gui" else [None])]
    # Resume: an episode that already finished cleanly in this directory is kept;
    # one that errored is run again, and the newest record wins in the summary.
    previous = []
    trace_path = out / "trace.jsonl"
    if trace_path.exists():
        previous = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    key = lambda r: (r["model"], r["surface"], r["mode"], r.get("evalset_task") or r["index"],  # noqa: E731
                     r.get("skin_forced"))
    done_keys = {key(r) for r in previous if not r.get("error")}
    todo = [j for j in jobs if j not in done_keys]
    print(f"{len(jobs)} episodes against {base_url} -> {out}"
          + (f" ({len(jobs) - len(todo)} already done, {len(todo)} to run)" if previous else ""))
    jobs = todo
    trace = open(trace_path, "a")
    lock = threading.Lock()
    records = []

    def job(j):
        m, s, mode, ref, sk = j
        pos, entry = entries.get(ref, (ref, None))
        idx = pos
        try:
            rec = run_episode(base_url, m, s, mode, args.split, pos, args, out, entry, skin=sk)
            rec["skin_forced"] = sk
        except Exception as exc:
            rec = {"model": m, "surface": s, "mode": mode, "index": idx, "reward": None, "skin_forced": sk,
                   **({"tier": entry["tier"], "evalset_task": entry["task_id"]} if entry else {}),
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:],
                   "turns": [], "n_turns": 0, "tool_calls": 0, "tool_errors": 0, "usage": {}}
        with lock:
            trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace.flush()
            records.append(rec)
            rw = rec["reward"]
            print(f"  {m:42s} {s:5s} {mode:11s} {rec.get('task_id', idx)!s:18s} "
                  f"reward {'—' if rw is None else f'{rw:.3f}'}  turns {rec['n_turns']:3d}  "
                  f"calls {rec['tool_calls']:3d}  {rec.get('seconds', 0)}s"
                  + (f"  ERROR {rec['error'][:120]}" if rec.get("error") else ""), flush=True)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(job, jobs))
    trace.close()
    latest = {}
    for r in previous + records:
        latest[key(r)] = r
    summary, table = summarise(list(latest.values()))
    summary.update(base_url=base_url, split=args.split, started=out.name, models=models,
                   surfaces=args.surfaces.split(","), modes=args.modes.split(","),
                   evalset_id=evalset_id, subset=args.subset if evalset_id else None, tiers=args.tiers)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
