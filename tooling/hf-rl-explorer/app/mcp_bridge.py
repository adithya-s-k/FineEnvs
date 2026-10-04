"""MCP servers, so coding agents (Claude Code, Codex, Cursor, VS Code, Gemini CLI, ...) can use an environment directly:

  /mcp/<org>/<name>     every OpenEnv Space: play it (below)
  /mcp/d/<org>/<name>   every environment dataset, any format: describe it, list, search and read its tasks and their
                        files (app/envs/tools.py; the same contract the pages use)

OpenEnv servers answer `tools/list` and `tools/call` on POST /mcp but not MCP's `initialize` handshake, and servers
that take actions through reset/step have no tools at all. This bridge speaks Streamable HTTP MCP (JSON replies, no
server-sent stream) and offers:

  * the Space's own MCP tools, called inside one OpenEnv MCP session per client session (so state carries over);
  * for reset/step servers: `reset`, `step` (its input is the server's action schema) and `state`, over one WebSocket
    session per client session;
  * for servers with the Task API: `list_splits`, `list_tasks`, `get_task` (answer-like fields left out).

Images an observation carries as base64 come back as MCP image content, so agents that can look at images do.
Sessions are capped per address and in total, and end after 10 idle minutes (see spaces_live).
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import catalog, spaces_live as live
from .http import client_ip, same_origin
from .version import app_version

router = APIRouter()
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_BODY = 512 * 1024
RATE = (240, 60)          # requests per window (seconds) per address

_clients: dict[str, dict[str, Any]] = {}    # Mcp-Session-Id -> {spec, play, owner, used}
_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def _owner(request: Request) -> str:
    return "mcp:" + client_ip(request)


def _limited(owner: str) -> bool:
    now = time.time()
    with _lock:
        hits = [t for t in _hits.get(owner, []) if now - t < RATE[1]]
        if len(hits) >= RATE[0]:
            _hits[owner] = hits
            return True
        hits.append(now)
        _hits[owner] = hits
        if len(_hits) > 5000:
            for k in list(_hits)[:1000]:
                _hits.pop(k, None)
        return len(hits) > RATE[0]


def _ok(rid: Any, result: Any, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result}, headers=headers)


def _err(rid: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}, status_code=status)


# ── tools ────────────────────────────────────────────────────────────────────
def _tools(info: dict) -> list[dict[str, Any]]:
    # no outputSchema: MCP clients then insist on structured content, which not every server sends
    tools = [{k: v for k, v in t.items() if k != "outputSchema"} for t in info.get("mcp") or []]
    names = {t["name"] for t in tools}
    schema = (info.get("schema") or {}).get("action") or {"type": "object"}
    if info.get("step_api") and not names & {"reset", "state"}:
        episode = [
            {"name": "reset", "description": "Start a new episode and return the first observation. `params` takes the server's "
             "own reset options, e.g. {\"seed\": 1} or {\"split\": \"test\", \"index\": 3} to pick a task from its Task API.",
             "inputSchema": {"type": "object", "properties": {"params": {"type": "object", "description": "Reset options, as the server takes them"}}}},
            {"name": "step", "description": "Take one action in the current episode; returns the observation, the reward and "
             "whether the episode is done. Call reset first.", "inputSchema": _action_input(schema)},
            {"name": "state", "description": "The current episode's state (its id and step count).", "inputSchema": {"type": "object", "properties": {}}},
        ]
        # a tool server takes its actions as tool calls: it gets reset and state, not step
        tools += [t for t in episode if not (names and t["name"] == "step")]
    # Keep published grading/seed routes alongside native tools, without naming collisions.
    if info.get("api"):
        taken = {t["name"] for t in tools}
        for r in info["api"]:
            name = _route_tool(r)
            if name in taken:
                continue
            taken.add(name)
            # Separate request locations: a body key named 'id' or 'query' must
            # not overwrite a path/query parameter with the same name.
            props, required = {}, []
            if r["params"]:
                props["path_params"] = {"type": "object", "properties": {p: {"type": "string"} for p in r["params"]},
                                        "required": r["params"], "additionalProperties": False}
                required.append("path_params")
            if (r.get("query_schema") or {}).get("properties"):
                props["query"] = r["query_schema"]
                if r["query_schema"].get("required"):
                    required.append("query")
            if r["method"] == "POST":
                props["body"] = r.get("schema") or {"type": "object"}
                if r.get("body_required"):
                    required.append("body")
            tools.append({"name": name, "description": f"{r['method']} {r['path']}" + (f": {r['summary']}" if r["summary"] else ""),
                          "inputSchema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}})
    if info.get("task_api"):
        extra = [
            {"name": "list_splits", "description": "The server's task splits and how many tasks each holds.", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "list_tasks", "description": "A page of tasks from a split (at most 50).", "inputSchema": {"type": "object", "properties": {
                "split": {"type": "string"}, "start": {"type": "integer", "default": 0}, "stop": {"type": "integer", "default": 20}}, "required": ["split"]}},
            {"name": "get_task", "description": "One task from a split, by index.", "inputSchema": {"type": "object", "properties": {
                "split": {"type": "string"}, "index": {"type": "integer"}}, "required": ["split", "index"]}},
        ]
        tools += [t for t in extra if t["name"] not in names]
    return tools


def _route_tool(r: dict) -> str:
    """A route's tool name: its path in snake case, with the method when it isn't POST (get_history, get_tasks_task_id)."""
    base = re.sub(r"[^A-Za-z0-9]+", "_", r["path"]).strip("_").lower() or "root"
    return (base if r["method"] == "POST" else f"get_{base}" if not base.startswith("get_") else base)[:64]


def _action_input(schema: dict) -> dict:
    """The step tool's input: the action schema itself when it's a plain object (its fields become the arguments)."""
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        out = {k: v for k, v in schema.items() if k in ("type", "properties", "required", "$defs", "description")}
        out["properties"] = {k: v for k, v in schema["properties"].items() if k != "metadata"}
        return out
    return {"type": "object", "properties": {"action": schema}, "required": ["action"]}


def _content(result: Any) -> dict[str, Any]:
    """Any reply as MCP tool content: passed through when it already is, otherwise JSON text plus its images."""
    if isinstance(result, dict) and isinstance(result.get("content"), list):   # already MCP (FastMCP writes snake_case)
        out: dict[str, Any] = {"content": [{k: v for k, v in c.items() if v is not None and k != "meta"} for c in result["content"] if isinstance(c, dict)]}
        sc = result.get("structuredContent", result.get("structured_content"))
        if isinstance(sc, dict):
            out["structuredContent"] = sc
        if result.get("isError", result.get("is_error")):
            out["isError"] = True
        return out
    images = live.image_parts(result)
    text = json.dumps(_strip_images(result), ensure_ascii=False, default=str)
    if len(text) > 200_000:
        text = text[:200_000] + "… (cut at 200,000 characters)"
    return {"content": [{"type": "text", "text": text}, *({"type": "image", "data": b64, "mimeType": mime} for mime, b64 in images)]}


def _strip_images(v: Any, depth: int = 0) -> Any:
    if depth > 6:
        return v
    if isinstance(v, dict):
        return {k: ("‹image, sent as image content›" if isinstance(x, str) and len(x) > 2000 and any(w in k.lower() for w in ("image", "img", "png", "jpeg", "frame", "screenshot"))
                    else _strip_images(x, depth + 1)) for k, x in v.items()}
    if isinstance(v, list):
        return [_strip_images(x, depth + 1) for x in v]
    return v


def _call(spec: str, info: dict, client: dict, name: str, args: dict) -> dict[str, Any]:
    own = {t["name"] for t in info.get("mcp") or []}
    if name in own:
        r = live.act(_play(spec, client), client["owner"], "call", {"name": name, "arguments": args})
        return _content(r["result"])
    if name in ("list_splits", "list_tasks", "get_task") and info.get("task_api"):
        if name == "list_splits":
            return _content(info["task_api"]["splits"])
        if name == "list_tasks":
            return _content(live.tasks(spec, str(args.get("split", "")), int(args.get("start", 0) or 0), int(args.get("stop", 20) or 20)))
        return _content(live.task(spec, str(args.get("split", "")), int(args.get("index", 0) or 0)))
    route = next((r for r in info.get("api") or [] if _route_tool(r) == name), None)
    if route is not None:
        params = args.get("path_params") if isinstance(args.get("path_params"), dict) else {}
        query = args.get("query") if isinstance(args.get("query"), dict) else {}
        body = args.get("body") if route["method"] == "POST" else None
        out = live.act(_play(spec, client), client["owner"], "http", {"method": route["method"], "path": route["path"], "params": params, "query": query, "body": body})["result"]
        payload = out.get("json", out.get("text"))
        result = _content(payload if payload is not None else {"status": out["status"]})
        if out["status"] >= 400:
            result["isError"] = True
        return result
    if name in ("reset", "step", "state") and info.get("step_api") and name not in own:
        if name == "reset":
            data = args.get("params") if isinstance(args.get("params"), dict) else {}
        elif name == "step":
            data = args.get("action") if isinstance(args.get("action"), dict) and set(args) == {"action"} else args
        else:
            data = {}
        r = live.act(_play(spec, client), client["owner"], name, data)
        return _content(r["result"])
    raise live.SpaceError(f"no tool named {name!r}", 404)


def _play(spec: str, client: dict) -> str:
    """The playground session behind a client session, started on first use and restarted if it timed out."""
    if client.get("play"):
        s = live._sessions.get(client["play"])
        if s is not None:
            return client["play"]
    client["play"] = live.start(spec, client["owner"])["session"]
    return client["play"]


# ── the endpoint ─────────────────────────────────────────────────────────────
async def _message(request: Request) -> dict | Response:
    """The JSON-RPC request, or the reply that refuses it (cross-origin, too many, too large, malformed, a notification)."""
    if not same_origin(request):     # agents send no Origin; a web page elsewhere can't drive this from a browser
        return _err(None, -32600, "cross-origin requests are not accepted", 403)
    if _limited(_owner(request)):
        return _err(None, -32000, "too many requests: slow down", 429)
    body = await request.body()
    if len(body) > MAX_BODY:
        return _err(None, -32600, "request too large", 413)
    try:
        msg = json.loads(body)
    except ValueError:
        return _err(None, -32700, "parse error", 400)
    if isinstance(msg, list):
        return _err(None, -32600, "batches are not supported", 400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request", 400)
    if "id" not in msg:                                   # a notification (initialized, cancelled, ...): nothing to say
        return Response(status_code=202)
    return msg


@router.post("/mcp/{org}/{name}")
async def mcp(org: str, name: str, request: Request):
    import anyio

    msg = await _message(request)
    if isinstance(msg, Response):
        return msg
    owner = _owner(request)
    rid, method, params = msg.get("id"), msg["method"], msg.get("params") if isinstance(msg.get("params"), dict) else {}
    try:
        spec = catalog.check_spec(f"{org}/{name}")
    except ValueError:
        return _err(rid, -32602, "not a Space id", 404)

    sid = request.headers.get("mcp-session-id")
    if method == "initialize":
        try:
            info = await anyio.to_thread.run_sync(live.probe, spec)
        except PermissionError:
            return _err(rid, -32001, "this Space is private", 404)
        except Exception as e:  # noqa: BLE001
            return _err(rid, -32001, f"couldn't look up this Space: {type(e).__name__}", 404)
        want = params.get("protocolVersion")
        sid = secrets.token_urlsafe(24)
        with _lock:
            _clients[sid] = {"spec": spec, "play": None, "owner": owner, "used": time.time()}
            for k in [k for k, c in _clients.items() if time.time() - c["used"] > live.IDLE * 3]:
                _clients.pop(k, None)
        state = "running" if info.get("running") else f"{info.get('stage', 'unknown').lower()} (it may take a minute to wake: call any tool to retry)"
        return _ok(rid, {
            "protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": f"{spec} (OpenEnv)", "title": spec, "version": app_version()},
            "instructions": f"The OpenEnv environment {spec} on Hugging Face Spaces, through HF RL Explorer. The Space is {state}. "
                            + ("Call reset, then step with actions; each step returns the observation, the reward and done. "
                               if info.get("step_api") and not info.get("mcp") else "")
                            + ("Its Task API lists the tasks it can serve (list_splits, list_tasks, get_task)." if info.get("task_api") else ""),
        }, headers={"Mcp-Session-Id": sid})
    if method == "ping":
        return _ok(rid, {})

    client = _clients.get(sid or "")
    if client is None or client["spec"] != spec:          # no session: serve this one request on a fresh one
        client = {"spec": spec, "play": None, "owner": owner, "used": time.time()}
    client["used"] = time.time()
    try:
        info = await anyio.to_thread.run_sync(live.probe, spec)
        if not info.get("running"):
            await anyio.to_thread.run_sync(live.wake, spec)
            return _err(rid, -32002, f"the Space is {info.get('stage', 'not running').lower()}; it has been asked to wake. Try again in a minute.")
        if method == "tools/list":
            return _ok(rid, {"tools": _tools(info)})
        if method == "tools/call":
            tool = params.get("name")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if not isinstance(tool, str):
                return _err(rid, -32602, "missing tool name")
            try:
                result = await anyio.to_thread.run_sync(_call, spec, info, client, tool, args)
            except live.SpaceError as e:                  # a failed call is the tool's result, so the agent sees why
                return _ok(rid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
            return _ok(rid, result)
        if method in ("resources/list", "prompts/list"):
            return _ok(rid, {"resources": []} if method.startswith("resources") else {"prompts": []})
        return _err(rid, -32601, f"method not found: {method}")
    except live.SpaceError as e:
        return _err(rid, -32000, str(e))
    except PermissionError:
        return _err(rid, -32001, "this Space is private", 404)


@router.get("/mcp/{org}/{name}")
def mcp_stream(org: str, name: str):
    return Response(status_code=405, headers={"Allow": "POST, DELETE"})


@router.delete("/mcp/{org}/{name}")
def mcp_end(org: str, name: str, request: Request):
    sid = request.headers.get("mcp-session-id") or ""
    with _lock:
        client = _clients.pop(sid, None)
    if client and client.get("play"):
        live.end(client["play"], client["owner"])
    return Response(status_code=204)


# ── environment datasets ─────────────────────────────────────────────────────
@router.post("/mcp/d/{org}/{name}")
async def mcp_dataset(org: str, name: str, request: Request):
    import anyio

    from .envs import registry, tools as env_tools

    msg = await _message(request)
    if isinstance(msg, Response):
        return msg
    rid, method, params = msg.get("id"), msg["method"], msg.get("params") if isinstance(msg.get("params"), dict) else {}
    try:
        spec = catalog.check_spec(f"{org}/{name}")
    except ValueError:
        return _err(rid, -32602, "not a dataset id", 404)
    try:
        if method == "initialize":
            s = await anyio.to_thread.run_sync(registry.summary, spec)
            want = params.get("protocolVersion")
            fw = s["env"]["framework"]
            return _ok(rid, {
                "protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": f"{spec} ({fw})", "title": spec, "version": app_version()},
                "instructions": f"The RL environment dataset {spec} on Hugging Face ({fw}), through HF RL Explorer. "
                                "Call describe first, then list_tasks to find tasks, get_task to read one in full (what the agent is asked, "
                                "how it's graded, what it runs in), and read_file for its files. Answers are withheld.",
            }, headers={"Mcp-Session-Id": secrets.token_urlsafe(24)})
        if method == "ping":
            return _ok(rid, {})
        if method == "tools/list":
            return _ok(rid, {"tools": await anyio.to_thread.run_sync(env_tools.tools, spec)})
        if method == "tools/call":
            tool = params.get("name")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if not isinstance(tool, str):
                return _err(rid, -32602, "missing tool name")
            try:
                result = await anyio.to_thread.run_sync(env_tools.call, spec, tool, args)
            except (LookupError, ValueError) as e:        # a bad ref or path is the tool's result, so the agent sees why
                return _ok(rid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
            if isinstance(result, str):
                return _ok(rid, {"content": [{"type": "text", "text": result}]})
            return _ok(rid, _content(result))
        if method in ("resources/list", "prompts/list"):
            return _ok(rid, {"resources": []} if method.startswith("resources") else {"prompts": []})
        return _err(rid, -32601, f"method not found: {method}")
    except PermissionError:
        return _err(rid, -32001, "no such dataset, or it is private", 404)
    except Exception as e:  # noqa: BLE001 - the Hub or the viewer failing: say so, briefly
        from huggingface_hub.errors import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError):
            return _err(rid, -32001, "no such dataset, or it is private", 404)
        return _err(rid, -32000, f"couldn't read this dataset: {type(e).__name__}: {str(e)[:200]}")


@router.get("/mcp/d/{org}/{name}")
def mcp_dataset_stream(org: str, name: str):
    return Response(status_code=405, headers={"Allow": "POST, DELETE"})


@router.delete("/mcp/d/{org}/{name}")
def mcp_dataset_end(org: str, name: str):
    return Response(status_code=204)
