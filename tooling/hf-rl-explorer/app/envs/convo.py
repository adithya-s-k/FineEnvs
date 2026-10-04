"""Conversations and tool lists, in whichever shape a row carries them, as one shape the page draws (the `transcript`
and `tools` views of web/js/renderers/rl.js) and agents read (Markdown).

Conversations come as OpenAI Responses API items (NeMo Gym's `responses_create_params.input`: `message`, `reasoning`,
`function_call`, `function_call_output`), or as chat messages (verl's and Verifiers' `prompt`: `role`/`content`,
assistant `tool_calls`, `tool` replies). `turns()` reads both into a list of turns, each a message, a reasoning summary,
or a tool call with its output paired to it by call id. `tools()` reads tool definitions in the Responses API's flat
form, the chat API's nested `function` form, or MCP's `input_schema` form into signatures: name, description, and each
parameter's type, whether it's required, and its allowed values.

Nothing is cut short except what would make a page unreasonable (a text of over MAX_TEXT characters, more than
MAX_TURNS turns), and that is said where it happens.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import is_messages, parse_json

MAX_TURNS = 600
MAX_TEXT = 200_000
MAX_TOOLS = 300
_IMG = re.compile(r"^(https://\S+|data:image/(png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+)$")


def _cap(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"\n… ({len(text) - MAX_TEXT:,} more characters not shown)"


def _dump(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, indent=2, default=str)


def content_text(content: Any) -> tuple[str, list[str]]:
    """A message's content as text, and the images it carries (https or data: URLs): a string, or a list of parts
    (Responses API `input_text`/`output_text`/`input_image`, chat `text`/`image_url`)."""
    content = parse_json(content)
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content), []
    out: list[str] = []
    images: list[str] = []
    for p in content[:400]:
        if isinstance(p, str):
            out.append(p)
            continue
        if not isinstance(p, dict):
            out.append(str(p))
            continue
        t = str(p.get("type") or "")
        if isinstance(p.get("text"), str):
            out.append(p["text"])
        elif isinstance(p.get("refusal"), str):
            out.append(p["refusal"])
        elif "image" in t:
            url = p.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if isinstance(url, str) and _IMG.match(url) and len(url) < 3 * 2**20:
                images.append(url)
            else:
                out.append("[an image]")
        elif "audio" in t:
            out.append("[audio]")
        elif "file" in t:
            out.append(f"[a file{': ' + str(p.get('filename')) if p.get('filename') else ''}]")
        else:
            out.append(_dump(p))
    return "\n\n".join(x for x in out if x), images


def _args(a: Any) -> str:
    """Tool-call arguments, pretty-printed when they are JSON."""
    a = parse_json(a)
    if isinstance(a, (dict, list)):
        return _dump(a)
    return "" if a is None else str(a)


def _role(r: Any) -> str:
    r = str(r or "").lower()
    return {"human": "user", "gpt": "assistant", "model": "assistant", "bot": "assistant", "function": "tool", "ipython": "tool",
            "environment": "tool", "observation": "tool"}.get(r, r or "user")


def turns(items: Any) -> list[dict[str, Any]]:
    """The conversation as turns: {kind: message|reasoning|call|output|other, role, text, name?, args?, call_id?,
    output?, images?}. A tool call's output is paired to it (by call id, else the next unanswered call)."""
    items = parse_json(items)
    if isinstance(items, str):
        return [{"kind": "message", "role": "user", "text": _cap(items)}]
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items[:MAX_TURNS]:
        it = parse_json(it)
        if isinstance(it, str):
            out.append({"kind": "message", "role": "user", "text": _cap(it)})
            continue
        if not isinstance(it, dict):
            continue
        t = str(it.get("type") or "")
        if t == "reasoning":
            parts = it.get("summary") or it.get("content") or []
            text = "\n\n".join(p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)) if isinstance(parts, list) else str(parts)
            out.append({"kind": "reasoning", "role": "assistant", "text": _cap(text) if text else "(encrypted)" if it.get("encrypted_content") else ""})
        elif t in ("function_call", "custom_tool_call", "mcp_call", "tool_call") or (t.endswith("_call") and "role" not in it):
            name = it.get("name") or t.removesuffix("_call")
            args = it.get("arguments") if "arguments" in it else it.get("input") if "input" in it else it.get("action")
            out.append({"kind": "call", "role": "assistant", "name": str(name), "args": _cap(_args(args)), "call_id": it.get("call_id") or it.get("id")})
        elif t.endswith("_call_output") or t == "tool_result":
            text, images = content_text(it.get("output") if "output" in it else it.get("content"))
            out.append({"kind": "output", "role": "tool", "text": _cap(text), "call_id": it.get("call_id") or it.get("tool_call_id"), "images": images})
        elif "role" in it or t == "message":
            role = _role(it.get("role") or it.get("from"))
            think = it.get("reasoning_content") or it.get("reasoning") or it.get("thinking")
            if isinstance(think, str) and think.strip():
                out.append({"kind": "reasoning", "role": "assistant", "text": _cap(think)})
            text, images = content_text(it.get("content") if "content" in it else it.get("value"))
            if role == "tool":
                out.append({"kind": "output", "role": "tool", "text": _cap(text), "call_id": it.get("tool_call_id") or it.get("call_id"),
                            "name": it.get("name"), "images": images})
                continue
            if text or images or not it.get("tool_calls"):
                out.append({"kind": "message", "role": role, "text": _cap(text), "images": images, **({"name": str(it["name"])} if it.get("name") else {})})
            for call in it.get("tool_calls") or []:
                if isinstance(call, dict):
                    fn = call.get("function") if isinstance(call.get("function"), dict) else call
                    out.append({"kind": "call", "role": "assistant", "name": str(fn.get("name") or "tool"), "args": _cap(_args(fn.get("arguments"))),
                                "call_id": call.get("id") or call.get("call_id")})
        else:
            out.append({"kind": "other", "role": "other", "text": _cap(_dump(it)), "name": t or None})
    _pair(out)
    if isinstance(items, list) and len(items) > MAX_TURNS:
        out.append({"kind": "other", "role": "other", "text": f"… and {len(items) - MAX_TURNS:,} more items not shown"})
    return out


def _pair(ts: list[dict[str, Any]]) -> None:
    """Each tool output joins its call: by call id when both have one, else the earliest call still unanswered."""
    calls = [t for t in ts if t["kind"] == "call"]
    by_id = {t["call_id"]: t for t in calls if t.get("call_id")}
    drop = []
    for i, t in enumerate(ts):
        if t["kind"] != "output":
            continue
        call = by_id.get(t.get("call_id")) if t.get("call_id") else None
        if call is None or "output" in call:
            call = next((c for c in calls if "output" not in c and ts.index(c) < i and (not t.get("name") or c["name"] == t["name"])), None)
        if call is not None and "output" not in call:
            call["output"] = t["text"]
            if t.get("images"):
                call["images"] = t["images"]
            drop.append(i)
    for i in reversed(drop):
        ts.pop(i)


_SKIP_LINE = re.compile(r"^(<[^>]{1,80}>|/[\w./-]+|\[[A-Z ]{2,20}\]|-{3,}|={3,}|[\w ]{1,24}:)$")
_TITLE_LINE = re.compile(r"^\s*(?:\*\*|#+\s*)?(?:title|issue title)\s*:?\s*(?:\*\*)?\s*:?\s*(.{6,})$", re.IGNORECASE | re.MULTILINE)


def headline(text: str, n: int = 180) -> str:
    """A title from a text: its first line that says something (not a bare tag, path or rule), whole when it's short
    enough, else cut at the end of a sentence when one comes soon enough, else at a word, with an ellipsis."""
    line = ""
    for ln in str(text or "").splitlines():
        ln = re.sub(r"^[#>*\-\s`]+", "", re.sub(r"<(image|video|audio)>\s*", "", ln)).strip()
        if len(ln) > 3 and not _SKIP_LINE.match(ln):
            line = ln
            break
    if len(line) <= n:
        return line
    cut = max((m.end() for m in re.finditer(r"[.!?。？！](\s|$)", line[:n + 1])), default=0)
    if cut >= n * 0.4:
        return line[:cut].strip()
    return line[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def task_title(text: str, template: str | None = None) -> str:
    """The title of a task's text: an issue's own "Title:" line when it has one; else its first line that says
    something, past a template's preamble (the paragraphs before the template's `{question}`/`{problem}` slot)."""
    text = str(text or "")
    if isinstance(template, str) and re.search(r"\{\w+\}", template):
        pre = re.split(r"\{\w+\}", template, maxsplit=1)[0]
        k = len([p for p in re.split(r"\n\s*\n", pre) if p.strip()])
        paras = re.split(r"(\n\s*\n)", text)
        if k and len(paras) > 2 * k:
            text = "".join(paras[2 * k:])
    m = _TITLE_LINE.search(text[:6000])
    if m and not m.group(1).strip().startswith("{"):
        return headline(m.group(1).strip().strip("*").strip())
    return headline(text)


def after_title(text: str, title: str) -> str:
    """The rest of a task's text past its title, for a card's second line ("…" first when the title was cut short)."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    head = title.rstrip("…")
    at = flat.find(head)
    if at < 0:
        return ""
    rest = flat[at + len(head):].strip(" .:")
    return ("…" + rest if title.endswith("…") else rest) if len(rest) > 12 else ""


def first_user(ts: list[dict[str, Any]]) -> str:
    return next((t["text"] for t in ts if t["kind"] == "message" and t["role"] == "user" and t["text"].strip()), "")


def last_user(ts: list[dict[str, Any]]) -> str:
    return next((t["text"] for t in reversed(ts) if t["kind"] == "message" and t["role"] == "user" and t["text"].strip()), "")


def is_conversation(v: Any) -> bool:
    """A list of chat messages or Responses API items."""
    v = parse_json(v)
    return is_messages(v) or (isinstance(v, list) and bool(v) and all(isinstance(x, dict) and ("role" in x or "type" in x) for x in v[:20]))


def transcript_text(ts: list[dict[str, Any]], limit: int = 40_000) -> str:
    """The turns as Markdown, for agents (MCP) and pages that can't draw them."""
    out = []
    for t in ts:
        k = t["kind"]
        if k == "message":
            out.append(f"**{t['role'].title()}:** {t['text']}")
        elif k == "reasoning":
            out.append(f"**Assistant, reasoning:** {t['text']}")
        elif k == "call":
            out.append(f"**Assistant calls `{t['name']}`:**\n```json\n{t['args']}\n```" + (f"\n**Tool output:** {t['output']}" if "output" in t else ""))
        elif k == "output":
            out.append(f"**Tool output:** {t['text']}")
        else:
            out.append(t["text"])
    text = "\n\n".join(out)
    return text if len(text) <= limit else text[:limit] + "\n… (cut)"


# ── tools ────────────────────────────────────────────────────────────────────
def _type(p: Any) -> str:
    if not isinstance(p, dict):
        return ""
    t = p.get("type")
    if isinstance(t, list):
        return " | ".join(str(x) for x in t)
    if t == "array":
        inner = _type(p.get("items") or {})
        return f"array of {inner}" if inner else "array"
    if t:
        return str(t)
    for k in ("anyOf", "oneOf", "allOf"):
        if isinstance(p.get(k), list):
            return " | ".join(x for x in (_type(y) for y in p[k]) if x) or k
    if isinstance(p.get("$ref"), str):
        return p["$ref"].rsplit("/", 1)[-1]
    if "enum" in p:
        return "enum"
    if "const" in p:
        return f"const {json.dumps(p['const'])}"
    return ""


def _params(schema: Any, depth: int = 0) -> list[dict[str, Any]]:
    schema = parse_json(schema)
    if not isinstance(schema, dict):
        return []
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema = schema["items"]
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    req = set(schema.get("required") or []) if isinstance(schema.get("required"), list) else set()
    out = []
    for name, p in list(props.items())[:80]:
        p = p if isinstance(p, dict) else {}
        items = p.get("items") if isinstance(p.get("items"), dict) else {}
        enum = p.get("enum") or items.get("enum")
        row: dict[str, Any] = {"name": str(name), "type": _type(p), "required": name in req, "description": str(p.get("description") or "")[:2000]}
        if isinstance(enum, list) and enum:
            row["enum"] = [json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in enum[:40]]
        if "default" in p:
            row["default"] = json.dumps(p["default"], ensure_ascii=False)[:200]
        if depth < 2:
            kids = _params(p, depth + 1)
            if kids:
                row["params"] = kids
        out.append(row)
    return out


def tools(v: Any) -> list[dict[str, Any]]:
    """Tool definitions as signatures: [{name, kind, description, params: [{name, type, required, description, enum?,
    default?, params?}]}]."""
    v = parse_json(v)
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for t in v[:MAX_TOOLS]:
        t = parse_json(t)
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name") or t.get("name") or t.get("type")
        if not name:
            continue
        schema = fn.get("parameters") if "parameters" in fn else fn.get("input_schema") or fn.get("inputSchema") or fn.get("args_schema")
        out.append({"name": str(name), "kind": str(t.get("type") or "function"), "description": str(fn.get("description") or "")[:6000],
                    "params": _params(schema), "built_in": not fn.get("name") and not t.get("name")})
    return out


def tools_text(ts: list[dict[str, Any]]) -> str:
    out = []
    for t in ts:
        sig = ", ".join(p["name"] + ("" if p["required"] else "?") for p in t["params"])
        out.append(f"- `{t['name']}({sig})`" + (f": {t['description']}" if t["description"] else ""))
        for p in t["params"]:
            out.append(f"  - `{p['name']}` {p['type']}{', required' if p['required'] else ''}" + (f", one of {', '.join(p['enum'])}" if p.get("enum") else "")
                       + (f": {p['description']}" if p["description"] else ""))
    return "\n".join(out)


# ── a row's other fields ─────────────────────────────────────────────────────
def _options(v: Any) -> str | None:
    """MCQA options as NeMo Gym and others store them, a list of one-key dicts ({"A": text}), as "A: text" lines."""
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and all(len([k for k, y in x.items() if y is not None]) <= 1 for x in v):
        lines = [f"{k}: {y}" for x in v for k, y in x.items() if y is not None]
        return "\n".join(lines) if lines else None
    if isinstance(v, dict) and v and all(isinstance(k, str) and len(k) <= 3 for k in v) and all(isinstance(y, (str, type(None))) for y in v.values()):
        return "\n".join(f"{k}: {y}" for k, y in v.items() if y is not None)
    return None


def field_rows(d: dict[str, Any], limit: int = 60) -> list[dict[str, Any]]:
    """Fields as label/value rows for the `fields` view: short scalars inline, everything else as text or indented
    JSON (long ones fold on the page). Empty values are left out."""
    out = []
    for k, v in list(d.items())[:limit]:
        v = parse_json(v)
        if v in (None, "", [], {}):
            continue
        opts = _options(v)
        if opts is not None:
            out.append({"key": str(k), "text": _cap(opts), "code": False})
        elif isinstance(v, str):
            out.append({"key": str(k), "text": _cap(v), "code": False})
        elif isinstance(v, (bool, int, float)):
            out.append({"key": str(k), "text": json.dumps(v), "code": True})
        else:
            out.append({"key": str(k), "text": _cap(_dump(v)), "code": True})
    return out


def fields_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {r['key']}: " + (f"\n```\n{r['text']}\n```" if r["code"] and "\n" in r["text"] else r["text"]) for r in rows)
