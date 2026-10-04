"""What a coding agent can do with any environment dataset, through MCP (/mcp/d/<org>/<name>, app/mcp_bridge.py): the
same contract the pages use, as tools. Every environment gets these; an adapter adds its own (`mcp_tools`).

    describe       how it's read, how many tasks, its subsets and facets, how its tasks are graded
    list_tasks     a page of tasks (search, filter by facets, a subset)
    get_task       one task in full, as text: what the agent is asked, how it's graded, what it runs in, its files
    read_file      one of a task's files
    random_task    a random task's ref

Public datasets only (no visitor token here), and answers stay out: views and files are the adapters' own, which
withhold them.
"""

from __future__ import annotations

import json
from typing import Any

from . import registry

PAGE = 25
MAX_TEXT = 60_000

TOOLS: list[dict[str, Any]] = [
    {"name": "describe", "description": "What this environment is: its format, how many tasks, subsets, facets to filter by, and how its tasks are graded.",
     "inputSchema": {"type": "object", "properties": {"subset": {"type": "string", "description": "a subset id from `subsets` (optional)"}}}},
    {"name": "list_tasks", "description": "A page of tasks (25), each with the ref get_task takes. Search with `query`, filter with `filters` ({facet: [values]}).",
     "inputSchema": {"type": "object", "properties": {
         "subset": {"type": "string"}, "query": {"type": "string"},
         "filters": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
         "offset": {"type": "integer", "minimum": 0}}}},
    {"name": "get_task", "description": "One task in full, as text: what the agent is asked (word for word), how it's graded, what it runs in, its files and how it can be run. Answers are withheld.",
     "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}},
    {"name": "read_file", "description": "One of a task's files (paths from get_task). Files that hold answers are withheld.",
     "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}, "path": {"type": "string"}}, "required": ["ref", "path"]}},
    {"name": "random_task", "description": "A random task's ref.",
     "inputSchema": {"type": "object", "properties": {"subset": {"type": "string"}}}},
]


def tools(spec: str) -> list[dict[str, Any]]:
    env = registry.resolve("dataset", spec)
    return TOOLS + list(env.adapter.mcp_tools(env))


def call(spec: str, name: str, args: dict[str, Any]) -> Any:
    if name == "describe":
        s = registry.summary(spec, None, _str(args.get("subset")))
        if s.get("state") != "ready":
            return {"state": s.get("state"), "note": "Its index is being built; try again in a minute.", "progress": s.get("progress")}
        return {"id": spec, "format": s["env"]["framework"], "read_as": (s.get("how") or {}).get("name"), "about": (s.get("how") or {}).get("about"),
                "support": s.get("support"),
                "tasks": s.get("total"), "subsets": [x["id"] for x in s.get("subsets") or []], "subset": s.get("subset"),
                "facets": {f["key"]: f["label"] for f in s.get("facets") or []},
                "overview": blocks_text([b for sec in s.get("overview") or [] for b in [{"type": "markdown", "text": f"## {sec['title']}"}, *sec["blocks"]]])}
    if name == "list_tasks":
        return list_tasks(spec, _str(args.get("subset")), _str(args.get("query")) or "", args.get("filters") if isinstance(args.get("filters"), dict) else {},
                          max(0, int(args.get("offset") or 0)))
    if name == "get_task":
        return task_text(registry.task(spec, _need(args, "ref")))
    if name == "read_file":
        f = registry.file(spec, _need(args, "ref"), _need(args, "path"))
        if f.get("withheld"):
            return f"{f.get('path')}: withheld (it holds the answer)."
        if f.get("error") or f.get("binary"):
            return f"{f.get('path')}: {f.get('error') or 'a binary file'}"
        text = f.get("text") or ""
        return text[:MAX_TEXT] + (f"\n… (cut at {MAX_TEXT:,} characters)" if len(text) > MAX_TEXT or f.get("truncated") else "")
    if name == "random_task":
        return {"ref": registry.random(spec, _str(args.get("subset")))}
    env = registry.resolve("dataset", spec)
    return env.adapter.mcp_call(env, name, args)


def list_tasks(spec: str, subset: str | None, q: str, filters: dict[str, Any], offset: int) -> dict[str, Any]:
    filters = {str(k): [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in filters.items()}
    s = registry.summary(spec, None, subset)
    if s.get("state") != "ready":
        return {"state": s.get("state"), "note": "Its index is being built; try again in a minute."}
    if s.get("inline"):   # every task at once: search and filter here, as the page does
        cards = registry.tasks(spec, None, everything=True)["cards"]
        words = q.lower().split()
        hit = [c for c in cards if all(w in f"{c['title']} {c['ref']} {c.get('id') or ''} {c['brief']} {c['text']}".lower() for w in words)
               and all(set(c["facets"].get(k, [])) & set(v) for k, v in filters.items() if v)]
        page, total = hit[offset: offset + PAGE], len(hit)
    else:
        r = registry.tasks(spec, None, subset=subset, q=q, filters=filters, offset=offset)
        page, total = r["cards"], r["total"]
    return {"total": total, "offset": offset, "tasks": [{"ref": c["ref"], "title": c["title"], **({"id": c["id"]} if c.get("id") else {}),
                                                         **({"brief": c["brief"]} if c.get("brief") else {}), **({"tags": c["chips"]} if c.get("chips") else {})} for c in page]}


def task_text(v: dict[str, Any]) -> str:
    """A task view (registry.task) as Markdown an agent reads."""
    out = [f"# {v['title']}", f"ref: {v['ref']}" + (f" · id: {v['id']}" if v.get("id") and v["id"] != v["title"] else "")
           + f" · format: {v['env']['framework']}"]
    for sec in v.get("sections") or []:
        out.append(f"\n## {sec['title']}" + (f" ({sec['note']})" if sec.get("note") else ""))
        out.append(blocks_text(sec["blocks"]))
    if v.get("glance"):
        out.append("\n## At a glance\n" + "\n".join(f"- {a}: {b}" for a, b in v["glance"]))
    if v.get("withheld"):
        out.append("\nWithheld, as they hold the answer: " + ", ".join(v["withheld"]))
    opts = (v.get("run") or {}).get("options") or []
    support = v.get("support") or {}
    if support.get("framework"):
        f = support["framework"]
        out.append(f"\nLifecycle: {f['lifecycle']}\nReward: {f['reward']}")
    out.append("\n## Running it\n" + ("\n".join(f"- {o['label']}: " + ("runs here" if o["ok"] else f"can't run here ({o['why']})") for o in opts)
                                     or (v.get("run") or {}).get("note") or ""))
    for link in v.get("links") or []:
        href = link.get("href") or ""
        if link.get("rel") == "same" and href.lstrip("#").startswith("/t/"):
            org, name, *ref = href.lstrip("#")[3:].split("/")
            out.append(f"\nThe same task is also {org}/{name} · ref {'/'.join(ref)}.")
    text = "\n".join(x for x in out if x is not None)
    return text[:MAX_TEXT * 2]


def blocks_text(blocks: list[dict[str, Any]]) -> str:
    out = []
    for b in blocks:
        t = b.get("type")
        if t == "kv":
            out += [f"- {k}: {_cell(v)}" for k, v in b["rows"]]
        elif t == "shares":
            out += [f"- {a}: {n:,} of {b['of']:,}" for a, n in b["rows"]]
        elif t == "markdown":
            out.append(b["text"])
        elif t == "code":
            text = b["text"] if len(b["text"]) <= 20_000 else b["text"][:20_000] + "\n… (cut; read_file has it whole)"
            out.append(f"{b.get('label') or b.get('path') or ''}\n```\n{text}\n```")
        elif t == "note":
            out.append(b["text"])
        elif t in ("value", "messages"):
            out.append(_json(b["value"]))
        elif t == "steps":
            out += [f"{i}. {a} {c}".rstrip() for i, (a, c) in enumerate(b["items"], 1)]
        elif t == "disclose":
            out.append(f"{b['label']}:\n{blocks_text(b['blocks'])}")
        elif t == "stats":
            out.append(" · ".join(f"{a}: {c}" for a, c in b["rows"]))
        elif t == "links":
            out += [f"- {a}: {c}" for a, c in b["items"]]
        elif t == "custom":
            out.append(b.get("text") or "")
        elif t == "files":
            files = [f for f in b["tree"] if not f.get("dir")]
            out.append("Files (read_file reads them):\n" + "\n".join(f"- {f['path']}" + (" (withheld)" if f.get("withheld") else "") for f in files[:300])
                       + (f"\n… and {len(files) - 300} more" if len(files) > 300 else ""))
    return "\n".join(out)


def _cell(v: Any) -> str:
    if isinstance(v, dict) and v.get("type") == "shares":
        return "; ".join(f"{a} {n:,}" for a, n in v["rows"])
    return str(v)


def _json(v: Any) -> str:
    if isinstance(v, list) and v and all(isinstance(m, dict) and "role" in m for m in v):
        return "\n".join(f"[{m['role']}] {m.get('content') if isinstance(m.get('content'), str) else _json(m.get('content'))}" for m in v)
    text = v if isinstance(v, str) else json.dumps(_no_media(v), ensure_ascii=False, indent=1, default=str)
    return text if len(text) <= 20_000 else text[:20_000] + "\n… (cut)"


def _no_media(v: Any, depth: int = 0) -> Any:
    """Inline images and audio (data URLs, base64) as a short placeholder: text tools can't use them."""
    if depth > 12:
        return "…"
    if isinstance(v, dict):
        return {k: _no_media(x, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        return [_no_media(x, depth + 1) for x in v[:200]]
    if isinstance(v, str) and (v.startswith("data:") or len(v) > 5000 and " " not in v[:2000]):
        return f"[{len(v):,} characters of encoded data]"
    return v


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _need(args: dict[str, Any], key: str) -> str:
    v = args.get(key)
    if not isinstance(v, str):
        raise ValueError(f"{key} is required")
    return v
