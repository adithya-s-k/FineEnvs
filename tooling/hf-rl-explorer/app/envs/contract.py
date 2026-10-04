"""The contract every environment format implements, so the explorer can show, search, run and connect to any of them
the same way.

An **environment** is a source on the Hub (a dataset or a Space) read by an **adapter** that understands its format.
Every adapter answers the same five questions, which are the five parts of every page:

    tasks        what it asks: a list to search and filter, and each task in full (`summary`, `tasks`, `task`)
    environment  what a task runs in (a section of the task view)
    harness      how a task can be run and graded: the run options it offers, each by a runner (`run_options`,
                 `materialize` for the Harbor runner; the agent that plays it is picked at run time)
    rollouts     runs of a task, wherever they were started (`aliases` links a task to the same task elsewhere)
    mcp          how an agent uses it (`mcp_tools`, `mcp_call`)

A task is named by a **ref**, a string the adapter defines and parses ("tasks/fix-bug", "code/train/6"), so every task
has one URL: `/t/<org>/<name>/<ref>`. Nothing here is specific to a format: a new one subclasses `Adapter` (or
`RowsAdapter`, for tasks that are dataset rows) and registers itself; see CUSTOM_ENVS.md.

Everything returned is plain JSON (dicts, lists, strings, numbers): adapters build it with the helpers below.

**Answers never leave an adapter.** Anything that could hold a task's answer (a reference solution, a gold label, a
rubric's anchors) is left out of everything an adapter returns, and named in the view's `withheld` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── the environment ──────────────────────────────────────────────────────────


@dataclass
class Env:
    """A resolved environment: where it lives, what the Hub says about it, and who is asking."""
    kind: str                          # "dataset" or "space"
    id: str                            # org/name
    meta: dict[str, Any]               # the Hub's record (sha, tags, card fields, restricted, ...)
    adapter: "Adapter"
    token: str | None = None           # the visitor's token; adapters use it only for restricted sources

    @property
    def key(self) -> str:
        return self.id if self.kind == "dataset" else f"space:{self.id}"

    @property
    def read_token(self) -> str | None:
        """The token to read the source with: the visitor's for a private or gated one, else none (shared caches)."""
        return self.token if self.meta.get("restricted") else None


# ── what adapters return ─────────────────────────────────────────────────────
# Built as plain dicts so they go straight to the page. The page knows each shape; an adapter fills in what it has.

def card(ref: str, title: str, *, id: str | None = None, brief: str = "", lead: str | None = None, sub: list[str] | None = None,
         chips: list[str] | None = None, facets: dict[str, list[str]] | None = None, text: str = "", icon: str | None = None,
         color: str | None = None, stats: list[tuple[Any, str]] | None = None) -> dict[str, Any]:
    """One task in a list. `lead`/`sub` head the card (a group, a difficulty), `chips` close it, `stats` are counts
    ([(7, "tools")]), `facets` are the values it is filtered by ({facet key: [values]}), `text` is what search matches
    besides the title. `icon` and `color` (a palette name: code, webdev, cyber, music, general, hub, ...) mark its kind."""
    return {"ref": ref, "title": title[:240], "id": id, "brief": brief[:400], "lead": lead, "sub": sub or [], "chips": chips or [],
            "facets": facets or {}, "text": text[:2000], "icon": icon, "color": color, "stats": [[a, b] for a, b in stats or []]}


def facet(key: str, label: str, order: str = "count", values: list[list[Any]] | None = None, sampled: bool = False,
          scope: list[str] | None = None) -> dict[str, Any]:
    """A facet to filter by. `order` "level" sorts easy→hard; `values` [[value, count]] when the adapter counts them
    itself (a paged list), else the page counts its cards (an inline list). `scope` shows it only with one of these
    tiles picked ("*": only when none is), for facets that mean something in one part of the environment."""
    return {"key": key, "label": label, "order": order, "values": values, "sampled": sampled, "scope": scope}


def tiles(key: str, items: list[dict[str, Any]], *, all_label: str = "All tasks", all_note: str = "") -> dict[str, Any]:
    """Big buttons above an inline list, one per value of facet `key` (the parts of an environment: its domains):
    items [{value, note, icon, color}]."""
    return {"key": key, "items": items, "all_label": all_label, "all_note": all_note}


def treemap(by_tile: dict[str, str] | None = None, key: str | None = None, top: dict[str, int] | None = None) -> dict[str, Any]:
    """The map of an inline list: tasks by facet `key`, or, with tiles, by each tile's own facet (`by_tile`: tile
    value -> facet key; with no tile picked, every tile's blocks side by side, its `top` values each)."""
    return {"key": key, "by_tile": by_tile or {}, "top": top or {}}


# blocks: what a section is made of. The page draws each kind.
def kv(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    """Label/value rows. Values are text with inline Markdown (`code`, **bold**, links), or numbers."""
    return {"type": "kv", "rows": [[k, v] for k, v in rows if v not in (None, "", [])]}


def shares(rows: list[tuple[str, int]], of: int, note: str = "") -> dict[str, Any]:
    """Parts of a whole, each with a bar: [(label, count)] out of `of` (a kv value, or a block)."""
    return {"type": "shares", "of": of, "rows": [[a, b] for a, b in rows], "note": note}


def markdown(text: str) -> dict[str, Any]:
    return {"type": "markdown", "text": text}


def code(text: str, path: str, label: str | None = None) -> dict[str, Any]:
    """A file shown inline, highlighted by its name; `label` heads it."""
    return {"type": "code", "text": text, "path": path, "label": label}


def note(text: str, icon: str = "info") -> dict[str, Any]:
    """A line of fine print (inline Markdown) with an icon."""
    return {"type": "note", "text": text, "icon": icon}


def value(v: Any) -> dict[str, Any]:
    """Any JSON, drawn richly: images, audio, chats, tables, nested rows that fold."""
    return {"type": "value", "value": v}


def messages(v: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "messages", "value": v}


def files(tree: list[dict[str, Any]], truncated: bool = False, hub: str | None = None, raw: bool = True) -> dict[str, Any]:
    """The file viewer: [{path, size, withheld}] and folders listed when opened ({path: "x/", dir, lazy}). The adapter
    serves contents through `file` and `folder`. `hub` is the folder's URL on the Hub at a revision
    (https://huggingface.co/datasets/<id>/blob/<sha>/<folder>/), for "open on the Hub" links; `raw` false when its
    files can't be fetched from there without a token (a private source)."""
    return {"type": "files", "tree": tree, "truncated": truncated, "hub": hub, "raw": raw and bool(hub)}


def hub_folder(repo: str, sha: str, folder: str, kind: str = "datasets") -> str:
    """A folder's blob URL on the Hub, for `files(hub=...)`."""
    from urllib.parse import quote

    return f"https://huggingface.co/{kind}/{repo}/blob/{quote(sha, safe='')}/" + (quote(folder.strip('/')) + "/" if folder.strip("/") else "")


def steps(items: list[tuple[str, str]]) -> dict[str, Any]:
    """Numbered steps (a multi-step task)."""
    return {"type": "steps", "items": [[a, b] for a, b in items]}


def disclose(label: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Blocks folded under a line, opened on click."""
    return {"type": "disclose", "label": label, "blocks": blocks}


def stats(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    """A row of figures, each a label over a value (inline Markdown): a music brief's tempo and meter, a crash's target."""
    return {"type": "stats", "rows": [[k, v] for k, v in rows if v not in (None, "")]}


def links(items: list[tuple[str, str]]) -> dict[str, Any]:
    """Buttons to other pages: [(label, https URL)]."""
    return {"type": "links", "items": [[a, b] for a, b in items]}


def custom(renderer: str, kind: str, data: Any, text: str = "") -> dict[str, Any]:
    """A view no block covers, drawn by the page's renderer module web/js/renderers/<renderer>.js (its export
    `kind`). `text` is the same in Markdown, for agents (MCP) and anything else that can't draw it. Prefer blocks:
    a renderer is code to write and keep; it fetches anything more it needs through the adapter's `data`."""
    return {"type": "custom", "renderer": renderer, "kind": kind, "data": data, "text": text}


def section(id: str, title: str, blocks: list[dict[str, Any]], *, icon: str = "list", note: str = "", collapsed: bool = False) -> dict[str, Any]:
    """A panel of the page. `collapsed` folds it until opened (a long overview that isn't the point of the page)."""
    return {"id": id, "title": title, "icon": icon, "note": note, "collapsed": collapsed, "blocks": [b for b in blocks if b]}


def run_option(runner: str, label: str, *, ok: bool, why: str = "", default: bool = False, notes: list[str] | None = None,
               warnings: list[str] | None = None, fields: list[dict[str, Any]] | None = None, href: str | None = None,
               harnesses: list[str] | None = None, about: str = "", endpoint: bool = True, sandbox: bool = True,
               estimate: dict[str, Any] | None = None) -> dict[str, Any]:
    """One way to run a task, by a runner ("harbor": OpenEnv's Harbor runner on an HF Sandbox; "mimo": the MiMo
    release's own harness). `fields` are the inputs it takes beyond agent and model (see `field`); `harnesses` limits
    the agents it can drive (None: all); `endpoint` whether the visitor may bring their own OpenAI-compatible endpoint;
    `sandbox` false when no sandbox is billed (one model call); `estimate` what a typical run uses
    ({tokens_in, tokens_out, minutes, flavor}), for the cost shown; `href` sends the visitor somewhere instead (a
    Space's playground); `about` says what a run does (inline Markdown)."""
    return {"runner": runner, "label": label, "ok": ok, "why": why, "default": default, "notes": notes or [], "warnings": warnings or [],
            "fields": fields or [], "href": href, "harnesses": harnesses, "about": about, "endpoint": endpoint, "sandbox": sandbox,
            "estimate": estimate}


def field(key: str, label: str, type: str = "text", *, default: Any = None, options: list[Any] | None = None, help: str = "",
          advanced: bool = False, placeholder: str = "", min: float | None = None, max: float | None = None, step: float | None = None,
          pool: str | None = None, required: bool = False) -> dict[str, Any]:
    """An input a run option takes: `type` text, number, bool, select (with `options`, values or {value, label}) or
    model (a model from the explorer's catalog: `pool` "vision_judges" or "text_judges"). `advanced` puts it under
    Advanced settings."""
    return {"key": key, "label": label, "type": type, "default": default, "options": options, "help": help, "advanced": advanced,
            "placeholder": placeholder, "min": min, "max": max, "step": step, "pool": pool, "required": required}


def link(label: str, href: str, note: str = "", rel: str = "related") -> dict[str, Any]:
    """Another page about this task: `rel` "same" for the same task elsewhere (its rollouts are this task's too)."""
    return {"label": label, "href": href, "note": note, "rel": rel}


# ── the adapter ──────────────────────────────────────────────────────────────


class Adapter:
    """One environment format. Subclass it, set the class attributes, implement what applies, and register it in
    `app/envs/registry.py`. Methods that don't apply keep the defaults, which say so."""

    id = "base"                 # unique, short: used in logs, URLs and run records
    name = "Environment"        # shown in "How it's read"
    framework = "Environment"   # the badge on cards and page headers
    about = ""                  # one sentence: what a task is here, and how it's graded
    kinds: tuple[str, ...] = ("dataset",)

    # which sources it reads
    def detect(self, kind: str, meta: dict[str, Any]) -> float:
        """0 to 1: how sure this adapter is that it reads this source. The highest wins. Cheap checks only: `meta` is
        the Hub's record (tags, card); adapters that need to look at the data do it in `confirm`."""
        return 0.0

    def confirm(self, env: Env) -> bool:
        """A closer look, once, when `detect` was the best but unsure (< 1): False hands the source to the next."""
        return True

    def settled(self, env: Env) -> bool:
        """Whether choosing this adapter is final (kept a minute). False while it is still finding out (an index being
        built), so the choice is made again on the next request; `summary` returning {"state": "handoff"} gives it up."""
        return True

    # tasks
    def summary(self, env: Env, subset: str | None = None) -> dict[str, Any]:
        """The environment at a glance: {state, progress?, total, inline, subsets, subset, facets, overview, how, search}.
        With `subset` (one of `subsets`' ids), its total and facets are that subset's.

        `state` is "ready", or "indexing" with `progress` (the page waits and polls). `inline` true means `tasks` can
        return every task at once (the page then searches and filters as you type); else it pages. `overview` is a list
        of sections (how its tasks are graded, what they run in). `how` says how it's read: {framework, about, roles}.
        Inline lists may add `tiles` (see `tiles`), `map` (see `treemap`, or the name of a facet) and `order`
        ("shuffle": a fresh order each visit, so people who arrive together don't all open the first task)."""
        raise NotImplementedError

    def tasks(self, env: Env, *, subset: str | None = None, q: str = "", filters: dict[str, list[str]] | None = None,
              offset: int = 0, everything: bool = False) -> dict[str, Any]:
        """Tasks as cards: {cards, total, offset, page, note?}. `everything` (inline adapters only) returns them all."""
        raise NotImplementedError

    def task(self, env: Env, ref: str) -> dict[str, Any]:
        """One task in full: {ref, title, id, chips, sections, glance, withheld, links, nav?, hub?}."""
        raise NotImplementedError

    def file(self, env: Env, ref: str, path: str) -> dict[str, Any]:
        """One file of a task, for the viewer: {path, size, text | binary | withheld | error, truncated?}."""
        return {"path": path, "error": "this environment's tasks have no files"}

    def folder(self, env: Env, ref: str, path: str) -> dict[str, Any]:
        """A folder the tree left unlisted: {entries: [{path, size, withheld, dir?, lazy?}], truncated}."""
        raise LookupError("this environment lists its folders up front")

    def random(self, env: Env, subset: str | None = None) -> str:
        """A random task's ref."""
        raise LookupError("no random task here")

    def data(self, env: Env, ref: str, name: str, params: dict[str, str]) -> Any:
        """More of a task, for a custom renderer: JSON by `name` (a database table's rows, a file's preview)."""
        raise LookupError(f"nothing named {name!r} here")

    def raw(self, env: Env, ref: str, path: str) -> tuple[bytes, str]:
        """One of a task's files as it is, (bytes, media type), for an image or a page shown in a sandboxed frame.
        Served with a sandboxing CSP, so a task's HTML never runs with the explorer's origin."""
        raise LookupError("this environment's files aren't served raw")

    # running
    def run_options(self, env: Env, ref: str, view: dict[str, Any]) -> list[dict[str, Any]]:
        """How this task can be run, best first (see `run_option`). Empty when it can't be, with the reason in
        `run_note`."""
        return []

    run_note = "No harness runs this environment's tasks here yet."

    def materialize(self, env: Env, ref: str) -> Path:
        """The task as a Harbor task folder on disk, for the Harbor harness (OpenEnv's runner on an HF Sandbox)."""
        raise LookupError("this task can't be written out as a Harbor task")

    def run_task(self, env: Env, ref: str) -> dict[str, Any]:
        """What a run record needs about the task: {title, restricted, sha, image?, replayed?}."""
        view = self.task(env, ref)
        return {"title": view["title"], "restricted": bool(env.meta.get("restricted")), "sha": env.meta.get("sha")}

    # the same task elsewhere
    def aliases(self, env: Env, ref: str) -> list[tuple[str, str]]:
        """Other (environment key, ref) pairs that are this very task (a raw row and its Harbor conversion): their
        rollouts are listed with this task's."""
        return []

    # agents
    def mcp_tools(self, env: Env) -> list[dict[str, Any]]:
        """Extra MCP tools beyond the ones every environment gets (describe, list_tasks, get_task, read_file)."""
        return []

    def mcp_call(self, env: Env, name: str, args: dict[str, Any]) -> Any:
        raise LookupError(f"no tool named {name!r}")


