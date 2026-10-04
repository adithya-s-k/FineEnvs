"""Search over every environment and every indexed task, server side, from the catalog snapshot (app/snapshot.py).

    GET /api/search?q=&kind=&collection=&f=&sort=&page=&size=     a page of cards, the total, the facet counts
    GET /api/search/tasks?q=&env=&page=&size=                     tasks across indexed datasets, best match first

/api/search answers exactly as the Explore page always filtered its full listing in the browser:

  kind        OpenEnv Spaces require fresh successful API checks. Metadata-only matches are unverified-space; other
              environment Spaces (ORS included), Harbor datasets (tagged, or an index found task folders), Verifiers,
              NeMo Gym, OpenEnv datasets, verl and SkyRL, other RL datasets
  owner       exact Hub namespace, case-insensitive; applies to datasets and Spaces
  collection  an admin's collection, "other" (in none) or "mine" (the signed-in visitor's datasets, with mine=1)
  f           facet filters as the page's own URL writes them: `type:Benchmark|Neither;tags:code` (values
              URI-encoded); values of one facet are alternatives, facets all apply
  q           every word must be part of the id, heading, brief or tags (substrings, any case)
  sort        trending (Hub score, then likes, then monthly dataset downloads; no editorial boosts) | downloads | likes | tasks |
              rollouts | updated | new; ties keep the listing's order

Each facet is counted over what every *other* filter leaves (so picking a value never hides its alternatives).
Hidden environments, pins and collections are read from the admin settings on every query, so an admin's change
shows at once, between indexer runs. The page's query string works as is: `k`, `c` and `s` are read as kind,
collection and sort.

Include it with `app.include_router(search_api.router)`; its startup starts the snapshot watcher.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from . import snapshot, space_checks

router = APIRouter(on_startup=[snapshot.start])

KINDS = ("all", "harbor", "openenv", "unverified-space", "other-space", "verifiers", "nemo-gym", "openenv-data", "verl", "rows")
SORTS = ("trending", "downloads", "likes", "tasks", "rollouts", "updated", "new")
FACETS = ("type", "size", "health", "mode", "tools", "stage", "mcp", "oe", "oe_source", "evidence", "tags")
RANKING = {"version": 2, "source": "Hugging Face Hub", "order": ["trending", "likes", "downloads"],
           "description": "Hub trending score first; ties use total likes, then dataset downloads over the last month. "
                          "Spaces have no download metric. Remaining ties keep catalog order. Featured picks do not affect rank.",
           "scope": "Current search and filters", "quality": "Interest signals, not measured environment quality or live availability."}
OLD_KINDS = {"dataset": "harbor", "space": "openenv", "ors": "other-space"}   # the page's older links
FACET_LIMIT = 100       # values returned per facet (most common first); the rest are counted in facets_more
MAX_WORDS = 16
MAX_VALUES = 50


# ── the query ────────────────────────────────────────────────────────────────
@dataclass
class Query:
    q: str = ""
    scope: str = "all"
    kind: str = "all"
    owner: str | None = None
    coll: str | None = None
    sel: dict[str, list[str]] = field(default_factory=dict)
    sort: str = "trending"
    page: int = 1
    size: int = 40

    @property
    def words(self) -> list[str]:
        return self.q.lower().split()[:MAX_WORDS]

    def public(self) -> dict[str, Any]:
        return {"q": self.q, "scope": self.scope, "owner": self.owner, "kind": self.kind, "collection": self.coll, "f": self.sel, "sort": self.sort,
                "page": self.page, "size": self.size}


def parse_f(f: str) -> dict[str, list[str]]:
    """`key:v1|v2;key2:v` (each value URI-encoded, as the page writes it) as {key: [values]}; unknown keys dropped."""
    sel: dict[str, list[str]] = {}
    for pair in (f or "").split(";"):
        k, _, v = pair.partition(":")
        if k not in FACETS or not v:
            continue
        vals = [unquote(x)[:200] for x in v.split("|") if x]
        if k == "stage" and "Asleep or stopped" in vals:   # saved links from the former broad bucket
            vals = [x for x in vals if x != "Asleep or stopped"] + ["Sleeping", "Paused", "Building", "Starting", "Error", "Deleting"]
        if k == "mcp":
            vals = ["MCP tagged" if x == "Has MCP tools" else x for x in vals]
        for x in vals:
            if x not in sel.setdefault(k, []) and len(sel[k]) < MAX_VALUES:
                sel[k].append(x)
    return {k: v for k, v in sel.items() if v}


def _int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def parse_query(params: Any, collections: list[dict[str, Any]], mine: bool) -> Query:
    """The request's filters, normalised the way the page reads its own URL: an unknown kind, sort or collection is
    no filter rather than an error."""
    g = lambda *names: next((params.get(n) for n in names if params.get(n) not in (None, "")), None)
    kind = g("kind", "k") or "all"
    kind = OLD_KINDS.get(kind, kind)
    coll = g("collection", "c")
    known = {c["id"] for c in collections} | {"other"} | ({"mine"} if mine else set())
    sort = g("sort", "s") or "trending"
    return Query(q=(g("q") or "").strip()[:200], scope="ready" if g("scope") == "ready" else "all", owner=(g("owner") or "").strip()[:200] or None, kind=kind if kind in KINDS else "all", coll=coll if coll in known else None,
                 sel=parse_f(g("f") or ""), sort=sort if sort in SORTS else "trending",
                 page=_int(g("page"), 1, 1, 10_000), size=_int(g("size"), 40, 1, 100))


def envs_fts_query(words: list[str]) -> str | None:
    """The trigram FTS expression for search words: each one a quoted string (quotes doubled), so nothing a visitor
    types is read as FTS syntax (operators, NEAR, *, ^, column filters, parentheses). Words under three characters
    (and non-ASCII ones, whose case folding may differ) are left to the exact check alone."""
    terms = ['"' + w.replace('"', '""') + '"' for w in words if len(w) >= 3 and w.isascii() and w.isprintable()]
    return " ".join(terms) or None


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tasks_fts_query(q: str) -> str | None:
    """The FTS expression for a task search: the words' letters and digits only, each a quoted prefix ("word"*), all
    required. Quotes, operators, NEAR, column filters and stray asterisks can't get through."""
    toks = [t[:40] for t in _TOKEN.findall((q or "").lower())][:8]
    return " ".join(f'"{t}"*' for t in toks) or None


# ── what admins set, read per query ──────────────────────────────────────────
_roll: dict[str, Any] = {"at": 0.0, "map": {}}
_roll_lock = threading.Lock()


def rollout_counts() -> dict[str, int]:
    """Graded public rollouts per dataset (main._shareable's rule), refreshed every 30 s."""
    with _roll_lock:
        if time.time() - _roll["at"] < 30:
            return _roll["map"]
    from . import settings, store

    hidden = set(settings.get("hidden_runs", []))
    counts: dict[str, int] = {}
    for r in store.list_runs(public=True, limit=100000):
        if r.get("status") == "done" and r.get("reward") is not None and not r.get("restricted") and r.get("id") not in hidden:
            k = str(r.get("dataset"))
            counts[k] = counts.get(k, 0) + 1
    with _roll_lock:
        _roll.update(at=time.time(), map=counts)
    return counts


@dataclass
class Ctx:
    collections: list[dict[str, Any]]
    cmap: dict[str, str]
    pins: list[str]
    hidden: list[str]
    mine: list[str]
    rolls: dict[str, int]
    checks: dict[str, dict] = field(default_factory=dict)

    def params(self) -> dict[str, str]:
        """The admin settings as query parameters: each collection's keys, pins, hidden ones, the visitor's, rollouts.
        (Lists checked with IN, which SQLite evaluates once per statement: no per-row JSON.)"""
        out = {"pins": json.dumps(self.pins), "hidden": json.dumps(self.hidden), "mine": json.dumps(self.mine),
               "rolls": json.dumps(self.rolls), "checks": json.dumps([
                   {**r, "status": space_checks.status(r), "browseable": space_checks.browseable(r)} for r in self.checks.values()])}
        for i, g in enumerate(self.collections):
            out[f"c{i}"] = json.dumps([k for k, c in self.cmap.items() if c == g["id"]])
            out[f"cid{i}"] = g["id"]
        return out

    def base(self) -> str:
        """The query's common table: every visible environment with what admins set applied (see BASE)."""
        whens = " ".join(f"WHEN e.key IN (SELECT value FROM json_each(:c{i})) THEN :cid{i}" for i in range(len(self.collections)))
        return BASE.replace("{coll}", f"CASE {whens} END" if whens else "NULL")


def context(mine: list[str] | None = None) -> Ctx:
    from . import catalog

    colls = catalog.collections()
    cmap: dict[str, str] = {}
    for g in colls:
        for k in g.get("ids") or []:
            cmap.setdefault(k, g["id"])   # the first collection naming it, as catalog.collection_of
    return Ctx(collections=[{k: v for k, v in g.items() if k != "ids"} for g in colls], cmap=cmap, pins=list(catalog.pinned()),
               hidden=sorted(catalog.hidden()), mine=list(mine or []), rolls=rollout_counts(), checks=space_checks.inventory())


# ── SQL over the snapshot ────────────────────────────────────────────────────
BASE = """
WITH checks AS MATERIALIZED (
  SELECT CAST(json_extract(value, '$.id') AS TEXT) AS id, value AS check_json FROM json_each(:checks)
), b0 AS (
  SELECT e.rowid AS rid, e.ord, e.id, e.key, e.kind, e.framework, e.openenv, e.heading, e.brief, e.downloads, e.likes,
         e.trending, e.created, e.updated, e.created_ms, e.updated_ms, e.stage, e.mcp, e.openenv_version, e.manifest,
         e.hardware, e.badges, e.tags, e.tasks, e.indexed, e.size_f, e.stage_f, e.mcp_f, e.oe_f, e.blob,
         c.check_json,
         {coll} AS coll,
         (e.key IN (SELECT value FROM json_each(:pins))) AS pinned,
         (e.key IN (SELECT value FROM json_each(:mine))) AS mine
  FROM envs e
  LEFT JOIN checks c ON c.id = e.id AND e.kind = 'space'
  WHERE e.key NOT IN (SELECT value FROM json_each(:hidden))
),
base AS (
  SELECT b0.*,
         CASE WHEN kind = 'space' THEN CASE WHEN {verified} THEN 'openenv' WHEN framework = 'ors' THEN 'ors' ELSE 'space' END ELSE framework END AS framework_q,
         CASE WHEN kind = 'space' AND {verified} THEN 1 ELSE 0 END AS openenv_q,
         CASE WHEN kind = 'space' AND NOT COALESCE(({verified}), 0) THEN
           (SELECT json_group_array(value) FROM json_each(badges) WHERE value != 'OpenEnv') ELSE badges END AS badges_q,
         CASE WHEN kind = 'space' THEN (CASE WHEN {verified} THEN 'openenv' WHEN framework IS 'openenv' OR openenv THEN 'unverified-space' ELSE 'other-space' END)
              ELSE COALESCE(framework, 'harbor') END AS fw,
         CASE WHEN kind = 'space' AND (framework IS 'openenv' OR openenv)
              THEN CASE WHEN manifest IS NOT NULL THEN 'Manifest present' ELSE 'Hub tag only' END END AS evidence_f,
         CASE WHEN kind = 'space' THEN {stage} END AS runtime_f,
         CASE WHEN kind = 'space' THEN CASE WHEN mcp THEN 'MCP tagged' ELSE 'No MCP tag' END END AS mcp_tag_f,
         CASE WHEN kind = 'space' AND (framework IS 'openenv' OR openenv OR ({verified}))
              THEN COALESCE(json_extract(check_json, '$.status'), 'Not checked') END AS health_f,
         CASE WHEN json_extract(check_json, '$.status') = 'API checked' THEN json_extract(check_json, '$.mode') END AS mode_f,
         CASE WHEN kind = 'space' AND (framework IS 'openenv' OR openenv OR ({verified})) THEN
           CASE WHEN json_extract(check_json, '$.status') IN ('API checked', 'Checks failed') THEN
             CASE WHEN json_extract(check_json, '$.tools') > 0 THEN 'Tools discovered' ELSE 'No tools discovered' END
           ELSE 'Not checked' END END AS tools_f,
         CASE WHEN kind = 'space' AND (framework IS 'openenv' OR openenv OR ({verified})) THEN
           COALESCE(NULLIF(json_extract(check_json, '$.version.value'), 'Unknown'), openenv_version, 'Unknown') END AS version_f,
         CASE WHEN kind = 'space' AND (framework IS 'openenv' OR openenv OR ({verified})) THEN
           CASE WHEN json_extract(check_json, '$.version.value') IS NOT NULL AND json_extract(check_json, '$.version.value') != 'Unknown'
             THEN json_extract(check_json, '$.version.source') WHEN openenv_version IS NOT NULL THEN 'Hub tag' ELSE 'Unknown' END END AS version_source_f
  FROM b0
)
""".replace("{stage}", snapshot.stage_sql("stage")).replace("{verified}",
    "json_extract(check_json, '$.status') = 'API checked' AND json_extract(check_json, '$.stage') = 'RUNNING'")
ROLLOUTS = "COALESCE((SELECT value FROM json_each(:rolls) WHERE key = b.id), 0)"


TYPE_VALUES = "json_each(CASE WHEN json_array_length(b.badges_q) = 0 THEN '[\"Neither\"]' ELSE b.badges_q END)"
FACET_FILTER = {
    "type": f"EXISTS (SELECT 1 FROM {TYPE_VALUES} j WHERE j.value IN (SELECT value FROM json_each({{p}})))",
    "size": "b.size_f IN (SELECT value FROM json_each({p}))",
    "stage": "b.runtime_f IN (SELECT value FROM json_each({p}))",
    "health": "b.health_f IN (SELECT value FROM json_each({p}))",
    "mode": "b.mode_f IN (SELECT value FROM json_each({p}))",
    "tools": "b.tools_f IN (SELECT value FROM json_each({p}))",
    "mcp": "b.mcp_tag_f IN (SELECT value FROM json_each({p}))",
    "oe": "b.version_f IN (SELECT value FROM json_each({p}))",
    "oe_source": "b.version_source_f IN (SELECT value FROM json_each({p}))",
    "evidence": "b.evidence_f IN (SELECT value FROM json_each({p}))",
    "tags": "EXISTS (SELECT 1 FROM json_each(b.tags) j WHERE j.value IN (SELECT value FROM json_each({p})))",
}
FACET_COUNT = {   # FROM, value expression, extra condition
    "type": (f"base b, {TYPE_VALUES} j", "j.value", "1"),
    "size": ("base b", "b.size_f", "b.size_f IS NOT NULL"),
    "stage": ("base b", "b.runtime_f", "b.runtime_f IS NOT NULL"),
    "health": ("base b", "b.health_f", "b.health_f IS NOT NULL"),
    "mode": ("base b", "b.mode_f", "b.mode_f IS NOT NULL"),
    "tools": ("base b", "b.tools_f", "b.tools_f IS NOT NULL"),
    "mcp": ("base b", "b.mcp_tag_f", "b.mcp_tag_f IS NOT NULL"),
    "oe": ("base b", "b.version_f", "b.version_f IS NOT NULL"),
    "oe_source": ("base b", "b.version_source_f", "b.version_source_f IS NOT NULL"),
    "evidence": ("base b", "b.evidence_f", "b.evidence_f IS NOT NULL"),
    "tags": ("base b, json_each(b.tags) j", "j.value", "1"),
}
TRENDING = "b.trending"
SORT_SQL = {"trending": TRENDING, "downloads": "b.downloads", "likes": "b.likes", "tasks": "COALESCE(b.tasks, -1)",
            "rollouts": ROLLOUTS, "updated": "b.updated_ms", "new": "b.created_ms"}


def sort_sql(q: Query) -> str:
    return SORT_SQL[q.sort]


def order_sql(q: Query) -> str:
    return "sk DESC, " + ("b.likes DESC, b.downloads DESC, " if q.sort == "trending" else "") + "b.ord ASC"


def order_key(q: Query, d: dict[str, Any]) -> tuple:
    return (-d["_sk"],) + ((-(d.get("likes") or 0), -(d.get("downloads") or 0)) if q.sort == "trending" else ()) + (d["_ord"],)


def where(q: Query, skip: str | None = None, ignore_coll: bool = False, ignore_kind: bool = False) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    params: dict[str, Any] = {}
    if q.scope == "ready":
        parts.append("(b.kind = 'dataset' OR json_extract(b.check_json, '$.browseable') = 1)")
    if not ignore_kind and q.kind != "all":
        parts.append("b.fw = :kind")
        params["kind"] = q.kind
    if not ignore_coll and q.coll:
        if q.coll == "mine":
            parts.append("b.mine")
        else:
            parts.append("COALESCE(b.coll, 'other') = :coll")
            params["coll"] = q.coll
    if q.owner:
        parts.append("substr(b.id, 1, instr(b.id, '/') - 1) = :owner COLLATE NOCASE")
        params["owner"] = q.owner
    words = q.words
    if words:
        fts = envs_fts_query(words)
        if fts:   # the index narrows it down; the exact check below decides
            parts.append("b.rid IN (SELECT rowid FROM envs_fts WHERE envs_fts MATCH :fts)")
            params["fts"] = fts
        for i, w in enumerate(words):
            parts.append(f"instr(b.blob, :w{i}) > 0")
            params[f"w{i}"] = w
    for k, vals in q.sel.items():
        if k == skip or not vals or k not in FACET_FILTER:
            continue
        parts.append(FACET_FILTER[k].format(p=f":f_{k}"))
        params[f"f_{k}"] = json.dumps(vals)
    return (" AND ".join(parts) or "1"), params


def card(r: Any, rolls: dict[str, int]) -> dict[str, Any]:
    """A row as a card: what the list, the trending grid and the quick look show, nothing more."""
    return {"id": r["id"], "key": r["key"], "kind": r["kind"], "framework": r["framework_q"], "fw": r["fw"],
            "openenv": bool(r["openenv_q"]), "collection": r["coll"], "heading": r["heading"], "brief": r["brief"],
            "downloads": r["downloads"], "likes": r["likes"], "trending": r["trending"], "updated": r["updated"],
            "created": r["created"], "stage": r["stage"], "mcp": bool(r["mcp"]), "openenv_version": r["openenv_version"],
            "manifest": r["manifest"], "hardware": r["hardware"], "badges": json.loads(r["badges_q"]), "tags": json.loads(r["tags"]),
            "evidence": r["evidence_f"], "stage_label": r["runtime_f"],
            "api_status": r["health_f"], "api_check": json.loads(r["check_json"]) if r["check_json"] else None,
            "declared_version": r["version_f"], "version_source": r["version_source_f"], "api_mode": r["mode_f"], "tools_status": r["tools_f"],
            "pinned": bool(r["pinned"]), "mine": bool(r["mine"]), "indexed": json.loads(r["indexed"]) if r["indexed"] else None,
            "rollouts": rolls.get(r["id"], 0), "private": False, "_sk": r["sk"], "_ord": r["ord"]}


def _rank(counts: dict[str, int]) -> list[list[Any]]:
    return [[v, n] for v, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def sql_search(conn, q: Query, ctx: Ctx, *, facets: bool = True, limit: int | None = None, offset: int | None = None) -> dict[str, Any]:
    """The query over the snapshot: a page of cards (sorted, `_sk` and `_ord` kept for merging), the total and every
    facet's counts (raw: {value: n})."""
    base, B = ctx.params(), ctx.base()
    w, p = where(q)
    lim = q.size if limit is None else limit
    off = (q.page - 1) * q.size if offset is None else offset
    rows = conn.execute(f"{B} SELECT b.*, {sort_sql(q)} AS sk FROM base b WHERE {w} ORDER BY {order_sql(q)} LIMIT :lim OFFSET :off",
                        {**base, **p, "lim": lim, "off": off}).fetchall()
    total = conn.execute(f"{B} SELECT COUNT(*) FROM base b WHERE {w}", {**base, **p}).fetchone()[0]
    out: dict[str, Any] = {"rows": [card(r, ctx.rolls) for r in rows], "total": total}
    if not facets:
        return out
    fc: dict[str, dict[Any, int]] = {}
    w, p = where(q, ignore_kind=True)
    fc["kind"] = {k: n for k, n in conn.execute(f"{B} SELECT b.fw, COUNT(*) FROM base b WHERE {w} GROUP BY b.fw", {**base, **p})}
    w, p = where(q, ignore_coll=True)
    coll: dict[str, int] = {}
    mine = 0
    for k, n, m in conn.execute(f"{B} SELECT COALESCE(b.coll, 'other'), COUNT(*), SUM(b.mine) FROM base b WHERE {w} GROUP BY 1",
                                {**base, **p}):
        coll[k] = n
        mine += m or 0
    if ctx.mine:
        coll["mine"] = mine
    fc["collection"] = coll
    for key in FACETS:
        frm, val, cond = FACET_COUNT[key]
        w, p = where(q, skip=key)
        fc[key] = {v: n for v, n in conn.execute(f"{B} SELECT {val}, COUNT(*) FROM {frm} WHERE {cond} AND {w} GROUP BY {val}",
                                                  {**base, **p})}
    out["facets"] = fc
    return out


def sql_extra(conn, ctx: Ctx, trending: int, q: Query | None = None) -> dict[str, Any]:
    """Counts, task coverage and editorial picks all follow the current search."""
    base, B = ctx.params(), ctx.base()
    w, p = where(q or Query())
    params = {**base, **p}
    ds, sp = conn.execute(f"{B} SELECT SUM(b.kind = 'dataset'), SUM(b.kind = 'space') FROM base b WHERE {w}", params).fetchone()
    # Count actual searchable records, including adapters (MiMo) with no Harbor
    # summary. Health discovery must never imply that tasks were downloaded.
    tasks = conn.execute(f"{B} SELECT COUNT(*) FROM tasks t JOIN base b ON b.key=t.env WHERE b.kind='dataset' AND {w}", params).fetchone()[0]
    from . import space_tasks
    visible = [r[0] for r in conn.execute(f"{B} SELECT b.id FROM base b WHERE b.kind='space' AND {w}", params)]
    space_counts = space_tasks.census(ctx.checks, visible)
    out: dict[str, Any] = {"stats": {"datasets": ds or 0, "spaces": sp or 0, "tasks": tasks,
                                    "dataset_tasks": tasks, "space_tasks": space_counts["tasks"],
                                    "space_task_coverage": space_counts}}
    out["stats"]["openenv"] = dict(conn.execute(f"{B} SELECT b.evidence_f, COUNT(*) FROM base b WHERE b.evidence_f IS NOT NULL AND {w} GROUP BY 1", params))
    out["stats"]["openenv_running"] = conn.execute(f"{B} SELECT COUNT(*) FROM base b WHERE b.evidence_f IS NOT NULL AND b.stage = 'RUNNING' AND {w}", params).fetchone()[0]
    if trending:
        tq = replace(q or Query(), sort="trending", page=1, size=trending)
        out["trending"] = sql_search(conn, tq, ctx, facets=False)["rows"]
        w, p = where(tq)
        rows = conn.execute(f"{B} SELECT b.*, 0 AS sk FROM base b WHERE b.pinned AND {w} "
                            "ORDER BY (SELECT CAST(key AS INTEGER) FROM json_each(:pins) WHERE value = b.key) LIMIT 12", {**base, **p})
        out["featured"] = [card(r, ctx.rolls) for r in rows]
        # FineEnvs is an explicit editorial section, independent of admin pins
        # and interest rank. Owner/search filters still apply to it.
        promoted = conn.execute(f"{B} SELECT b.*, {TRENDING} AS sk FROM base b WHERE {w} "
                               "AND substr(b.id, 1, instr(b.id, '/') - 1) = 'FineEnvs' COLLATE NOCASE "
                               "AND (b.kind = 'dataset' OR json_extract(b.check_json, '$.browseable') = 1) "
                               "ORDER BY (b.kind = 'space') DESC, b.pinned DESC, " + order_sql(tq) + " LIMIT 4", {**base, **p})
        out["promoted"] = [card(r, ctx.rolls) for r in promoted]
    return out


# ── the same, in Python: the visitor's own datasets that aren't in the snapshot, and the tests' reference ────────
def py_fw(d: dict[str, Any], check=None) -> str:
    if d.get("kind") == "space":
        if space_checks.verified(check):
            return "openenv"
        return "unverified-space" if d.get("framework") == "openenv" or d.get("openenv") else "other-space"
    return d.get("framework") or "harbor"


PY_FACETS = {
    "type": lambda d: d.get("badges") or ["Neither"],
    "size": lambda d: [snapshot.size_bucket((d.get("indexed") or {}).get("tasks"))]
    if d.get("kind") == "dataset" and (d.get("framework") or "harbor") == "harbor" else [],
    "stage": lambda d: [snapshot.stage_value(d.get("stage"))] if d.get("kind") == "space" else [],
    "mcp": lambda d: (["MCP tagged" if d.get("mcp") else "No MCP tag"]) if d.get("kind") == "space" else [],
    "health": lambda d: [d["api_status"]] if d.get("api_status") else [],
    "mode": lambda d: [d["api_mode"]] if d.get("api_mode") else [],
    "tools": lambda d: [d["tools_status"]] if d.get("tools_status") else [],
    "oe": lambda d: [d["declared_version"]] if d.get("declared_version") else [],
    "oe_source": lambda d: [d["version_source"]] if d.get("version_source") else [],
    "evidence": lambda d: [d["evidence"]] if d.get("evidence") else [],
    "tags": lambda d: d.get("tags") or [],
}


def py_cards(rows: list[dict[str, Any]], ctx: Ctx, ord0: int = 0) -> list[dict[str, Any]]:
    """Apply visibility and editorial metadata without changing framework evidence or rank."""
    hidden, pins, mine = set(ctx.hidden), set(ctx.pins), set(ctx.mine)
    out = []
    for i, r in enumerate(rows):
        key = r.get("key") or r["id"]
        if key in hidden:
            continue
        coll = ctx.cmap.get(key)
        d = {**r, "key": key, "collection": coll, "pinned": key in pins, "mine": key in mine or bool(r.get("mine")),
             "rollouts": ctx.rolls.get(str(r["id"]), 0), "_ord": ord0 + i, "_blob": snapshot.blob_of(r),
             "badges": list(r.get("badges") or []), "tags": list(r.get("tags") or [])}
        check = ctx.checks.get(d["id"], {}) if d.get("kind") == "space" else {}
        declared = d.get("kind") == "space" and (d.get("framework") == "openenv" or d.get("openenv"))
        d["fw"] = py_fw(d, check)
        d["evidence"] = ("Manifest present" if d.get("manifest") else "Hub tag only") if declared else None
        d["stage_label"] = snapshot.stage_value(d.get("stage")) if d.get("kind") == "space" else None
        health = space_checks.status(check)
        candidate = declared or d["fw"] == "openenv"
        ver = check.get("version") or {}
        known_ver = ver.get("value") and ver["value"] != "Unknown"
        d.update(api_check={**check, "status": health} if check else None, api_status=health if candidate else None,
                 api_mode=check.get("mode") if health == space_checks.PASS else None,
                 tools_status=("Tools discovered" if check.get("tools", 0) > 0 else "No tools discovered")
                     if candidate and health in (space_checks.PASS, "Checks failed") else "Not checked" if candidate else None,
                 declared_version=(ver["value"] if known_ver else d.get("openenv_version") or "Unknown") if candidate else None,
                 version_source=(ver.get("source") if known_ver else "Hub tag" if d.get("openenv_version") else "Unknown") if candidate else None)
        if d.get("kind") == "space":
            d["openenv"] = d["fw"] == "openenv"
            d["framework"] = "openenv" if d["openenv"] else "ors" if d.get("framework") == "ors" else "space"
            if not d["openenv"]:
                d["badges"] = [b for b in d["badges"] if b != "OpenEnv"]
        out.append(d)
    return out


def _time(iso: Any) -> int:
    return snapshot.iso_ms(iso)


def py_key(q: Query):
    keys = {"trending": lambda d: d.get("trending") or 0, "downloads": lambda d: d.get("downloads") or 0,
            "likes": lambda d: d.get("likes") or 0, "tasks": lambda d: ((d.get("indexed") or {}).get("tasks") if (d.get("indexed") or {}).get("tasks") is not None else -1),
            "rollouts": lambda d: d["rollouts"], "updated": lambda d: _time(d.get("updated")), "new": lambda d: _time(d.get("created"))}
    return keys[q.sort]


def py_passes(d: dict[str, Any], q: Query, skip: str | None = None, ignore_coll: bool = False, ignore_kind: bool = False) -> bool:
    """The page's own `passes()`, line for line."""
    if q.scope == "ready" and d.get("kind") == "space" and not space_checks.browseable(d.get("api_check")):
        return False
    if not ignore_kind and q.kind != "all" and d["fw"] != q.kind:
        return False
    if not ignore_coll and q.coll and not (d["mine"] if q.coll == "mine" else (d.get("collection") or "other") == q.coll):
        return False
    if q.owner and d["id"].split("/", 1)[0].lower() != q.owner.lower():
        return False
    if q.words and not all(w in d["_blob"] for w in q.words):
        return False
    for k, vals in q.sel.items():
        if k == skip or not vals or k not in PY_FACETS:
            continue
        if not any(v in vals for v in PY_FACETS[k](d)):
            return False
    return True


def py_search(cards: list[dict[str, Any]], q: Query, *, with_mine: bool = False) -> dict[str, Any]:
    """`cards` (from py_cards) filtered, sorted and counted as the page did it: every match (sorted) and raw facets."""
    key = py_key(q)
    matches = [d for d in cards if py_passes(d, q)]
    for d in matches:
        d["_sk"] = key(d)
    matches.sort(key=lambda d: order_key(q, d))
    fc: dict[str, dict[Any, int]] = {"kind": {}, "collection": {}}
    for d in cards:
        if py_passes(d, q, ignore_kind=True):
            fc["kind"][d["fw"]] = fc["kind"].get(d["fw"], 0) + 1
        if py_passes(d, q, ignore_coll=True):
            c = d.get("collection") or "other"
            fc["collection"][c] = fc["collection"].get(c, 0) + 1
            if with_mine and d["mine"]:
                fc["collection"]["mine"] = fc["collection"].get("mine", 0) + 1
    if with_mine:
        fc["collection"].setdefault("mine", 0)
    for k in FACETS:
        counts: dict[Any, int] = {}
        for d in cards:
            if py_passes(d, q, skip=k):
                for v in PY_FACETS[k](d):
                    counts[v] = counts.get(v, 0) + 1
        fc[k] = counts
    return {"matches": matches, "total": len(matches), "facets": fc}


def _public_card(d: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "key", "kind", "framework", "fw", "openenv", "collection", "heading", "brief", "downloads", "likes",
            "trending", "updated", "created", "stage", "mcp", "openenv_version", "manifest", "hardware", "badges", "tags",
            "pinned", "mine", "indexed", "rollouts", "private", "evidence", "stage_label", "api_status", "api_check",
            "declared_version", "version_source", "api_mode", "tools_status")
    out = {k: d.get(k) for k in keep}
    out["openenv"], out["mcp"], out["private"] = bool(out["openenv"]), bool(out["mcp"]), bool(out["private"])
    ix = d.get("indexed")
    out["indexed"] = {k: ix.get(k) for k in ("tasks", "graded", "image") if ix.get(k) is not None} if isinstance(ix, dict) else None
    return out


def environments(include_hidden: bool = False) -> list[dict[str, Any]]:
    """Every public environment as catalog.environments() shapes it (collection, pins and hidden ones as admins set
    them now), from the snapshot, else the live catalog: for code that wants the whole listing (app/seo.py)."""
    ctx = context()
    if include_hidden:
        ctx.hidden = []
    with snapshot.use() as (_, conn):
        rows = conn.execute(f"{ctx.base()} SELECT b.*, 0 AS sk FROM base b ORDER BY b.ord", ctx.params()).fetchall()
        out = [card(r, ctx.rolls) for r in rows]
    for d in out:
        d.pop("_sk", None)
        d.pop("_ord", None)
        d["gated"] = False
    return out


# ── the endpoints ────────────────────────────────────────────────────────────
ANSWER_TTL, ANSWERS = 15.0, 256   # recent public answers kept in memory: the first page everyone opens is computed once
_answers: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_answers_lock = threading.Lock()


def _respond(request: Request, payload: dict[str, Any], cache: str, ms: float | None = None) -> Response:
    """Compact JSON with a weak ETag (a repeat comes back 304), the given Cache-Control, and how long the query took
    as Server-Timing (not in the body, so the ETag only changes when the answer does)."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    etag = 'W/"' + hashlib.sha1(body).hexdigest()[:20] + '"'
    headers = {"Cache-Control": cache, "ETag": etag}
    if ms is not None:
        headers["Server-Timing"] = f"db;dur={ms:.1f}"
    if cache.startswith("private"):
        headers["Vary"] = "Cookie"
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type="application/json", headers=headers)


MINE_FRESH, MINE_STALE = 120.0, 3600.0
_mine: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()   # by token hash, never the token
_mine_lock = threading.Lock()
_mine_refreshing: set[str] = set()


def _mine_rows(request: Request) -> list[dict[str, Any]] | None:
    """The signed-in visitor's own datasets (catalog.mine), or None when signed out. Kept per token: after two minutes
    the last list is still answered at once while a fresh one is fetched in the background, so a visitor's searches
    never wait on the Hub (catalog.mine is a few Hub calls)."""
    from . import auth, catalog

    u = auth.current_user(request)
    if not u or not u.get("token"):
        return None
    token = u["token"]
    who = hashlib.sha256(token.encode()).hexdigest()[:16]

    def fetch() -> list[dict[str, Any]]:
        try:
            rows = catalog.mine(token)
        except Exception:  # noqa: BLE001 - the Hub is down: no "Yours" this time
            return []
        with _mine_lock:
            _mine[who] = (time.time(), rows)
            _mine.move_to_end(who)
            while len(_mine) > 500:
                _mine.popitem(last=False)
        return rows

    with _mine_lock:
        hit = _mine.get(who)
    age = time.time() - hit[0] if hit else None
    if hit is None or age > MINE_STALE:
        return fetch()
    if age > MINE_FRESH and who not in _mine_refreshing:
        _mine_refreshing.add(who)

        def refresh() -> None:
            try:
                fetch()
            finally:
                _mine_refreshing.discard(who)

        threading.Thread(target=refresh, daemon=True, name="mine-refresh").start()
    return hit[1]


@router.get("/api/search")
def search(request: Request):
    """A page of environment cards, the total and the facet counts; the page's header numbers and trending row on
    request (`trending=<n>`). `facets=0` leaves the counts out (paging), `mine=1` adds the signed-in visitor's own
    datasets (a private response)."""
    params = request.query_params
    want_mine = params.get("mine") in ("1", "true")
    mine_rows = _mine_rows(request) if want_mine else None
    ctx = context()
    q = parse_query(params, ctx.collections, mine=bool(mine_rows))
    facets = params.get("facets", "1") not in ("0", "false")
    trending = _int(params.get("trending"), 0, 0, 48)
    t0 = time.time()
    cache = "private, max-age=30" if want_mine else "public, max-age=30, stale-while-revalidate=120"
    key = None
    if not want_mine:   # the same question about the same snapshot and settings: answered from memory for a while
        try:
            key = json.dumps([snapshot.get().name, ctx.params(), q.public(), facets, trending], sort_keys=True)
        except snapshot.SnapshotError as e:
            raise HTTPException(503, f"the catalog isn't ready: {e}")
        hit = _answers.get(key)
        if hit and time.time() - hit[0] < ANSWER_TTL:
            return _respond(request, hit[1], cache, (time.time() - t0) * 1000)
    try:
        with snapshot.use() as (snap, conn):
            extras: list[dict[str, Any]] = []
            if mine_rows:
                keys = [r.get("key") or r["id"] for r in mine_rows]
                ctx.mine = keys
                present = {k for (k,) in conn.execute("SELECT key FROM envs WHERE key IN (SELECT value FROM json_each(?))", (json.dumps(keys),))}
                extras = [r for r in mine_rows if (r.get("key") or r["id"]) not in present]
            if extras:   # the visitor's datasets the snapshot doesn't have (private ones): merged in, same rules
                off = (q.page - 1) * q.size
                res = sql_search(conn, q, ctx, facets=facets, limit=off + q.size, offset=0)
                more = py_search(py_cards([{**r, "key": r.get("key") or r["id"], "kind": "dataset", "mine": True} for r in extras],
                                          ctx, ord0=10**9), q, with_mine=True)
                merged = sorted(res["rows"] + more["matches"], key=lambda d: order_key(q, d))
                res["rows"], res["total"] = merged[off:off + q.size], res["total"] + more["total"]
                if facets:
                    for k, counts in more["facets"].items():
                        for v, n in counts.items():
                            res["facets"][k][v] = res["facets"][k].get(v, 0) + n
            else:
                res = sql_search(conn, q, ctx, facets=facets)
            head = sql_extra(conn, ctx, trending, q) if trending or facets else {}
            if extras and head:
                head["stats"]["datasets"] += more["total"]
                private_tasks = sum((r.get("indexed") or {}).get("tasks") or 0 for r in more["matches"])
                for metric in ("tasks", "dataset_tasks"):
                    head["stats"][metric] += private_tasks
                if trending:
                    tq = replace(q, sort="trending")
                    more = py_search(py_cards([{**r, "key": r.get("key") or r["id"], "kind": "dataset", "mine": True} for r in extras],
                                              ctx, ord0=10**9), tq)["matches"]
                    head["trending"] = sorted(head["trending"] + more, key=lambda d: order_key(tq, d))[:trending]
            info = {"built_at": snap.built_at, "source": snap.source}
    except snapshot.SnapshotError as e:
        raise HTTPException(503, f"the catalog isn't ready: {e}")
    out: dict[str, Any] = {"rows": [_public_card(d) for d in res["rows"]], "total": res["total"], "query": q.public()}
    if facets:
        fc = res["facets"]
        out["facets"] = {"kind": {**fc["kind"], "all": sum(fc["kind"].values())}, "collection": fc["collection"]}
        out["facets_more"] = {}
        for k in FACETS:
            ranked = _rank(fc[k])
            out["facets"][k] = ranked[:FACET_LIMIT]
            if len(ranked) > FACET_LIMIT:
                out["facets_more"][k] = len(ranked) - FACET_LIMIT
        out["collections"] = ctx.collections
        out["mine"] = len(mine_rows or [])
    if head.get("stats"):
        out["stats"] = head["stats"]
    if trending:
        out["trending"] = [_public_card(d) for d in head.get("trending", [])]
        out["featured"] = [_public_card(d) for d in head.get("featured", [])]
        out["promoted"] = [_public_card(d) for d in head.get("promoted", [])]
    out["ranking"] = RANKING
    out["api_checks"] = {"fresh_seconds": space_checks.FRESH, "description": "Running on the Hub and OpenEnv API checks passed within the last hour. "
                         "No episode or reward validation. Versions come from repository dependencies or Hub tags, not the installed runtime."}
    out["snapshot"] = info
    if key is not None:
        with _answers_lock:
            _answers[key] = (time.time(), out)
            while len(_answers) > ANSWERS:
                _answers.popitem(last=False)
    return _respond(request, out, cache, (time.time() - t0) * 1000)


@router.get("/api/search/tasks")
def search_tasks(request: Request, q: str = "", env: str = "", page: int = 1, size: int = 20):
    """Tasks across every indexed dataset, best match first: the dataset, the task's ref (its page is
    /t/<env>/<ref>), title, brief and facets. Every word must start a word of the title, brief, path, category or
    tags. At most 10,000 are counted."""
    expr = tasks_fts_query(q[:200])
    page, size = _int(page, 1, 1, 500), _int(size, 20, 1, 50)
    if env and not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", env):
        raise HTTPException(400, "env: a dataset id, like org/name")
    if not expr:
        return _respond(request, {"rows": [], "total": 0, "q": q, "snapshot": None}, "public, max-age=60")
    from . import catalog

    t0 = time.time()
    p = {"m": expr, "hidden": json.dumps(sorted(catalog.hidden())), "env": env, "lim": size, "off": (page - 1) * size}
    w = "tasks_fts MATCH :m AND t.env NOT IN (SELECT value FROM json_each(:hidden))" + (" AND t.env = :env" if env else "")
    try:
        with snapshot.use() as (snap, conn):
            rows = conn.execute("SELECT t.env, t.ref, t.title, t.brief, t.category, t.difficulty, t.grading, t.run "
                                f"FROM tasks_fts JOIN tasks t ON t.rowid = tasks_fts.rowid WHERE {w} "
                                "ORDER BY tasks_fts.rank LIMIT :lim OFFSET :off", p).fetchall()
            total = conn.execute(f"SELECT COUNT(*) FROM (SELECT 1 FROM tasks_fts JOIN tasks t ON t.rowid = tasks_fts.rowid WHERE {w} LIMIT 10001)",
                                 p).fetchone()[0]
            info = {"built_at": snap.built_at, "source": snap.source}
    except snapshot.SnapshotError as e:
        raise HTTPException(503, f"the catalog isn't ready: {e}")
    out = {"rows": [{"env": r["env"], "ref": r["ref"], "title": r["title"], "brief": r["brief"], "category": r["category"],
                     "difficulty": r["difficulty"], "grading": r["grading"], "run": r["run"]} for r in rows],
           "total": min(total, 10000), "more": total > 10000, "q": q, "page": page, "size": size,
           "snapshot": info}
    return _respond(request, out, "public, max-age=60, stale-while-revalidate=300", (time.time() - t0) * 1000)


@router.get("/api/search/rank")
def search_rank(key: str, request: Request, scope: str = "all"):
    """Where one environment stands among its kind (Spaces, or datasets) by the page's trending order:
    {n, of}. One query, so a page can say "#599 of 6,818 Spaces" without loading the whole listing."""
    ctx = context()
    w, params = where(Query(scope="ready" if scope == "ready" else "all"))
    sql = ctx.base() + f""", ranked AS (
SELECT key, ROW_NUMBER() OVER (PARTITION BY kind ORDER BY trending DESC, likes DESC, downloads DESC, ord) AS n,
       COUNT(*) OVER (PARTITION BY kind) AS total FROM base b WHERE {w})
SELECT n, total FROM ranked WHERE key = :key"""
    try:
        with snapshot.use() as (_snap, conn):
            row = conn.execute(sql, {**ctx.params(), **params, "key": key}).fetchone()
    except snapshot.SnapshotError as e:
        raise HTTPException(503, f"the catalog isn't ready: {e}")
    if not row:
        raise HTTPException(404, "not in the catalog")
    return _respond(request, {"n": row[0], "of": row[1]}, "public, max-age=60, stale-while-revalidate=300")


@router.get("/api/search/status")
def search_status():
    """Which snapshot this process reads, and when it last looked for a new one."""
    return snapshot.status()
