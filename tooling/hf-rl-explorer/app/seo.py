"""Search engines and link previews. Every page of the app has its own path, and the server answers each with the app
plus, in its head, the page's own title, description, canonical address, Open Graph and Twitter card, and schema.org
data (a Dataset for an environment, a SoftwareApplication for a Space, a task as part of its environment, breadcrumbs);
in its body, the page's content as plain HTML, for crawlers that don't run JavaScript (the app replaces it as it
starts). Plus robots.txt, a sitemap of every environment and every indexed task, and a preview image per page.

Catalog pages use public indexes. Row and Space task pages may make bounded anonymous
reads of the same withheld task views shown in the UI. Crawlers never start indexing,
wake a Space, run an episode or receive a visitor's credentials.
"""

from __future__ import annotations

import html
import io
import json
import re
import threading
import time
from functools import lru_cache
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from . import catalog, config, seo_tasks, snapshot, space_checks, spaces_live

SITE = "HF RL Explorer"
TAGLINE = "RL environments on the Hugging Face Hub"
DESCRIPTION = ("Explore reinforcement learning environments and tasks on the Hugging Face Hub. Browse OpenEnv, Harbor, "
               "MiMo, NeMo Gym and Verifiers, inspect rewards, and run supported agent rollouts.")
KIND = {"harbor": "Harbor dataset", "verifiers": "Verifiers environment", "nemo-gym": "NeMo Gym dataset", "rows": "RL dataset",
        "mimo": "MiMo RL release", "openenv": "OpenEnv Space", "space": "environment Space"}
MIMO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
NOINDEX = re.compile(r"^/(run/|runs$|compare/)")
SITEMAP_CHUNK = 10_000


# ── where we are ─────────────────────────────────────────────────────────────
def base_url(request: Request) -> str:
    """The site's own origin, for canonical addresses: the Space's public URL, else what the request came to."""
    if config.PUBLIC_URL:
        return config.PUBLIC_URL
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"


def enc(spec: str) -> str:
    return "/".join(quote(p, safe="") for p in spec.split("/"))


def esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"[#*_`>|\[\]]+", " ", str(text or ""))).strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


@lru_cache(maxsize=4)
def _template(mtime: float) -> str:
    return (config.WEB_DIR / "index.html").read_text()


def template() -> str:
    p = config.WEB_DIR / "index.html"
    return _template(p.stat().st_mtime)


# ── what we know, cheaply ────────────────────────────────────────────────────
_listing: dict[str, Any] = {"at": 0.0, "by": {}, "rows": []}
_lock = threading.Lock()


def listing() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The catalog's listing of public environments, by key (`org/name`, `space:org/name`), refreshed every 10 min."""
    with _lock:
        if time.time() - _listing["at"] < 600 and _listing["rows"]:
            return _listing["by"], _listing["rows"]
    try:
        from . import search_api

        rows = search_api.environments()   # the catalog snapshot (public rows only), else the live catalog
    except Exception:  # noqa: BLE001 - the Hub is unreachable: pages still render, with less in them
        rows = []
    by = {r["key"]: r for r in rows}
    with _lock:
        _listing.update(at=time.time(), by=by, rows=rows)
    return by, rows


def public_rows(rows):
    hidden = set(catalog.hidden())
    return [r for r in rows if r["key"] not in hidden and not any(r.get(k) for k in ("private", "gated", "restricted"))]


def discoverable(r, records=None):
    return r.get("kind") == "dataset" or space_checks.browseable((space_checks.inventory() if records is None else records).get(r["id"]))


def index_rows(spec: str) -> list[dict[str, Any]]:
    """An environment's tasks, if they are already known here (never built for a crawler): [{path, title, brief, category}]."""
    if spec == MIMO:
        return _mimo_rows()
    try:
        p = catalog._index_path(spec)
        mtime = p.stat().st_mtime
    except (OSError, ValueError):
        try:
            return _snapshot_rows(spec, snapshot.get().name)
        except snapshot.SnapshotError:
            return []
    return _harbor_rows(spec, mtime)


@lru_cache(maxsize=1)
def _mimo_rows() -> list[dict[str, Any]]:
    from .mimo import catalog as mcat

    return [{"path": e["id"], "title": e["t"], "brief": e["s"], "category": e["d"]} for e in mcat.index()["envs"]]


@lru_cache(maxsize=32)
def _harbor_rows(spec: str, mtime: float) -> list[dict[str, Any]]:
    idx = catalog._read_index(spec)
    if not idx or (idx.get("info") or {}).get("restricted"):
        return []
    return [{"path": t["path"], "title": t.get("title"), "brief": t.get("brief"), "category": t.get("category")} for t in (idx or {}).get("tasks") or []]


@lru_cache(maxsize=32)
def _snapshot_rows(spec: str, revision: str) -> list[dict[str, Any]]:
    with snapshot.use() as (_, conn):
        return [dict(r) for r in conn.execute("SELECT ref AS path, title, brief, category FROM tasks WHERE env = ? ORDER BY ref", (spec,))]


def display_name(row: dict[str, Any] | None, spec: str) -> str:
    """What to call an environment: its card's heading when that names it, else the repository name with the heading."""
    repo = spec.split("/")[-1]
    heading = (row or {}).get("heading") or ""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())  # noqa: E731
    if not heading or norm(heading) == norm(repo):
        return repo
    return heading if norm(repo) in norm(heading) else f"{repo}: {heading}"


def kind_of(row: dict[str, Any] | None, spec: str = "") -> str:
    if spec == MIMO:
        return KIND["mimo"]
    if not row:
        return "RL environment"
    if row.get("kind") == "space":
        if row.get("fw") == "unverified-space":
            return "Unverified Space"
        return {"openenv": KIND["openenv"], "ors": "ORS Space"}.get(row.get("framework") or ("openenv" if row.get("openenv") else ""), KIND["space"])
    return KIND.get(row.get("framework") or "", "RL dataset")


# ── a page ───────────────────────────────────────────────────────────────────
def page(request: Request, *, title: str, description: str, path: str, body: str = "", jsonld: list[dict[str, Any]] | None = None,
         image: str | None = None, noindex: bool = False, og_type: str = "website", status: int = 200) -> HTMLResponse:
    base = base_url(request)
    url = base + path
    full = title if SITE in title else f"{title} · {SITE}"
    img = base + (image or "/social/rl-explorer.png")
    head = "\n".join([
        f'<link rel="canonical" href="{esc(url)}">',
        f'<meta name="rlx-page-url" content="{esc(url)}">',
        '<meta name="robots" content="noindex, follow">' if noindex else '<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">',
        f'<meta property="og:site_name" content="{SITE}">',
        f'<meta property="og:type" content="{og_type}">',
        f'<meta property="og:title" content="{esc(full)}">',
        f'<meta property="og:description" content="{esc(description)}">',
        f'<meta property="og:url" content="{esc(url)}">',
        f'<meta property="og:image" content="{esc(img)}">',
        '<meta property="og:image:type" content="image/png">',
        '<meta property="og:locale" content="en_US">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{esc(full)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(full)}">',
        f'<meta name="twitter:description" content="{esc(description)}">',
        f'<meta name="twitter:image" content="{esc(img)}">',
        f'<meta name="twitter:image:alt" content="{esc(full)}">',
        *[f'<script type="application/ld+json">{_ld(x)}</script>' for x in jsonld or []],
    ])
    doc = template()
    doc = re.sub(r"<title>.*?</title>", f"<title>{esc(full)}</title>", doc, count=1, flags=re.S)
    doc = re.sub(r'<meta name="description" content="[^"]*">', f'<meta name="description" content="{esc(description)}">', doc, count=1)
    doc = doc.replace("</head>", f"{head}\n</head>", 1)
    if body:   # the page as plain HTML until the app starts (crawlers that don't run JavaScript read this)
        doc = re.sub(r'<main id="view">(.*?)</main>', lambda m: f'<main id="view">{m.group(1)}<div class="wrap page ssr">{body}</div></main>', doc, count=1, flags=re.S)
    return HTMLResponse(doc, status_code=status, headers={"Cache-Control": "no-cache", **({"Retry-After": "60"} if status == 503 else {})})


def _ld(x: dict[str, Any]) -> str:
    # inside a <script>: "</" must not end it
    return json.dumps({"@context": "https://schema.org", **x}, ensure_ascii=False).replace("</", "<\\/")


def crumbs(base: str, items: list[tuple[str, str]]) -> tuple[str, dict[str, Any]]:
    """Breadcrumbs as HTML and as schema.org BreadcrumbList: [(name, path)]."""
    html_ = '<nav class="crumbs" aria-label="Breadcrumb">' + " › ".join(
        f'<a href="{esc(p)}">{esc(n)}</a>' if p else f"<span>{esc(n)}</span>" for n, p in items) + "</nav>"
    ld = {"@type": "BreadcrumbList", "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, **({"item": base + p} if p else {})}
                                                          for i, (n, p) in enumerate(items)]}
    return html_, ld


# ── the pages ────────────────────────────────────────────────────────────────
def home(request: Request) -> HTMLResponse:
    base = base_url(request)
    by, rows = listing()
    checks = space_checks.inventory()
    rows = [r for r in public_rows(rows) if discoverable(r, checks)]
    top = sorted(rows, key=lambda r: (r.get("trending") or 0) * 1e9 + (r.get("downloads") or 0), reverse=True)[:120]
    desc = DESCRIPTION
    body = (f"<header class=\"tp-head\"><h1>{TAGLINE}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + _env_list(top) + '<p><a href="/community">Community rollouts</a> · <a href="/d/XiaomiMiMo/MiMo-V2.6-RL-oss">MiMo-V2.6 RL</a></p>')
    ld = [{"@type": "WebSite", "name": SITE, "alternateName": TAGLINE, "url": base + "/", "description": desc,
           "potentialAction": {"@type": "SearchAction", "target": {"@type": "EntryPoint", "urlTemplate": base + "/?q={search_term_string}"},
                               "query-input": "required name=search_term_string"},
           "publisher": {"@type": "Organization", "name": "FineEnvs", "url": "https://huggingface.co/FineEnvs"}},
          {"@type": "CollectionPage", "name": TAGLINE, "url": base + "/", "about": "Reinforcement learning environments",
           "mainEntity": {"@type": "ItemList", "numberOfItems": len(top), "itemListElement": [
               {"@type": "ListItem", "position": i + 1, "url": base + _href(r), "name": r.get("heading") or r["id"]} for i, r in enumerate(top[:50])]}}]
    return page(request, title=f"{SITE}: {TAGLINE}", description=desc, path="/", body=body, jsonld=ld)


def _href(r: dict[str, Any]) -> str:
    return f"/{'s' if r['kind'] == 'space' else 'd'}/{enc(r['id'])}"


def _env_list(rows: list[dict[str, Any]]) -> str:
    return "<ul>" + "".join(f'<li><a href="{_href(r)}">{esc(r.get("heading") or r["id"])}</a> <span>({esc(r["id"])}, {esc(kind_of(r, r["id"]))})</span>'
                            + (f" <span>{esc(clip(r.get('brief') or '', 160))}</span>" if r.get("brief") else "") + "</li>" for r in rows) + "</ul>"


def environment(request: Request, spec: str) -> HTMLResponse:
    base = base_url(request)
    by, _ = listing()
    r = next((r for r in public_rows(list(by.values())) if r["key"] == spec), None)
    name = display_name(r, spec)
    kind = kind_of(r, spec)
    tasks = index_rows(spec) if r else []
    n = len(tasks) or ((r or {}).get("indexed") or {}).get("tasks")
    brief = clip((r or {}).get("brief") or "", 300)
    desc = clip(f"{name}: {kind} on Hugging Face{f' with {n:,} tasks' if n else ''}. "
                f"{brief or 'See what each task asks, how it is graded and what it runs in, then run an agent on it.'}", 300)
    path = f"/d/{enc(spec)}"
    cr, cld = crumbs(base, [("Environments", "/"), (name, "")])
    body = (cr + f"<header class=\"tp-head\"><h1>{esc(spec)}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + (f"<h2>Tasks</h2><ul>" + "".join(f'<li><a href="/t/{enc(spec)}/{enc(t["path"])}">{esc(clip(t.get("title") or t["path"], 140))}</a></li>'
                                              for t in tasks[:150]) + "</ul>" + (f"<p>{len(tasks) - 150:,} more tasks.</p>" if len(tasks) > 150 else "") if tasks else "")
            + f'<p><a href="https://huggingface.co/datasets/{enc(spec)}" rel="noopener">{esc(spec)} on the Hugging Face Hub</a></p>')
    ld = [{"@type": "Dataset", "name": name, "alternateName": spec, "description": desc, "url": base + path,
           "sameAs": f"https://huggingface.co/datasets/{spec}", "identifier": spec, "isAccessibleForFree": True,
           "creator": {"@type": "Organization", "name": spec.split("/")[0], "url": f"https://huggingface.co/{spec.split('/')[0]}"},
           "keywords": [k for k in ["reinforcement learning", "RL environment", kind, *((r or {}).get("tags") or [])[:12]] if k],
           **({"dateModified": r["updated"]} if r and r.get("updated") else {}), **({"dateCreated": r["created"]} if r and r.get("created") else {}),
           "includedInDataCatalog": {"@type": "DataCatalog", "name": SITE, "url": base + "/"},
           "distribution": [{"@type": "DataDownload", "encodingFormat": "application/octet-stream", "contentUrl": f"https://huggingface.co/datasets/{spec}"}]},
          cld]
    return page(request, title=f"{name} · {kind}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if r else None, noindex=not r)


def task(request: Request, spec: str, ref: str) -> HTMLResponse:
    base = base_url(request)
    by, _ = listing()
    r = next((r for r in public_rows(list(by.values())) if r["key"] == spec), None)
    env_name = display_name(r, spec)
    known = r is not None
    row = next((t for t in index_rows(spec) if t["path"] == ref), None) if known else None
    status = 200
    if known and not row and re.fullmatch(r"(?:[^/]+/){1,2}\d{1,10}", ref):
        try:
            row = seo_tasks.row(spec, ref)
        except seo_tasks.Unavailable as exc:
            status = exc.status
    elif known and not row:
        status = 404
    split, _, n = ref.rpartition("/")
    # a row of a dataset read as rows (no index here): "Row 3 (train)" rather than a bare "3"
    fallback = f"Row {n} ({split})" if split and n.isdigit() and "/" not in split else ref.rsplit("/", 1)[-1]
    title = clip((row or {}).get("title") or fallback or env_name, 110)
    what = clip((row or {}).get("brief") or "", 260)
    desc = clip(f"{title}: a task in {env_name} ({kind_of(r, spec)}). {what or 'What the agent is asked, how it is graded, what it runs in, and its files.'}", 300)
    path = f"/t/{enc(spec)}/{enc(ref)}"
    cr, cld = crumbs(base, [("Environments", "/"), (env_name, f"/d/{enc(spec)}"), (title, "")])
    body = (cr + f"<header class=\"tp-head\"><h1>{esc(title)}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + (f'<section><h2>The task</h2><p class="seo-prompt">{esc(row.get("brief") or "")}</p></section>' if row else "")
            + f'<p>Part of <a href="/d/{enc(spec)}">{esc(spec)}</a>.</p>')
    if row and type(row.get("total")) is int and n.isdigit():
        body += '<nav aria-label="Other tasks">' + " · ".join(
            f'<a href="/t/{enc(spec)}/{enc(split)}/{i}">{label}</a>'
            for i, label in ((int(n) - 1, "Previous task"), (int(n) + 1, "Next task")) if 0 <= i < row["total"]) + '</nav>'
    ld = [{"@type": "CreativeWork", "name": title, "description": desc, "url": base + path, "learningResourceType": "RL environment task",
           "isPartOf": {"@type": "Dataset", "name": env_name, "url": f"{base}/d/{enc(spec)}", "sameAs": f"https://huggingface.co/datasets/{spec}"},
           "about": "reinforcement learning", **({"genre": row["category"]} if row and row.get("category") else {})}, cld]
    return page(request, title=f"{title} · {env_name}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if known else None,
                noindex=not row and status != 503, og_type="article", status=status)


def space(request: Request, spec: str) -> HTMLResponse:
    base = base_url(request)
    by, _ = listing()
    r = next((r for r in public_rows(list(by.values())) if r["key"] == f"space:{spec}"), None)
    if "task" in request.query_params:
        return space_task(request, spec, r)
    name = display_name(r, spec)
    kind = kind_of(r, spec) if r else "environment Space"
    desc = clip(f"{name}: an {kind} on Hugging Face. {clip((r or {}).get('brief') or '', 220) or 'See it live: its app, a playground with rewards, its tasks, and MCP for coding agents.'}", 300)
    path = f"/s/{enc(spec)}"
    cr, cld = crumbs(base, [("Environments", "/"), ("Spaces", "/?k=openenv"), (name, "")])
    body = (cr + f"<header class=\"tp-head\"><h1>{esc(spec)}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + f'<p><a href="https://huggingface.co/spaces/{enc(spec)}" rel="noopener">{esc(spec)} on the Hugging Face Hub</a></p>')
    ld = [{"@type": "SoftwareApplication", "name": name, "alternateName": spec, "description": desc, "url": base + path,
           "sameAs": f"https://huggingface.co/spaces/{spec}", "applicationCategory": "DeveloperApplication", "operatingSystem": "Web",
           "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
           "author": {"@type": "Organization", "name": spec.split("/")[0], "url": f"https://huggingface.co/{spec.split('/')[0]}"},
           "keywords": ["reinforcement learning", "RL environment", *(["OpenEnv"] if (r or {}).get("openenv") else []), *((r or {}).get("tags") or [])[:10]]}, cld]
    if r and discoverable(r):
        ranges = seo_tasks.ranges((spaces_live.last_seen(spec) or {}).get("task_api"))
        body += "".join(f'<section><h2>{esc(split)} tasks</h2><ul>' + "".join(
            f'<li><a href="{esc(seo_tasks.space_path(spec, env, split, i))}">Task {i + 1}</a></li>' for i in range(min(n, 12)))
            + "</ul></section>" for env, split, n in ranges)
    return page(request, title=f"{name} · {kind}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if r else None, noindex=not r or not discoverable(r))


def space_task(request: Request, spec: str, environment: dict | None) -> HTMLResponse:
    q = request.query_params
    env, split, raw = q.get("env", ""), q.get("split", ""), q.get("task", "")
    if not environment or not re.fullmatch(r"\d{1,10}", raw) or not env or not split or len(env) > 80 or len(split) > 200:
        return not_found(request)
    index = int(raw)
    path = seo_tasks.space_path(spec, env, split, index)
    status, row = 200, None
    try:
        row = seo_tasks.space(spec, env, split, index)
    except seo_tasks.Unavailable as exc:
        status = exc.status
    except Exception:
        status = 503
    if status == 404:
        return not_found(request)
    title = clip((row or {}).get("title") or f"Task {index + 1}", 110)
    desc = clip(f"{title}: {split} task in {spec}, an RL environment on the Hugging Face Hub. {(row or {}).get('brief') or ''}", 300)
    cr, cld = crumbs(base_url(request), [("Environments", "/"), (spec, f"/s/{enc(spec)}"), (title, "")])
    body = cr + f'<header class="tp-head"><h1>{esc(title)}</h1><p class="lede">{esc(desc)}</p></header>'
    if row:
        body += f'<section><h2>The task</h2><p class="seo-prompt">{esc(row["brief"])}</p></section>'
        if row.get("fields"):
            body += '<h2>Task details</h2><dl>' + "".join(
                f'<dt>{esc(k.replace("_", " "))}</dt><dd>{esc(str(v)[:1000])}</dd>' for k, v in row["fields"].items()) + '</dl>'
        body += '<nav aria-label="Other tasks">' + " · ".join(
            f'<a href="{esc(seo_tasks.space_path(spec, env, split, i))}">{label}</a>'
            for i, label in ((index - 1, "Previous task"), (index + 1, "Next task")) if 0 <= i < row["total"]) + '</nav>'
    else:
        body += '<p>The task server is temporarily unavailable. Please try again shortly.</p>'
    ld = [{"@type": "CreativeWork", "name": title, "description": desc, "url": base_url(request) + path,
           "isPartOf": {"@type": "SoftwareApplication", "name": spec, "url": base_url(request) + f"/s/{enc(spec)}"}}, cld]
    return page(request, title=f"{title} · {spec}", description=desc, path=path, body=body, jsonld=ld,
                image=f"/og/s/{enc(spec)}.png?" + urlencode({"env": env, "split": split, "task": index}),
                og_type="article", status=status)


def simple(request: Request, path: str) -> HTMLResponse:
    """The app's other pages: community (indexed), a rollout, your rollouts, a comparison (not indexed: they change, or are personal)."""
    if path.startswith("/community"):
        return page(request, title="Community rollouts", path="/community",
                    description="Public, graded RL rollouts across Harbor, MiMo, NeMo Gym and other supported runners. Explore traces and model results grouped by environment and scoring metric.",
                    body="<header class=\"tp-head\"><h1>Community rollouts</h1><p class=\"lede\">Public rollouts from everyone, shown without who ran them.</p></header>")
    titles = {"/runs": "My rollouts"}
    return page(request, title=titles.get(path, "Rollout" if path.startswith("/run/") else "Compare rollouts"), path=path,
                description=DESCRIPTION, noindex=True)


def not_found(request: Request) -> HTMLResponse:
    return page(request, title="Page not found", path=request.url.path, description=DESCRIPTION, noindex=True, status=404,
                body='<header class="tp-head"><h1>Page not found</h1><p class="lede">There is nothing at this address. <a href="/">Explore RL environments</a>.</p></header>')


# ── robots and the sitemap ───────────────────────────────────────────────────
def robots(request: Request) -> PlainTextResponse:
    base = base_url(request)
    return PlainTextResponse("\n".join([
        "User-agent: *", "Allow: /", "Disallow: /api/", "Disallow: /mcp/", "Disallow: /capture/", "Disallow: /run/", "Disallow: /runs",
        "Allow: /api/env/", "Allow: /api/spaces/", "Allow: /api/search", "Allow: /api/environments",
        "Disallow: /api/environments/mine", "Disallow: /compare/", "Disallow: /login", "Disallow: /logout", "", f"Sitemap: {base}/sitemap.xml", ""]))


_sitemap: dict[str, Any] = {"at": 0.0, "pages": [], "tasks": []}


def _entries() -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
    with _lock:
        if time.time() - _sitemap["at"] < 300 and _sitemap["pages"]:
            return _sitemap["pages"], _sitemap["tasks"]
    by, rows = listing()
    rows = public_rows(rows)
    checks = space_checks.inventory()
    pages: list[tuple[str, str | None]] = [("/", None), ("/community", None)]
    # Every public dataset and only Spaces with evidence of a supported API.
    pages += [(_href(r), (r.get("updated") or "")[:10] or None) for r in rows
              if discoverable(r, checks)]
    tasks: list[tuple[str, str | None]] = []
    specs = list(dict.fromkeys(r["id"] for r in rows if r["kind"] == "dataset"))
    covered = set()
    try:
        # The snapshot already holds public task refs. One read avoids hundreds
        # of remote bucket stat/open operations during a crawler's first visit.
        with snapshot.use() as (_, conn):
            refs = conn.execute("SELECT env, ref FROM tasks WHERE env IN (SELECT value FROM json_each(?)) ORDER BY env, ref",
                                (json.dumps(specs),)).fetchall()
        for r in refs:
            covered.add(r["env"])
            tasks.append((f"/t/{enc(r['env'])}/{enc(r['ref'])}", None))
    except snapshot.SnapshotError:
        pass
    for spec in dict.fromkeys(r["id"] for r in rows if r["kind"] == "dataset" and r["id"] not in covered
                             and (r["id"] == MIMO or (r.get("indexed") or {}).get("tasks"))):
        try:
            tasks += [(f"/t/{enc(spec)}/{enc(t['path'])}", None) for t in index_rows(spec)]
        except Exception:  # noqa: BLE001 - one index unreadable leaves the rest
            continue
    with _lock:
        tasks = list(dict.fromkeys(tasks))
        _sitemap.update(at=time.time(), pages=pages, tasks=tasks)
    return pages, tasks


def _space_ranges():
    """Represent task coordinates as ranges; do not allocate millions of URLs or fetch any tasks."""
    records = space_checks.inventory()
    _, rows = listing()
    out = []
    for r in sorted(public_rows(rows), key=lambda r: r["key"]):
        if r.get("kind") != "space" or not space_checks.browseable(records.get(r["id"])):
            continue
        rec = records[r["id"]]
        if not (rec.get("task_catalog") or {}).get("tasks") and not rec.get("task_splits"):
            continue
        ranges = rec.get("task_splits") or seo_tasks.ranges((spaces_live.last_seen(r["id"]) or {}).get("task_api"))
        for env, split, n in ranges:
            if type(n) is int and 0 < n <= 1_000_000_000:
                out.append((r["id"], env, split, n))
    return out


def _urlset(base: str, items: list[tuple[str, str | None]]) -> Response:
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    xml += [f"<url><loc>{esc(base + p)}</loc>{f'<lastmod>{esc(m)}</lastmod>' if m else ''}</url>" for p, m in items if len((base + p).encode()) <= 2048]
    xml.append("</urlset>")
    return Response("\n".join(xml), media_type="application/xml")


def sitemap_index(request: Request) -> Response:
    base = base_url(request)
    _, tasks = _entries()
    parts = ["/sitemap-pages.xml", *[f"/sitemap-tasks-{i + 1}.xml" for i in range((len(tasks) + SITEMAP_CHUNK - 1) // SITEMAP_CHUNK)]]
    total = sum(r[3] for r in _space_ranges())
    parts += [f"/sitemap-space-tasks-{i + 1}.xml" for i in range(min(49000, (total + SITEMAP_CHUNK - 1) // SITEMAP_CHUNK))]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    xml += [f"<sitemap><loc>{esc(base + p)}</loc></sitemap>" for p in parts]
    xml.append("</sitemapindex>")
    return Response("\n".join(xml), media_type="application/xml")


def sitemap_pages(request: Request) -> Response:
    pages, _ = _entries()
    return _urlset(base_url(request), pages)


def sitemap_tasks(request: Request, n: int) -> Response:
    _, tasks = _entries()
    chunk = tasks[(n - 1) * SITEMAP_CHUNK: n * SITEMAP_CHUNK]
    if n < 1 or not chunk:
        return Response("not found", status_code=404)
    return _urlset(base_url(request), chunk)


def sitemap_space_tasks(request: Request, n: int) -> Response:
    if not 1 <= n <= 49000:
        return Response("not found", status_code=404)
    offset, remaining, urls = (n - 1) * SITEMAP_CHUNK, SITEMAP_CHUNK, []
    for spec, env, split, count in _space_ranges():
        if offset >= count:
            offset -= count
            continue
        stop = min(count, offset + remaining)
        urls.extend((seo_tasks.space_path(spec, env, split, i), None) for i in range(offset, stop))
        remaining -= stop - offset
        offset = 0
        if not remaining:
            break
    return _urlset(base_url(request), urls) if urls else Response("not found", status_code=404)


# ── preview images ───────────────────────────────────────────────────────────
# 1200×630, plain: the site's name, what the page is, its title and a line of facts. Drawn once per page and kept.
W, H = 1200, 630


@lru_cache(maxsize=512)
def card_png(kind: str, title: str, sub: str, facts: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), "#fffdf7")
    d = ImageDraw.Draw(img)
    font = lambda s: ImageFont.load_default(size=s)  # noqa: E731 - Pillow's own font, so no system fonts are needed
    d.rectangle([0, 0, W, 12], fill="#ffcd36")
    d.rounded_rectangle([56, 48, 198, 90], radius=10, fill="#ffdc62")
    d.text((73, 58), "FINEENVS", font=font(23), fill="#352c0e")
    d.text((220, 57), SITE, font=font(26), fill="#54504a")
    d.text((56, 142), _ellipsize(d, kind.upper(), font(23), 1088), font=font(23), fill="#827256")
    lines = _wrap(d, title, font(65), 1088)
    size = 65 if len(lines) <= 3 else 54
    lines = _wrap(d, title, font(size), 1088)[:3]
    for i, line in enumerate(lines):
        d.text((56, 194 + i * 77), _ellipsize(d, line, font(size), 1088), font=font(size), fill="#1c1b19")
    if sub:
        d.text((56, 452), _ellipsize(d, sub, font(27), 1088), font=font(27), fill="#665e51")
    d.line([56, 516, 1144, 516], fill="#e5dfd1", width=2)
    d.text((56, 550), _ellipsize(d, facts or "Discover tasks. Inspect rewards. Run agents.", font(24), 680), font=font(24), fill="#524d42")
    d.text((1144, 550), "Hugging Face Hub", font=font(24), fill="#827256", anchor="ra")
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def _ellipsize(d, text, font, width):
    text = str(text)
    if d.textlength(text, font=font) <= width:
        return text
    while text and d.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _wrap(d, text: str, font, width: int) -> list[str]:
    if "\n" in text:
        return [line for part in text.splitlines() for line in _wrap(d, part, font, width)]
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        nxt = f"{cur} {w}".strip()
        if d.textlength(nxt, font=font) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > 3:
        lines = lines[:3]
        lines[2] = lines[2].rstrip(".,;:") + "…"
    return lines or [""]


def og_image(path: str, query=None) -> Response:
    """The preview image of a page, from its path (`/d/org/name`, `/t/org/name/ref`, `/s/org/name`, or the site's)."""
    if path == "/":
        return Response(card_png("Reinforcement learning", "Explore RL environments\non the Hugging Face Hub",
                                 "OpenEnv  ·  Harbor  ·  MiMo  ·  NeMo Gym  ·  Verifiers", ""),
                        media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    by, _ = listing()
    by = {r["key"]: r for r in public_rows(list(by.values()))}
    parts = [p for p in path.strip("/").split("/") if p]
    if parts and not (len(parts) >= 3 and parts[0] in ("d", "s", "t")):
        return Response(status_code=404)
    kind, title, sub, facts = "Reinforcement learning environments", TAGLINE, "Across OpenEnv, Harbor, Verifiers, NeMo Gym and more", ""
    if len(parts) >= 3 and parts[0] in ("d", "s", "t"):
        spec = f"{parts[1]}/{parts[2]}"
        r = by.get(spec if parts[0] != "s" else f"space:{spec}")
        if r is None:
            return Response(status_code=404)
        name = display_name(r, spec)
        k = kind_of(r, spec)
        if parts[0] == "t" and (r or spec == MIMO):
            ref = "/".join(parts[3:])
            row = next((t for t in index_rows(spec) if t["path"] == ref), None)
            if not row and re.fullmatch(r"(?:[^/]+/){1,2}\d{1,10}", ref):
                try:
                    row = seo_tasks.row(spec, ref)
                except seo_tasks.Unavailable as exc:
                    return Response(status_code=exc.status)
            if not row:
                return Response(status_code=404)
            kind, title, sub = f"A task in {name}", clip((row or {}).get("title") or ref, 140), spec
        elif parts[0] == "s" and query and "task" in query:
            raw, env, split = query.get("task", ""), query.get("env", ""), query.get("split", "")
            if not re.fullmatch(r"\d{1,10}", raw) or not env or not split or len(env) > 80 or len(split) > 200:
                return Response(status_code=404)
            try:
                row = seo_tasks.space(spec, env, split, int(raw))
            except Exception:
                return Response(status_code=503, headers={"Retry-After": "60"})
            kind, title, sub = f"{split} task", clip(row["title"], 140), spec
        else:
            n = ((r or {}).get("indexed") or {}).get("tasks") or (len(index_rows(spec)) if spec == MIMO else None)
            kind, title, sub = k, name, spec
            facts = " · ".join(x for x in [f"{n:,} tasks" if n else "", f"{(r or {}).get('downloads', 0):,} downloads" if r and r.get("downloads") else "",
                                            f"{r['likes']:,} likes" if r and r.get("likes") else ""] if x)
    return Response(card_png(kind, title, sub, facts), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
