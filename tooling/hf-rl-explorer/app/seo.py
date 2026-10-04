"""Search engines and link previews. Every page of the app has its own path, and the server answers each with the app
plus, in its head, the page's own title, description, canonical address, Open Graph and Twitter card, and schema.org
data (a Dataset for an environment, a SoftwareApplication for a Space, a task as part of its environment, breadcrumbs);
in its body, the page's content as plain HTML, for crawlers that don't run JavaScript (the app replaces it as it
starts). Plus robots.txt, a sitemap of every environment and every indexed task, and a preview image per page.

Cheap and safe by design: only what is already known is used (the catalog's listing of public environments, indexes
already built, the MiMo release's local index). A crawler never starts indexing a dataset, wakes a Space, or reads
anything private; task text comes from the same withheld views the pages show.
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
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from . import catalog, config

SITE = "HF RL Explorer"
TAGLINE = "Explore RL environments on Hugging Face"
DESCRIPTION = ("Explore reinforcement learning environments on Hugging Face across OpenEnv, Harbor, Verifiers, NeMo Gym and "
               "more: what each task asks, how it's graded, what it runs in, and run an agent on it.")
KIND = {"harbor": "Harbor dataset", "verifiers": "Verifiers environment", "nemo-gym": "NeMo Gym dataset", "rows": "RL dataset",
        "mimo": "MiMo RL release", "openenv": "OpenEnv Space", "space": "environment Space"}
MIMO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
NOINDEX = re.compile(r"^/(run/|runs$|compare/)")
SITEMAP_CHUNK = 40_000


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


def index_rows(spec: str) -> list[dict[str, Any]]:
    """An environment's tasks, if they are already known here (never built for a crawler): [{path, title, brief, category}]."""
    if spec == MIMO:
        return _mimo_rows()
    try:
        p = catalog._index_path(spec)
        mtime = p.stat().st_mtime
    except (OSError, ValueError):
        return []
    return _harbor_rows(spec, mtime)


@lru_cache(maxsize=1)
def _mimo_rows() -> list[dict[str, Any]]:
    from .mimo import catalog as mcat

    return [{"path": e["id"], "title": e["t"], "brief": e["s"], "category": e["d"]} for e in mcat.index()["envs"]]


@lru_cache(maxsize=32)
def _harbor_rows(spec: str, mtime: float) -> list[dict[str, Any]]:
    idx = catalog._read_index(spec)
    return [{"path": t["path"], "title": t.get("title"), "brief": t.get("brief"), "category": t.get("category")} for t in (idx or {}).get("tasks") or []]


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
        return {"openenv": KIND["openenv"], "ors": "ORS Space"}.get(row.get("framework") or ("openenv" if row.get("openenv") else ""), KIND["space"])
    return KIND.get(row.get("framework") or "", "RL dataset")


# ── a page ───────────────────────────────────────────────────────────────────
def page(request: Request, *, title: str, description: str, path: str, body: str = "", jsonld: list[dict[str, Any]] | None = None,
         image: str | None = None, noindex: bool = False, og_type: str = "website", status: int = 200) -> HTMLResponse:
    base = base_url(request)
    url = base + path
    full = title if SITE in title else f"{title} · {SITE}"
    img = base + (image or "/og.png")
    head = "\n".join([
        f'<link rel="canonical" href="{esc(url)}">',
        '<meta name="robots" content="noindex, follow">' if noindex else '<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">',
        f'<meta property="og:site_name" content="{SITE}">',
        f'<meta property="og:type" content="{og_type}">',
        f'<meta property="og:title" content="{esc(full)}">',
        f'<meta property="og:description" content="{esc(description)}">',
        f'<meta property="og:url" content="{esc(url)}">',
        f'<meta property="og:image" content="{esc(img)}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{esc(full)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(full)}">',
        f'<meta name="twitter:description" content="{esc(description)}">',
        f'<meta name="twitter:image" content="{esc(img)}">',
        *[f'<script type="application/ld+json">{_ld(x)}</script>' for x in jsonld or []],
    ])
    doc = template()
    doc = re.sub(r"<title>.*?</title>", f"<title>{esc(full)}</title>", doc, count=1, flags=re.S)
    doc = re.sub(r'<meta name="description" content="[^"]*">', f'<meta name="description" content="{esc(description)}">', doc, count=1)
    doc = doc.replace("</head>", f"{head}\n</head>", 1)
    if body:   # the page as plain HTML until the app starts (crawlers that don't run JavaScript read this)
        doc = re.sub(r'<main id="view">(.*?)</main>', lambda m: f'<main id="view">{m.group(1)}<div class="wrap page ssr">{body}</div></main>', doc, count=1, flags=re.S)
    return HTMLResponse(doc, status_code=status, headers={"Cache-Control": "no-cache"})


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
    top = sorted(rows, key=lambda r: (r.get("trending") or 0) * 1e9 + (r.get("downloads") or 0), reverse=True)[:120]
    ds = sum(1 for r in rows if r["kind"] == "dataset")
    sp = sum(1 for r in rows if r["kind"] == "space")
    tasks = sum((r.get("indexed") or {}).get("tasks") or 0 for r in rows)
    desc = (f"{ds:,} RL environment datasets and {sp:,} environment Spaces on Hugging Face, across OpenEnv, Harbor, Verifiers, "
            f"NeMo Gym and more: see what each task asks and how it's graded, then run an agent on it." if rows else DESCRIPTION)
    body = (f"<header class=\"tp-head\"><h1>{TAGLINE}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + _env_list(top) + '<p><a href="/community">Community rollouts</a> · <a href="/d/XiaomiMiMo/MiMo-V2.6-RL-oss">MiMo-V2.6 RL</a></p>')
    ld = [{"@type": "WebSite", "name": SITE, "alternateName": TAGLINE, "url": base + "/", "description": desc,
           "potentialAction": {"@type": "SearchAction", "target": {"@type": "EntryPoint", "urlTemplate": base + "/?q={search_term_string}"},
                               "query-input": "required name=search_term_string"},
           "publisher": {"@type": "Organization", "name": "Hugging Face", "url": "https://huggingface.co"}},
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
    r = by.get(spec)
    name = display_name(r, spec)
    kind = kind_of(r, spec)
    tasks = index_rows(spec) if r or spec == MIMO else []
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
    return page(request, title=f"{name} · {kind}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if (r or spec == MIMO) else None, noindex=not r and spec != MIMO)


def task(request: Request, spec: str, ref: str) -> HTMLResponse:
    base = base_url(request)
    by, _ = listing()
    r = by.get(spec)
    env_name = display_name(r, spec)
    known = r is not None or spec == MIMO
    row = next((t for t in index_rows(spec) if t["path"] == ref), None) if known else None
    split, _, n = ref.rpartition("/")
    # a row of a dataset read as rows (no index here): "Row 3 (train)" rather than a bare "3"
    fallback = f"Row {n} ({split})" if split and n.isdigit() and "/" not in split else ref.rsplit("/", 1)[-1]
    title = clip((row or {}).get("title") or fallback or env_name, 110)
    what = clip((row or {}).get("brief") or "", 260)
    desc = clip(f"{title}: a task in {env_name} ({kind_of(r, spec)}). {what or 'What the agent is asked, how it is graded, what it runs in, and its files.'}", 300)
    path = f"/t/{enc(spec)}/{enc(ref)}"
    cr, cld = crumbs(base, [("Environments", "/"), (env_name, f"/d/{enc(spec)}"), (title, "")])
    body = (cr + f"<header class=\"tp-head\"><h1>{esc(title)}</h1><p class=\"lede\">{esc(desc)}</p></header>"
            + f'<p>Part of <a href="/d/{enc(spec)}">{esc(spec)}</a>.</p>')
    ld = [{"@type": "CreativeWork", "name": title, "description": desc, "url": base + path, "learningResourceType": "RL environment task",
           "isPartOf": {"@type": "Dataset", "name": env_name, "url": f"{base}/d/{enc(spec)}", "sameAs": f"https://huggingface.co/datasets/{spec}"},
           "about": "reinforcement learning", **({"genre": row["category"]} if row and row.get("category") else {})}, cld]
    return page(request, title=f"{title} · {env_name}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if known else None,
                noindex=not row, og_type="article")


def space(request: Request, spec: str) -> HTMLResponse:
    base = base_url(request)
    by, _ = listing()
    r = by.get(f"space:{spec}")
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
           "keywords": ["reinforcement learning", "RL environment", "OpenEnv", *((r or {}).get("tags") or [])[:10]]}, cld]
    return page(request, title=f"{name} · {kind}", description=desc, path=path, body=body, jsonld=ld, image=f"/og{path}.png" if r else None, noindex=not r)


def simple(request: Request, path: str) -> HTMLResponse:
    """The app's other pages: community (indexed), a rollout, your rollouts, a comparison (not indexed: they change, or are personal)."""
    if path.startswith("/community"):
        return page(request, title="Community rollouts", path="/community",
                    description="Public rollouts of RL environments on Hugging Face, by every model: a leaderboard by environment, the traces, and the tasks nobody has tried yet.",
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
        "Disallow: /compare/", "Disallow: /login", "Disallow: /logout", "", f"Sitemap: {base}/sitemap.xml", ""]))


_sitemap: dict[str, Any] = {"at": 0.0, "pages": [], "tasks": []}


def _entries() -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
    with _lock:
        if time.time() - _sitemap["at"] < 3600 and _sitemap["pages"]:
            return _sitemap["pages"], _sitemap["tasks"]
    by, rows = listing()
    pages: list[tuple[str, str | None]] = [("/", None), ("/community", None)]
    # every dataset; a Space when someone liked it, it runs, or it's featured (thousands are near-copies from hackathons)
    pages += [(_href(r), (r.get("updated") or "")[:10] or None) for r in rows
              if r["kind"] == "dataset" or (r.get("likes") or 0) > 0 or r.get("stage") == "RUNNING" or r.get("collection")]
    if f"{MIMO}" not in by:
        pages.append((f"/d/{enc(MIMO)}", None))
    tasks: list[tuple[str, str | None]] = []
    for spec in [MIMO, *[r["id"] for r in rows if r["kind"] == "dataset" and (r.get("indexed") or {}).get("tasks")]]:
        try:
            tasks += [(f"/t/{enc(spec)}/{enc(t['path'])}", None) for t in index_rows(spec)]
        except Exception:  # noqa: BLE001 - one index unreadable leaves the rest
            continue
    with _lock:
        _sitemap.update(at=time.time(), pages=pages, tasks=tasks)
    return pages, tasks


def _urlset(base: str, items: list[tuple[str, str | None]]) -> Response:
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    xml += [f"<url><loc>{esc(base + p)}</loc>{f'<lastmod>{esc(m)}</lastmod>' if m else ''}</url>" for p, m in items]
    xml.append("</urlset>")
    return Response("\n".join(xml), media_type="application/xml")


def sitemap_index(request: Request) -> Response:
    base = base_url(request)
    _, tasks = _entries()
    parts = ["/sitemap-pages.xml", *[f"/sitemap-tasks-{i + 1}.xml" for i in range((len(tasks) + SITEMAP_CHUNK - 1) // SITEMAP_CHUNK)]]
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


# ── preview images ───────────────────────────────────────────────────────────
# 1200×630, plain: the site's name, what the page is, its title and a line of facts. Drawn once per page and kept.
W, H = 1200, 630


@lru_cache(maxsize=512)
def card_png(kind: str, title: str, sub: str, facts: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    font = lambda s: ImageFont.load_default(size=s)  # noqa: E731 - Pillow's own font, so no system fonts are needed
    d.rectangle([0, 0, W, 6], fill="#111827")
    d.text((72, 70), SITE, font=font(34), fill="#6b7280")
    d.text((72, 150), kind.upper(), font=font(28), fill="#9ca3af")
    y = 200
    for line in _wrap(d, title, font(66), W - 144)[:3]:
        d.text((72, y), line, font=font(66), fill="#111827")
        y += 82
    if sub:
        d.text((72, y + 16), clip(sub, 70), font=font(32), fill="#4b5563")
    if facts:
        d.text((72, H - 104), facts, font=font(30), fill="#374151")
    d.text((W - 72, H - 104), "RL environments on Hugging Face", font=font(28), fill="#9ca3af", anchor="ra")
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def _wrap(d, text: str, font, width: int) -> list[str]:
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


def og_image(path: str) -> Response:
    """The preview image of a page, from its path (`/d/org/name`, `/t/org/name/ref`, `/s/org/name`, or the site's)."""
    by, _ = listing()
    parts = [p for p in path.strip("/").split("/") if p]
    if parts and not (len(parts) >= 3 and parts[0] in ("d", "s", "t")):
        return Response(status_code=404)
    kind, title, sub, facts = "Reinforcement learning environments", TAGLINE, "Across OpenEnv, Harbor, Verifiers, NeMo Gym and more", ""
    if len(parts) >= 3 and parts[0] in ("d", "s", "t"):
        spec = f"{parts[1]}/{parts[2]}"
        r = by.get(spec if parts[0] != "s" else f"space:{spec}")
        if r is None and spec != MIMO:   # nothing listed there: no image (its page points at the site's), no work done
            return Response(status_code=404)
        name = display_name(r, spec)
        k = kind_of(r, spec)
        if parts[0] == "t" and (r or spec == MIMO):
            ref = "/".join(parts[3:])
            row = next((t for t in index_rows(spec) if t["path"] == ref), None)
            kind, title, sub = f"A task in {name}", clip((row or {}).get("title") or ref, 140), spec
        else:
            n = ((r or {}).get("indexed") or {}).get("tasks") or (len(index_rows(spec)) if spec == MIMO else None)
            kind, title, sub = k, name, spec
            facts = " · ".join(x for x in [f"{n:,} tasks" if n else "", f"{(r or {}).get('downloads', 0):,} downloads" if r and r.get("downloads") else "",
                                            f"{r['likes']:,} likes" if r and r.get("likes") else ""] if x)
    return Response(card_png(kind, title, sub, facts), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
