"""Live environment Spaces: what a running OpenEnv server offers, waking or restarting it, its Task API, and a
playground session on it (reset/step over its WebSocket, or its MCP tools).

Everything here talks to one host only: the Space's own `https://<subdomain>.hf.space`, taken from the Hub's record
of a public Space, never from the visitor. Nothing a visitor sends is passed on as a URL.

What a server offers comes from its own `/openapi.json` (every OpenEnv server is FastAPI), its `/metadata` and
`/schema`, `/list_environments` (the Task API), and `tools/list` on `/mcp`. Its web UI is found by asking for the
`base_path` its README declares, then `/web`, then `/`.

The last good probe of each Space is kept (STORAGE_DIR/space-probes), so a Space that has gone to sleep still shows
what it offered when it was last seen running, and how it did against `openenv validate`'s six runtime criteria.
"""

from __future__ import annotations

import base64
import json
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from . import catalog, config

HOST = re.compile(r"^https://[a-z0-9-]+\.hf\.space$")
UP = "RUNNING"
DOWN = {"SLEEPING", "PAUSED", "STOPPED", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING"}
STARTING = {"APP_STARTING", "BUILDING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"}
# Task fields that may hold the answer: left out of the task browser (a playground's terminal observation is the
# environment's own feedback, so it is shown as the environment sends it)
ANSWER = re.compile(r"(gold|answer|solution|expected|reference|ground_?truth|oracle|^target(?!_?lang)|_target$|^true_|correct|^label$|^labels$)", re.I)
STANDARD_ROUTES = {"/reset", "/step", "/state", "/metadata", "/health", "/healthz", "/schema", "/mcp", "/list_environments",
                   "/{env_name}/splits", "/{env_name}/tasks", "/{env_name}/num_tasks", "/{env_name}/task", "/{env_name}/task_range",
                   "/web/metadata", "/web/reset", "/web/step", "/web/state", "/docs", "/redoc", "/openapi.json"}
MAX_REPLY = 12 * 2**20          # bytes from a Space for one reply (observations can carry images)
_client = httpx.Client(timeout=httpx.Timeout(12, connect=6), follow_redirects=False,
                       headers={"User-Agent": "hf-rl-explorer (+https://huggingface.co/FineEnvs)"},
                       limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
_lock = threading.Lock()


class SpaceError(Exception):
    """A problem worth showing as it is (the Space is asleep, a call failed): becomes a 4xx/5xx with this text."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


# ── the Space's record ───────────────────────────────────────────────────────
def record(spec: str, fresh: bool = False) -> dict[str, Any]:
    """Host, stage and declared UI path of a public Space (cached a few seconds: the page polls it while it wakes)."""
    spec = catalog.check_spec(spec)

    def fetch():
        sp = catalog._api().space_info(spec, expand=["runtime", "subdomain", "cardData", "private", "sdk", "tags"])
        if sp.private:
            raise PermissionError("this Space is private")
        card = (sp.card_data.to_dict() if sp.card_data else {}) or {}
        host = f"https://{sp.subdomain}.hf.space" if getattr(sp, "subdomain", None) else ""
        stage = str(getattr(sp.runtime, "stage", "") or "") if sp.runtime is not None else ""
        base = card.get("base_path") if isinstance(card.get("base_path"), str) else None
        return {"id": spec, "host": host if HOST.match(host) else "", "stage": stage or "UNKNOWN", "sdk": sp.sdk,
                "base_path": base if base and re.match(r"^/(?!/)[\w\-./]{0,80}$", base) and ".." not in base else None,
                "hardware": getattr(sp.runtime, "hardware", None) if sp.runtime is not None else None,
                "tags": list(sp.tags or [])}

    if fresh:
        catalog._memo.pop(("space-record", spec), None)
    return catalog._cached(("space-record", spec), 4, fetch)


def _url(rec: dict, path: str) -> str:
    if not rec["host"]:
        raise SpaceError("this Space has no app address", 404)
    return rec["host"] + path


def _json(r: httpx.Response) -> Any:
    if len(r.content) > MAX_REPLY:
        raise SpaceError("the Space's reply is too large to show", 502)
    body = r.text
    if "text/event-stream" in r.headers.get("content-type", ""):   # a single SSE message: its data line is the reply
        body = next((ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")), "null")
    return json.loads(body)


# ── what a running server offers ─────────────────────────────────────────────
def _get_here(rec: dict, path: str, hops: int = 3) -> httpx.Response | None:
    """GET a path on the Space, following redirects only while they stay on its own host (never to another address)."""
    url = _url(rec, path)
    for _ in range(hops + 1):
        r = _client.get(url, timeout=8)
        if r.status_code not in (301, 302, 303, 307, 308):
            return r
        nxt = r.url.join(r.headers.get("location", ""))
        if nxt.scheme != "https" or nxt.host != httpx.URL(rec["host"]).host:
            return None
        url = str(nxt)
    return None


def _ui_path(rec: dict) -> str | None:
    """The path that serves a page meant for people: the README's base_path, OpenEnv's /web, or the root."""
    for cand in [rec.get("base_path"), "/web", "/"]:
        if not cand:
            continue
        try:
            r = _get_here(rec, cand)
        except Exception:  # noqa: BLE001 - slow or down: try the next
            continue
        if (r is not None and r.status_code == 200 and "text/html" in r.headers.get("content-type", "")
                and not r.url.path.rstrip("/").endswith(("/docs", "/redoc"))):
            return r.url.path or "/"
    return None


def _get_json(rec: dict, path: str) -> Any:
    try:
        r = _client.get(_url(rec, path))
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            return _json(r)
    except Exception:  # noqa: BLE001
        pass
    return None


def mcp_call(rec: dict, method: str, params: dict | None = None, timeout: float = 30, raw: bool = False) -> Any:
    """One JSON-RPC call to the Space's /mcp. Returns `result`; raises SpaceError with the server's message on error.
    `raw`: the whole JSON-RPC reply, an error reply included (HTTP failures still raise)."""
    try:
        r = _client.post(_url(rec, "/mcp"), timeout=httpx.Timeout(timeout, connect=6),
                         headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
                         json={"jsonrpc": "2.0", "id": secrets.randbelow(10**9), "method": method, "params": params or {}})
    except httpx.TimeoutException:
        raise SpaceError(f"the Space didn't answer within {int(timeout)} s", 504) from None
    except httpx.HTTPError as e:
        raise SpaceError(f"couldn't reach the Space: {type(e).__name__}", 502) from None
    if r.status_code != 200:
        raise SpaceError(f"the Space's /mcp answered HTTP {r.status_code}", 502)
    try:
        doc = _json(r)
    except ValueError:
        raise SpaceError("the Space's /mcp didn't answer JSON", 502) from None
    if not isinstance(doc, dict):
        raise SpaceError("the Space's /mcp answered something unexpected", 502)
    if raw:
        return doc
    if doc.get("error"):
        err = doc["error"] if isinstance(doc["error"], dict) else {"message": str(doc["error"])}
        raise SpaceError(str(err.get("message") or "error")[:500], 422)
    return doc.get("result")


def _tools(rec: dict) -> tuple[list[dict] | None, str | None, dict]:
    """Its MCP tools (or why there are none), and how /mcp answered: a JSON-RPC 2.0 reply, as `openenv validate`
    asks of it, or the failure."""
    try:
        doc = mcp_call(rec, "tools/list", timeout=10, raw=True)
    except SpaceError as e:
        return None, str(e), {"ok": False, "detail": str(e)}
    rpc = {"ok": doc.get("jsonrpc") == "2.0", "detail": "JSON-RPC 2.0" if doc.get("jsonrpc") == "2.0" else "not a JSON-RPC 2.0 reply"}
    if doc.get("error"):
        err = doc["error"] if isinstance(doc["error"], dict) else {"message": str(doc["error"])}
        return None, str(err.get("message") or "error")[:500], rpc
    res = doc.get("result")
    tools = (res or {}).get("tools") if isinstance(res, dict) else None
    if not isinstance(tools, list):
        return None, "no tools", rpc
    keep = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")
    return [{k: t.get(k) for k in keep if t.get(k) is not None} for t in tools[:120] if isinstance(t, dict) and t.get("name")], None, rpc


def _api_routes(api: Any, openenv: bool) -> list[dict[str, Any]]:
    """The server's own routes from its OpenAPI, for the playground and the MCP bridge, on servers that aren't OpenEnv
    (NeMo Gym, ORS, custom): an OpenEnv server is used through reset/step and its tools, and its other routes are its
    own plumbing (geoguesser's /geoguesser/task/{i} returns the answer's coordinates). GET and POST only."""
    if openenv or not isinstance(api, dict) or not isinstance(api.get("paths"), dict):
        return []
    comps = ((api.get("components") or {}).get("schemas") or {}) if isinstance(api.get("components"), dict) else {}

    def resolve(x: Any, depth: int = 0) -> Any:
        if not isinstance(x, (dict, list)):
            return x
        if depth > 12:   # deep or self-referencing schemas: cut, not looped
            return {} if isinstance(x, dict) else []
        if isinstance(x, dict):
            if "$ref" in x and isinstance(x["$ref"], str):
                return resolve(comps.get(x["$ref"].split("/")[-1], {}), depth + 1)
            return {k: resolve(v, depth + 1) for k, v in x.items() if k != "$defs"}
        if isinstance(x, list):
            return [resolve(v, depth + 1) for v in x[:50]]
        return x

    out = []
    for path, ops in api["paths"].items():
        # only plain paths on the Space itself: never "//host/…", a scheme, "..", backslashes or a query
        if (not isinstance(ops, dict) or not isinstance(path, str) or not re.match(r"^/(?!/)[\w\-.~{}/]{0,199}$", path)
                or ".." in path):
            continue
        if path in STANDARD_ROUTES or path.startswith(("/web", "/login", "/logout", "/docs", "/redoc", "/static", "/gradio", "/assets")):
            continue
        if openenv and path in ("/healthz",):
            continue
        for method in ("post", "get"):
            op = ops.get(method)
            if not isinstance(op, dict):
                continue
            body = (((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema")
            schema = resolve(body) if body else None
            out.append({"method": method.upper(), "path": path, "summary": str(op.get("summary") or op.get("description") or "")[:200],
                        "params": re.findall(r"\{([A-Za-z_][\w-]{0,40})\}", path),
                        "schema": schema if isinstance(schema, dict) and schema.get("type") == "object" else None})
        if len(out) >= 40:
            break
    return out


def route_url(info: dict, method: str, path: str, params: dict[str, Any]) -> str:
    """A route the server published, its path parameters filled in safely; anything else is refused."""
    route = next((r for r in info.get("api") or [] if r["method"] == method and r["path"] == path), None)
    if route is None:
        raise SpaceError("that route isn't one this server publishes", 400)
    filled = path
    for name in route["params"]:
        v = str((params or {}).get(name, ""))
        if not re.match(r"^[\w.\-~]{1,200}$", v):
            raise SpaceError(f"{name}: letters, digits, . - _ ~ only", 400)
        filled = filled.replace("{" + name + "}", v)
    return filled


def probe(spec: str, fresh: bool = False) -> dict[str, Any]:
    """Everything a running server says about itself, in one record. Asleep or broken: just the stage."""
    rec = record(spec, fresh=fresh)
    if rec["stage"] != UP or not rec["host"]:
        return {"running": False, "stage": rec["stage"], "host": rec["host"], "base_path": rec["base_path"],
                "last_seen": last_seen(rec["id"])}

    def fetch():
        with ThreadPoolExecutor(6) as pool:
            f_api = pool.submit(_get_json, rec, "/openapi.json")
            f_meta = pool.submit(_get_json, rec, "/metadata")
            f_schema = pool.submit(_get_json, rec, "/schema")
            f_health = pool.submit(_get_json, rec, "/health")
            f_envs = pool.submit(_get_json, rec, "/list_environments")
            f_ui = pool.submit(_ui_path, rec)
            api, meta, schema, health, envs, ui = (f.result() for f in (f_api, f_meta, f_schema, f_health, f_envs, f_ui))
        routes = sorted((api or {}).get("paths", {}).keys()) if isinstance(api, dict) else []
        has = set(routes)
        openenv = isinstance(schema, dict) and ("action" in schema or "observation" in schema)
        tools, tools_error, rpc = _tools(rec) if ("/mcp" in has or openenv) else (None, None, None)
        env_names = [e for e in envs if isinstance(e, str) and re.match(r"^[\w.-]{1,80}$", e)] if isinstance(envs, list) else []
        task_api = bool(env_names) and "/{env_name}/splits" in has
        splits = []
        if task_api:
            raw = _get_json(rec, f"/{env_names[0]}/splits")
            for s in (raw or [])[:40] if isinstance(raw, list) else []:
                if isinstance(s, dict) and isinstance(s.get("name"), str):
                    splits.append({"name": s["name"], "type": s.get("type"), "num_tasks": s.get("num_tasks"), "default": s.get("default")})
            missing = [s for s in splits if not isinstance(s["num_tasks"], int)][:12]
            for s in missing:
                try:
                    r = _client.post(_url(rec, f"/{env_names[0]}/num_tasks"), json={"split": s["name"]}, timeout=8)
                    s["num_tasks"] = int(_json(r).get("num_tasks")) if r.status_code == 200 else None
                except Exception:  # noqa: BLE001
                    s["num_tasks"] = None
        names = {t["name"] for t in tools or []}
        harbor = {"run_rollout", "capabilities"} <= names or "harbor_env" in env_names
        caps = _harbor_caps(rec) if "capabilities" in names and harbor else None
        obs = (schema or {}).get("observation") if isinstance(schema, dict) else None
        version = (api.get("info") or {}).get("version") if isinstance(api, dict) and isinstance(api.get("info"), dict) else None
        info = {
            "running": True, "stage": rec["stage"], "host": rec["host"], "base_path": rec["base_path"],
            "ui": ui, "openenv": openenv or "/reset" in has,
            "metadata": meta if isinstance(meta, dict) else None, "health": health if isinstance(health, dict) else None,
            "schema": schema if isinstance(schema, dict) else None,
            "step_api": "/reset" in has and "/step" in has, "docs": "/openapi.json" in has or api is not None,
            "mcp": tools, "mcp_error": None if tools else tools_error,
            "task_api": {"env": env_names[0], "splits": splits} if task_api else None,
            "harbor": harbor, "harbor_caps": caps,
            "rewards": bool(isinstance(obs, dict) and "reward" in (obs.get("properties") or {})),
            "graders": sorted(n for n in names if re.search(r"(grade|score|submit|verify|reward|judge)", n, re.I)),
            "routes": [r for r in routes if r not in STANDARD_ROUTES][:40],
            "api": _api_routes(api, openenv or "/reset" in has),
            # which framework the server speaks, from what it serves
            "framework": "harbor" if harbor else "openenv" if (openenv or "/reset" in has) else
                         "nemo-gym" if {"/seed_session", "/verify"} <= has else
                         "ors" if {"/create_session", "/{env_name}/call"} <= has else "mcp" if tools else "api" if api else None,
            "openapi_version": version if isinstance(version, str) else None,
            # OpenEnv's two modes: simulation serves /reset, /step and /state; production serves only MCP and /ws
            "mode": ("simulation" if "/reset" in has else "production") if routes else None,
            "conformance": conformance({"openapi": api, "health": health, "metadata": meta, "schema": schema, "mcp": rpc,
                                        "mcp_tried": "/mcp" in has or openenv, "paths": routes}),
            "checked": time.time(),
        }
        if api is not None or isinstance(schema, dict) or isinstance(meta, dict) or tools:   # it answered: keep it
            remember(rec["id"], info)
        return info

    if fresh:
        catalog._memo.pop(("space-probe", spec), None)
    return catalog._cached(("space-probe", spec), 300, fetch)


# ── `openenv validate`, from what the probe saw ─────────────────────────────
# The six runtime criteria of `openenv validate <url>` (openenv/cli/_validation.py), each judged from the probe's
# own requests rather than by calling the Space again.
CRITERIA = (
    ("openapi_version_available", "GET /openapi.json returns OpenAPI info.version"),
    ("health_endpoint", "GET /health returns healthy status"),
    ("metadata_endpoint", "GET /metadata returns name and description"),
    ("schema_endpoint", "GET /schema returns action, observation, and state schemas"),
    ("mcp_endpoint", "POST /mcp is reachable and returns JSON-RPC payload"),
    ("mode_endpoint_consistency", "OpenAPI endpoint set matches OpenEnv mode contract"),
)


def conformance(facts: dict | None) -> list[dict[str, str]]:
    """pass / fail / unknown, with a short reason, for each criterion. `facts`: what the probe got back (`openapi`,
    `health`, `metadata`, `schema`: parsed JSON or None; `mcp`: {ok, detail} or None; `mcp_tried`; `paths`)."""
    def row(i: int, status: str, reason: str) -> dict[str, str]:
        return {"id": CRITERIA[i][0], "label": CRITERIA[i][1], "status": status, "reason": reason[:200]}

    if not facts:
        return [row(i, "unknown", "not seen running") for i in range(len(CRITERIA))]
    out = []
    api = facts.get("openapi")
    ver = (api.get("info") or {}).get("version") if isinstance(api, dict) and isinstance(api.get("info"), dict) else None
    out.append(row(0, "pass", f"version {ver}") if isinstance(ver, str) else
               row(0, "fail", "no info.version in its OpenAPI" if isinstance(api, dict) else "no JSON at /openapi.json"))
    h = facts.get("health")
    out.append(row(1, "pass", 'status "healthy"') if isinstance(h, dict) and h.get("status") == "healthy" else
               row(1, "fail", f'status {json.dumps(h.get("status"))[:60]}, not "healthy"' if isinstance(h, dict) else "no JSON at /health"))
    m = facts.get("metadata")
    missing = [k for k in ("name", "description") if not (isinstance(m, dict) and isinstance(m.get(k), str))]
    out.append(row(2, "pass", "name and description") if not missing else
               row(2, "fail", f"no {' or '.join(missing)}" if isinstance(m, dict) else "no JSON at /metadata"))
    sc = facts.get("schema")
    gone = [k for k in ("action", "observation", "state") if not (isinstance(sc, dict) and isinstance(sc.get(k), dict))]
    out.append(row(3, "pass", "action, observation and state") if not gone else
               row(3, "fail", f"no {', '.join(gone)} schema" if isinstance(sc, dict) else "no JSON at /schema"))
    rpc, paths = facts.get("mcp"), facts.get("paths")
    if isinstance(rpc, dict):
        out.append(row(4, "pass" if rpc.get("ok") else "fail", str(rpc.get("detail") or "")))
    elif paths:
        out.append(row(4, "fail", "no /mcp among its routes"))
    else:
        out.append(row(4, "unknown", "not asked: no OpenAPI and no schema"))
    if paths:
        has = set(paths)
        if "/reset" in has:
            lack = [p for p in ("/step", "/state") if p not in has]
            out.append(row(5, "pass", "simulation: /reset, /step and /state") if not lack else row(5, "fail", f"simulation without {' and '.join(lack)}"))
        else:
            extra = [p for p in ("/step", "/state") if p in has]
            out.append(row(5, "pass", "production: no /reset, /step or /state") if not extra else row(5, "fail", f"production with {' and '.join(extra)}"))
    else:
        out.append(row(5, "fail", "no OpenAPI paths to tell its mode"))
    return out


# ── last seen: the last good probe of each Space, kept ───────────────────────
SEEN_MAX = 512 * 1024   # bytes of one record; bigger tool and schema lists are trimmed to fit


def _seen_path(spec: str):
    return config.STORAGE_DIR / "space-probes" / f"{catalog._slug(catalog.check_spec(spec))}.json"


def _seen_record(spec: str, info: dict[str, Any]) -> dict[str, Any]:
    meta = info.get("metadata") if isinstance(info.get("metadata"), dict) else None
    out = {
        "spec": spec, "checked_at": info.get("checked") or time.time(),
        "framework": info.get("framework"), "openenv": info.get("openenv"), "step_api": info.get("step_api"),
        "mode": info.get("mode"), "ui": info.get("ui"), "base_path": info.get("base_path"), "harbor": info.get("harbor"),
        "rewards": info.get("rewards"), "openapi_version": info.get("openapi_version"),
        "metadata": {k: str(meta[k])[:2000] for k in ("name", "description", "version", "author", "documentation_url") if meta.get(k) is not None} if meta else None,
        "schema": info.get("schema"), "mcp": info.get("mcp"), "task_api": info.get("task_api"),
        "graders": info.get("graders"), "conformance": info.get("conformance"),
    }
    for trim in (None, "outputSchema", "inputSchema", "schema"):   # what's dropped first when it's too big to keep
        if trim == "schema":
            out["schema"] = None
        elif trim:
            out["mcp"] = [{k: v for k, v in t.items() if k != trim} for t in out.get("mcp") or []] or out.get("mcp")
        if len(json.dumps(out, default=str)) <= SEEN_MAX:
            break
    return out


def remember(spec: str, info: dict[str, Any]) -> None:
    """Keep this probe as the Space's last seen (best effort: a failed write never fails the probe)."""
    try:
        rec = _seen_record(spec, info)
        p = _seen_path(spec)
        old = last_seen(spec)
        same = old and {k: v for k, v in old.items() if k != "checked_at"} == json.loads(json.dumps({k: v for k, v in rec.items() if k != "checked_at"}, default=str))
        if same and rec["checked_at"] - (old.get("checked_at") or 0) < 60:
            return
        catalog.atomic_write(p, json.dumps(rec, default=str, separators=(",", ":")).encode())
        catalog._memo.pop(("space-seen", spec), None)
    except Exception:  # noqa: BLE001
        pass


def last_seen(spec: str) -> dict[str, Any] | None:
    """What the Space offered when it was last seen running here, with when (None if never)."""
    def read():
        try:
            doc = json.loads(_seen_path(spec).read_bytes())
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) and doc.get("spec") == spec else None

    try:
        return catalog._cached(("space-seen", spec), 30, read)
    except Exception:  # noqa: BLE001
        return None


def hub_dataset(name: str) -> str | None:
    """The Hub dataset an OpenEnv × Harbor server serves under a local name: `openenv harbor push` copies
    `org/name` to `/data/org__name`."""
    parts = name.rstrip("/").rsplit("/", 1)[-1].split("__")
    if len(parts) != 2 or not all(re.match(r"^[A-Za-z0-9][\w.-]{0,95}$", p) for p in parts):
        return None   # none, or more than one `__`: which one is the separator can't be told
    return f"{parts[0]}/{parts[1]}"


def _harbor_caps(rec: dict) -> dict[str, Any] | None:
    """What an OpenEnv × Harbor server can run: its engine, sandboxes, harnesses, and the Hub datasets it serves."""
    try:
        res = mcp_call(rec, "tools/call", {"name": "capabilities", "arguments": {}}, timeout=15)
        text = (res.get("content") or [{}])[0].get("text") if isinstance(res, dict) else res
        caps = json.loads(text) if isinstance(text, str) else text
    except (SpaceError, ValueError, TypeError, IndexError, AttributeError):
        return None
    if not isinstance(caps, dict):
        return None
    llm = caps.get("llm") if isinstance(caps.get("llm"), dict) else {}
    return {
        "engine": {"url": bool(llm.get("url")), "model": str(llm.get("model") or "")[:120], "ok": llm.get("ok")},
        "sandboxes": [{"name": str(x.get("name")), "available": bool(x.get("available"))} for x in caps.get("sandboxes") or [] if isinstance(x, dict)][:12],
        "harnesses": [{"name": str(x.get("name")), "status": str(x.get("status") or "")} for x in caps.get("harnesses") or [] if isinstance(x, dict)][:60],
        "datasets": [{"name": str(x.get("name")), "num_tasks": x.get("num_tasks"), "hub": hub_dataset(str(x.get("name") or ""))}
                     for x in caps.get("datasets") or [] if isinstance(x, dict)][:20],
    }


# ── waking and restarting ────────────────────────────────────────────────────
_woken: dict[str, float] = {}


def wake(spec: str) -> dict[str, Any]:
    """Wake a sleeping Space the way its Hub page does for any visitor ("Restart this Space"). At most once a minute."""
    rec = record(spec, fresh=True)
    if rec["stage"] in STARTING or rec["stage"] == UP:
        return {"stage": rec["stage"], "woken": False}
    if rec["stage"] != "SLEEPING":
        raise SpaceError(f"this Space is {rec['stage'].lower().replace('_', ' ')}: only its owner can restart it", 409)
    with _lock:
        if time.time() - _woken.get(rec["id"], 0) < 60:
            return {"stage": "APP_STARTING", "woken": True}
        _woken[rec["id"]] = time.time()
    try:
        r = httpx.post(f"https://huggingface.co/spaces/{rec['id']}/start", data={"csrf": ""}, timeout=15, follow_redirects=False)
    except httpx.HTTPError as e:
        raise SpaceError(f"couldn't reach the Hub: {type(e).__name__}", 502) from None
    if r.status_code not in (200, 302, 303, 409):
        raise SpaceError(f"the Hub refused to wake it (HTTP {r.status_code})", 502)
    return {"stage": record(spec, fresh=True)["stage"], "woken": True}


def restart(spec: str, token: str) -> dict[str, Any]:
    """Restart a Space with the visitor's own token: works when they may write to it (their Space, or their org's)."""
    from huggingface_hub.errors import HfHubHTTPError

    rec = record(spec, fresh=True)
    try:
        catalog._api(token).restart_space(rec["id"])
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code in (401, 403):
            raise SpaceError("you can't restart this Space: it takes write access to it (its owner, or a member of its "
                             "organization, signed in with a token that may write)", 403) from None
        raise SpaceError(f"the Hub refused the restart (HTTP {code})", 502) from None
    return {"stage": record(spec, fresh=True)["stage"]}


# ── the Task API ─────────────────────────────────────────────────────────────
def _withhold(v: Any, depth: int = 0) -> tuple[Any, list[str]]:
    """A task with answer-like fields left out (recursively), and the names it left out."""
    gone: list[str] = []
    if depth > 6:
        return v, gone
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if isinstance(k, str) and ANSWER.search(k):
                gone.append(k)
                continue
            out[k], g = _withhold(x, depth + 1)
            gone += g
        return out, gone
    if isinstance(v, list):
        out = []
        for x in v[:500]:
            y, g = _withhold(x, depth + 1)
            out.append(y)
            gone += g
        return out, gone
    return v, gone


def _task_api(spec: str) -> tuple[dict, str]:
    info = probe(spec)
    if not info.get("running"):
        raise SpaceError("this Space isn't running: wake it first", 409)
    if not info.get("task_api"):
        raise SpaceError("this server has no Task API", 404)
    return record(spec), info["task_api"]["env"]


def tasks(spec: str, split: str, start: int, stop: int) -> dict[str, Any]:
    rec, env = _task_api(spec)
    start, stop = max(0, int(start)), max(0, int(stop))
    stop = min(stop, start + 50)
    try:
        r = _client.post(_url(rec, f"/{env}/task_range"), json={"split": split, "start": start, "stop": stop}, timeout=20)
    except httpx.HTTPError as e:
        raise SpaceError(f"couldn't reach the Space: {type(e).__name__}", 502) from None
    if r.status_code != 200:
        raise SpaceError(f"the Task API answered HTTP {r.status_code}", 502)
    rows = (_json(r) or {}).get("tasks") or []
    rows, gone = _withhold(rows if isinstance(rows, list) else [])
    return {"tasks": rows, "start": start, "stop": stop, "withheld": sorted(set(gone))}


def task(spec: str, split: str, index: int) -> dict[str, Any]:
    rec, env = _task_api(spec)
    try:
        r = _client.post(_url(rec, f"/{env}/task"), json={"split": split, "index": int(index)}, timeout=20)
    except httpx.HTTPError as e:
        raise SpaceError(f"couldn't reach the Space: {type(e).__name__}", 502) from None
    if r.status_code != 200:
        raise SpaceError(f"the Task API answered HTTP {r.status_code}", 404 if r.status_code in (400, 404) else 502)
    t, gone = _withhold((_json(r) or {}).get("task"))
    return {"task": t, "withheld": sorted(set(gone))}


# ── playground sessions ──────────────────────────────────────────────────────
# One live episode per session, over one WebSocket to the server's /ws: it carries reset/step/state and MCP calls
# ({"type": "mcp"}) on the same environment instance (the HTTP /reset, /step and /mcp are stateless). A server whose
# socket won't open still gets its MCP tools, one stateless HTTP call at a time. Sessions belong to whoever started
# them (the page's address and account), end after 10 idle minutes or an hour, and are capped per owner and in total.
IDLE, LIFE, PER_OWNER, TOTAL = 600, 3600, 3, 80
PER_AGENT_HOST = 8   # MCP clients: several coding agents on one machine each keep their own episode


class Session:
    def __init__(self, spec: str, owner: str):
        self.id = secrets.token_urlsafe(18)
        self.spec, self.owner = spec, owner
        self.created = self.used = time.time()
        self.ws = None
        self.ws_failed = False
        self.http: httpx.Client | None = None   # the session's own cookie jar for the server's routes (NeMo Gym keeps state in a cookie)
        self.lock = threading.Lock()
        self.steps = 0
        self.rpc = 0

    def close(self) -> None:
        if self.http is not None:
            try:
                self.http.close()
            except Exception:  # noqa: BLE001
                pass
            self.http = None
        ws, self.ws = self.ws, None
        if ws is None:
            return
        try:
            ws.send(json.dumps({"type": "close"}))
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass


_sessions: dict[str, Session] = {}


def _sweep() -> None:
    now = time.time()
    with _lock:
        dead = [s for s in _sessions.values() if now - s.used > IDLE or now - s.created > LIFE]
        for s in dead:
            _sessions.pop(s.id, None)
    for s in dead:
        s.close()


def start(spec: str, owner: str) -> dict[str, Any]:
    info = probe(spec)
    if not info.get("running"):
        raise SpaceError("this Space isn't running: wake it first", 409)
    if not (info.get("openenv") or info.get("mcp") or info.get("api")):
        raise SpaceError("this Space has no API to play with", 404)
    _sweep()
    cap = PER_AGENT_HOST if owner.startswith("mcp:") else PER_OWNER
    with _lock:
        mine = [s for s in _sessions.values() if s.owner == owner]
        if len(mine) >= cap:   # the oldest of yours makes room
            oldest = min(mine, key=lambda s: s.used)
            _sessions.pop(oldest.id, None)
            threading.Thread(target=oldest.close, daemon=True).start()
        if len(_sessions) >= TOTAL:
            raise SpaceError("too many live sessions on this explorer right now: try again in a few minutes", 429)
        s = Session(catalog.check_spec(spec), owner)
        _sessions[s.id] = s
    return {"session": s.id}


def _get(sid: str, owner: str) -> Session:
    _sweep()
    s = _sessions.get(sid)
    if s is None or s.owner != owner:
        raise SpaceError("this session has ended: start a new one", 410)
    s.used = time.time()
    return s


def end(sid: str, owner: str) -> None:
    s = _sessions.get(sid)
    if s is not None and s.owner == owner:
        _sessions.pop(sid, None)
        s.close()


def _open(s: Session) -> None:
    from websockets.sync.client import connect

    rec = record(s.spec)
    if not rec["host"]:
        raise SpaceError("this Space has no app address", 404)
    try:   # the sync client never follows redirects, so this only ever reaches the Space's own host
        s.ws = connect("wss://" + rec["host"][len("https://"):] + "/ws", open_timeout=20, max_size=MAX_REPLY, close_timeout=3,
                       user_agent_header="hf-rl-explorer (+https://huggingface.co/FineEnvs)")
    except Exception as e:  # noqa: BLE001
        s.ws_failed = True
        raise SpaceError(f"couldn't open a session on the Space ({type(e).__name__})", 502) from None


def _ws_send(s: Session, msg: dict, timeout: float) -> dict:
    if s.ws is None:
        _open(s)
    try:
        s.ws.send(json.dumps(msg))
        reply = json.loads(s.ws.recv(timeout=timeout))
    except TimeoutError:
        s.close()
        raise SpaceError(f"the Space didn't answer within {int(timeout)} s; the session was closed", 504) from None
    except Exception as e:  # noqa: BLE001 - the socket dropped: the episode is gone
        s.close()
        raise SpaceError(f"the session dropped ({type(e).__name__}): start a new episode", 502) from None
    if not isinstance(reply, dict):
        raise SpaceError("the Space answered something unexpected", 502)
    if reply.get("type") == "error":
        data = reply.get("data") or {}
        raise SpaceError(str(data.get("message") or data.get("detail") or data)[:600] if isinstance(data, dict) else str(data)[:600], 422)
    return reply.get("data") or {}


def _rpc(s: Session, method: str, params: dict, timeout: float) -> Any:
    """An MCP call inside the session's episode; over plain HTTP (stateless) when the server has no socket for it."""
    if not s.ws_failed:
        try:
            s.rpc += 1
            data = _ws_send(s, {"type": "mcp", "data": {"jsonrpc": "2.0", "id": s.rpc, "method": method, "params": params}}, timeout)
        except SpaceError as e:
            if "capacity" in str(e).lower():   # its one session is someone else's: tools still work, one call at a time
                s.close()
                s.ws_failed = True
            elif not s.ws_failed:
                raise
        else:
            if isinstance(data, dict) and data.get("error"):
                err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
                raise SpaceError(str(err.get("message") or "error")[:600], 422)
            return data.get("result") if isinstance(data, dict) else data
    return mcp_call(record(s.spec), method, params, timeout=timeout)


def _http(s: Session, data: dict) -> dict[str, Any]:
    """One call to a route the server published, with the session's cookies; its status and body (JSON, text, image)."""
    method = str(data.get("method") or "").upper()
    if method not in ("GET", "POST"):
        raise SpaceError("GET or POST only", 400)
    info = probe(s.spec)
    path = route_url(info, method, str(data.get("path") or ""), data.get("params") if isinstance(data.get("params"), dict) else {})
    rec = record(s.spec)
    if s.http is None:
        s.http = httpx.Client(base_url=rec["host"], timeout=httpx.Timeout(120, connect=8), follow_redirects=False,
                              headers={"User-Agent": "hf-rl-explorer (+https://huggingface.co/FineEnvs)"})
    body = data.get("body")
    try:
        r = s.http.request(method, path, json=body if method == "POST" else None)
    except httpx.TimeoutException:
        raise SpaceError("the Space didn't answer within 120 s", 504) from None
    except httpx.HTTPError as e:
        raise SpaceError(f"couldn't reach the Space: {type(e).__name__}", 502) from None
    if len(r.content) > MAX_REPLY:
        raise SpaceError("the reply is too large to show", 502)
    ct = r.headers.get("content-type", "")
    out: dict[str, Any] = {"status": r.status_code, "content_type": ct.split(";")[0]}
    if "json" in ct:
        try:
            out["json"] = r.json()
        except ValueError:
            out["text"] = r.text[:200_000]
        else:
            if method == "GET":   # browsing what it serves: answer-like fields stay out, as in the Task API browser
                out["json"], gone = _withhold(out["json"])
                if gone:
                    out["withheld"] = sorted(set(gone))
    elif ct.startswith("image/") and ct.split(";")[0] in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        out["image"] = f"data:{ct.split(';')[0]};base64,{base64.b64encode(r.content).decode()}"
    else:
        out["text"] = r.text[:200_000]
    return out


def act(sid: str, owner: str, op: str, data: dict | None = None) -> dict[str, Any]:
    """reset / step / state, or call / tools (MCP), inside the session's episode."""
    s = _get(sid, owner)
    data = data if isinstance(data, dict) else {}
    with s.lock:
        t0 = time.time()
        if op == "reset":   # a new episode: safe to try again once on a fresh socket if the first one drops
            try:
                out = _ws_send(s, {"type": "reset", "data": data}, timeout=120)
            except SpaceError as e:
                if e.status != 502:
                    raise
                s.close()
                s.ws_failed = False
                out = _ws_send(s, {"type": "reset", "data": data}, timeout=120)
            s.steps = 0
        elif op == "step":
            out = _ws_send(s, {"type": "step", "data": data}, timeout=180)
            s.steps += 1
        elif op == "state":
            out = _ws_send(s, {"type": "state"}, timeout=30)
        elif op == "call":
            name = data.get("name")
            if not isinstance(name, str) or not re.match(r"^[\w.\-/]{1,120}$", name):
                raise SpaceError("which tool?", 400)
            args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
            out = _rpc(s, "tools/call", {"name": name, "arguments": args}, timeout=900 if name == "run_rollout" else 180)
            s.steps += 1
        elif op == "tools":
            out = _rpc(s, "tools/list", {}, timeout=20)
        elif op == "http":
            out = _http(s, data)
            s.steps += 1
        else:
            raise SpaceError(f"unknown operation {op!r}", 400)
        return {"result": out, "ms": int((time.time() - t0) * 1000), "steps": s.steps, "stateful": not s.ws_failed}


def live_sessions() -> int:
    _sweep()
    return len(_sessions)


def image_parts(obs: Any) -> list[tuple[str, str]]:
    """(mime type, base64) for every image a reply carries as base64: for MCP clients, which can show them."""
    found: list[tuple[str, str]] = []

    def walk(v: Any, key: str = "", depth: int = 0) -> None:
        if depth > 5 or len(found) >= 4:
            return
        if isinstance(v, dict):
            fmt = str(v.get("image_format") or v.get("format") or "png").lower()
            for k, x in v.items():
                if isinstance(x, str) and len(x) > 200 and re.search(r"(image|img|png|jpe?g|frame|screenshot|pixels)", k, re.I):
                    if x.startswith("data:image/"):
                        head, _, b64 = x.partition(",")
                        found.append((head[5:].split(";")[0], b64))
                    elif re.match(r"^[A-Za-z0-9+/=\s]+$", x[:400]):
                        try:
                            base64.b64decode(x[:64] + "=" * (-len(x[:64]) % 4))
                            found.append((f"image/{'jpeg' if fmt in ('jpg', 'jpeg') else fmt if fmt in ('png', 'webp', 'gif') else 'png'}", x))
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    walk(x, k, depth + 1)
        elif isinstance(v, list):
            for x in v[:50]:
                walk(x, key, depth + 1)

    walk(obs)
    return found
