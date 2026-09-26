"""MiMo RL Environment Explorer: browse the tasks, run a rollout on one, watch it, see it graded.

    uv run uvicorn app.main:app --reload      # local: your HF token, traces in ./.local-runs
"""

from __future__ import annotations

import json
import mimetypes
import os
import random
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from . import auth, catalog, config, endpoints, models, previews, store, version
from .runner import core

app = FastAPI(title="MiMo RL Environment Explorer", docs_url="/api/docs")
app.include_router(auth.router)


@app.middleware("http")
async def _revalidate(request: Request, call_next):
    """Code and styles revalidate on every load (cheap: ETag -> 304), so a deploy is never half-cached."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"} | {h.strip() for h in os.environ.get("MIMO_ALLOWED_HOSTS", "").split(",") if h.strip()}
CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data: blob: https:; connect-src 'self'; frame-src 'self'; object-src 'none'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'self' https://huggingface.co https://*.hf.space")


@app.middleware("http")
async def _security(request: Request, call_next):
    """Locally, only answer to localhost: a web page could otherwise DNS-rebind its own name to 127.0.0.1 and drive
    this server (and the machine's HF token) from the browser. Everywhere: the usual security headers."""
    if config.LOCAL_MODE:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not (request.headers.get("host") or "").startswith("[") \
            else (request.headers.get("host") or "").split("]")[0] + "]"
        if host not in LOCAL_HOSTS:
            return JSONResponse({"detail": "host not allowed (set MIMO_ALLOWED_HOSTS to serve under another name)"}, 403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    if not request.url.path.startswith("/api/"):
        resp.headers.setdefault("Content-Security-Policy", CSP)
    return resp


@app.middleware("http")
async def _same_origin_posts(request: Request, call_next):
    """The session cookie is SameSite=None (it has to work in the huggingface.co iframe), so a state-changing
    request must prove it came from this page: another site could otherwise start rollouts billed to you."""
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if origin and origin.split("://", 1)[-1] != host:
            return JSONResponse({"detail": "cross-site request refused"}, 403)
        if not origin and request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
            return JSONResponse({"detail": "cross-site request refused"}, 403)
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    store.mark_interrupted()
    import threading
    threading.Thread(target=catalog.warm, daemon=True, name="warm").start()


# ── who ──────────────────────────────────────────────────────────────────────
@app.get("/api/me")
def me(request: Request):
    u = auth.current_user(request)
    return {"user": auth.public(u), "local": config.LOCAL_MODE, "oauth": not config.LOCAL_MODE,
            "version": version.app_version(), "source": version.source_hash(),
            "missing_scopes": (u or {}).get("missing_scopes", []),
            "storage": "local folder" if not config.STORAGE_DIR.as_posix().startswith("/data") else "private bucket"}


# ── tasks ────────────────────────────────────────────────────────────────────
@app.get("/api/tasks/{task_id}")
def task(task_id: str):
    try:
        v = catalog.view(task_id)
    except FileNotFoundError as e:
        raise HTTPException(502, f"could not fetch this environment's files: {e}")
    if not v:
        raise HTTPException(404, "no such task")
    from .runner.domains import run_defaults
    return {**v, "run_defaults": run_defaults(v["id"], v["domain"])}


@app.get("/api/random")
def random_task(domain: str | None = None, runnable: bool = True):
    envs = [e for e in catalog.index()["envs"] if (not domain or e["d"] == domain)]
    return {"id": random.choice(envs)["id"]}


@app.get("/api/tasks/{task_id}/file")
def task_file(task_id: str, path: str, download: bool = False):
    try:
        p = catalog.workspace_path(task_id, path)
    except (PermissionError, TypeError):
        raise HTTPException(403, "outside the workspace")
    if not p.is_file():
        raise HTTPException(404, "no such file")
    mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    headers = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}   # never run task HTML with our origin
    return FileResponse(p, media_type=mt, headers=headers,
                        content_disposition_type="attachment" if download else "inline", filename=p.name)


@app.get("/api/tasks/{task_id}/preview")
def task_preview(task_id: str, path: str):
    try:
        p = catalog.workspace_path(task_id, path)
    except (PermissionError, TypeError):
        raise HTTPException(403, "outside the workspace")
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return previews.preview(p)


def _repo_snapshot(task_id: str) -> dict:
    import gzip

    if "/" in task_id or ".." in task_id:
        raise HTTPException(404, "no such task")
    p = config.STORAGE_DIR / "repo-snapshots" / f"{task_id}.json.gz"
    if not p.is_file():
        raise HTTPException(404, "this repository has not been snapshotted yet")
    return _read_snapshot(str(p), p.stat().st_mtime)


@lru_cache(maxsize=32)
def _read_snapshot(path: str, mtime: float) -> dict:
    import gzip

    return json.loads(gzip.decompress(open(path, "rb").read()))


@app.get("/api/tasks/{task_id}/repo")
def task_repo(task_id: str):
    """The repository inside a Code task's image: files, base commit, upstream, history truncation."""
    d = _repo_snapshot(task_id)
    return {k: d.get(k) for k in ("base", "remote", "history_truncated", "files", "cwd", "taken_at")}


@app.get("/api/tasks/{task_id}/repo/file")
def task_repo_file(task_id: str, path: str):
    d = _repo_snapshot(task_id)
    if path not in d.get("texts", {}):
        raise HTTPException(404, "no preview for this file")
    return {"path": path, "text": d["texts"][path]}


@app.get("/api/tasks/{task_id}/systems/{system}/{table}")
def system_table(task_id: str, system: str, table: str, limit: int = 50):
    try:
        return catalog.db_rows(catalog.system_db(task_id, system), table, min(limit, 500))
    except (PermissionError, KeyError):
        raise HTTPException(404, "no such table")


# ── models ───────────────────────────────────────────────────────────────────
@app.get("/api/models")
def list_models():
    try:
        return models.catalog()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach HF Inference Providers: {e}")


# ── rollouts ─────────────────────────────────────────────────────────────────
SAFE_NAME = r"^[A-Za-z0-9._:/@+\-]{1,200}$"   # model ids: letters, digits and the punctuation real ids use


class Endpoint(BaseModel):
    base_url: str = Field(max_length=500)
    model: str = Field(pattern=SAFE_NAME)
    api_key: str | None = Field(None, max_length=500)
    price_in: float | None = Field(None, ge=0, le=1000)    # $ per 1M tokens, only for the cost shown
    price_out: float | None = Field(None, ge=0, le=1000)


class Params(BaseModel):
    thinking: Literal["default", "none", "low", "medium", "high"] = "low"
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=256, le=128000)
    steps: int | None = Field(None, ge=1, le=1000)
    timeout_min: int | None = Field(None, ge=2, le=120)


class RunRequest(BaseModel):
    task_id: str = Field(max_length=200)
    model: str | None = Field(None, pattern=SAFE_NAME)
    provider: str | None = Field(None, pattern=SAFE_NAME)
    judge: str | None = Field(None, pattern=SAFE_NAME)
    endpoint: Endpoint | None = None
    params: Params = Params()
    visibility: Literal["public", "private"] = "public"


class EndpointProbe(BaseModel):
    base_url: str = Field(max_length=500)
    api_key: str | None = Field(None, max_length=500)
    model: str | None = Field(None, pattern=SAFE_NAME)


_probes: dict[str, list[float]] = {}


def _probe_limit(user: str) -> None:
    """This server makes the call, so cap how often one account can have it do so."""
    import time as _t
    now = _t.time()
    hits = [t for t in _probes.get(user, []) if now - t < 600]
    if len(hits) >= 30:
        raise HTTPException(429, "Too many endpoint checks. Wait a few minutes.")
    _probes[user] = hits + [now]


@app.post("/api/endpoints/models")
def endpoint_models(body: EndpointProbe, request: Request):
    _probe_limit(auth.require_user(request)["name"])   # this server makes the call, so only for signed-in users
    try:
        return {"models": endpoints.list_models(body.base_url, body.api_key)}
    except endpoints.EndpointError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Couldn't reach the endpoint: {type(e).__name__}")


@app.post("/api/endpoints/test")
def endpoint_test(body: EndpointProbe, request: Request):
    _probe_limit(auth.require_user(request)["name"])
    try:
        return endpoints.test(body.base_url, body.api_key, body.model or "")
    except endpoints.EndpointError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Couldn't reach the endpoint: {type(e).__name__}")


@app.post("/api/runs")
def start_run(body: RunRequest, request: Request):
    u = auth.require_user(request)
    v = catalog.view(body.task_id)
    if not v or not v.get("runnable"):
        raise HTTPException(400, "this task cannot be run here")
    endpoint, agent_key = None, None
    if body.endpoint:
        try:
            base = endpoints.check_url(body.endpoint.base_url)
        except endpoints.EndpointError as e:
            raise HTTPException(400, str(e))
        endpoint = {"base_url": base, "host": httpx.URL(base).host, "price_in": body.endpoint.price_in,
                    "price_out": body.endpoint.price_out}
        model, provider, agent_key = body.endpoint.model.strip(), None, body.endpoint.api_key
        if not model:
            raise HTTPException(400, "enter the model name your endpoint expects")
    else:
        m = models.get(body.model or "")
        if not m or not m["tools"]:
            raise HTTPException(400, "pick a model that supports tool calling")
        model, provider = body.model, body.provider or m["provider"]
    need = (v.get("verify") or {}).get("needs_judge")
    judge = body.judge if need else None
    if need and not judge:
        raise HTTPException(400, "this task is graded by a model: pick a judge")
    try:
        run = core.submit(u["name"], u["token"], {"id": v["id"], "domain": v["domain"], "title": v["short_title"],
                                                  "facets": v.get("facets")}, model, provider, judge,
                          endpoint=endpoint, agent_key=agent_key, params=body.params.model_dump(exclude_defaults=True),
                          visibility=body.visibility)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return run


@app.get("/api/runs")
def list_runs(request: Request, task_id: str | None = None):
    """Your own rollouts only, public and private."""
    u = auth.current_user(request)
    if not u:
        return {"runs": []}
    return {"runs": [_owner_view(r) for r in store.list_runs(user=u["name"], task_id=task_id)]}


# ── visibility ───────────────────────────────────────────────────────────────
# A public rollout is shown to everyone WITHOUT who ran it: the public projection drops the user, sandbox ids and
# a custom endpoint's URL, and scrubs the owner's username out of every string in the trace. Private rollouts are
# visible to their owner only and 404 for everyone else. Rollouts from before visibility existed stay private.
PUBLIC_FIELDS = ("id", "task_id", "domain", "title", "facets", "model", "provider", "judge", "status", "reward",
                 "reward_error", "tokens", "cost", "created_at", "started_at", "finished_at", "flavor", "harness",
                 "params", "image", "provenance", "phase")


def _is_public(run: dict) -> bool:
    return run.get("visibility") == "public"


def _scrub(value, secrets_: list[str] | str):
    if isinstance(secrets_, str):
        secrets_ = [secrets_] if secrets_ and len(secrets_) > 2 else []
    if isinstance(value, str):
        for x in secrets_:
            value = value.replace(x, "[redacted]" if "." in x or "/" in x else "[user]")
        return value
    if isinstance(value, dict):
        return {k: _scrub(v, secrets_) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, secrets_) for v in value]
    return value


def _private_strings(run: dict) -> list[str]:
    """What must never show publicly: the owner's name, and where their own endpoint lives."""
    ep = run.get("endpoint") or {}
    return [x for x in (run.get("user"), ep.get("base_url"), ep.get("host")) if x and len(x) > 2]


def _public_view(run: dict) -> dict:
    out = {k: run.get(k) for k in PUBLIC_FIELDS if k in run}
    ep = run.get("endpoint")
    if ep:   # which kind of model it was, and its settings, but never where it is served from
        out["endpoint"] = {"custom": True, "price_in": ep.get("price_in"), "price_out": ep.get("price_out")}
    if run.get("error"):
        out["error"] = run["error"]
    out["visibility"] = "public"
    return _scrub(out, _private_strings(run))


def _owner_view(run: dict) -> dict:
    return {**{k: v for k, v in run.items() if k not in ("scripted", "scripted_final")}, "is_owner": True,
            "visibility": run.get("visibility") or "private"}


def _visible(request: Request, run_id: str) -> tuple[dict, bool]:
    """(run, is_owner) if this person may see the rollout, else 404."""
    u = auth.current_user(request)
    try:
        run = store.get(run_id)
    except ValueError:
        run = None
    if run and u and run.get("user") == u["name"]:
        return run, True
    if run and _is_public(run):
        return run, False
    raise HTTPException(404, "no such rollout")


def _own(request: Request, run_id: str) -> dict:
    run, owner = _visible(request, run_id)
    if not owner:
        raise HTTPException(404, "no such rollout")
    return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request, after: int = 0):
    run, owner = _visible(request, run_id)
    live = core.live(run_id)
    if live is not None:
        run = {**run, **live.run, "cost": live.cost_now()}
    events = core.events(run_id, after)
    stream = dict(live.stream, age=round(time.time() - live.stream["since"], 1)) if live is not None and live.stream else None
    if owner:
        return {"run": _owner_view(run), "events": events, "live": live is not None, "stream": stream}
    return {"run": _public_view(run), "events": _scrub(events, _private_strings(run)), "live": live is not None, "stream": stream}


class Visibility(BaseModel):
    visibility: Literal["public", "private"]


@app.post("/api/runs/{run_id}/visibility")
def set_visibility(run_id: str, body: Visibility, request: Request):
    _own(request, run_id)
    live = core.live(run_id)
    if live is not None:
        live.run["visibility"] = body.visibility
    store.update(run_id, visibility=body.visibility)
    return {"visibility": body.visibility}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request):
    _own(request, run_id)
    return {"cancelled": core.cancel(run_id)}


# still running, or stopped before it said anything: not useful to anyone else yet
HIDDEN_STATES = {"queued", "starting", "setup", "running", "verifying", "interrupted", "cancelled"}


def _stats(runs: list[dict]) -> dict:
    scored = [r for r in runs if r.get("reward") is not None and r.get("status") == "done"]
    hist = [0] * 10
    for r in scored:
        hist[min(9, int(float(r["reward"]) * 10))] += 1
    by: dict[str, list[float]] = {}
    for r in scored:
        key = r.get("model") or "?"
        by.setdefault(key, []).append(float(r["reward"]))
    models_ = sorted(({"model": m, "custom": any(x.get("endpoint") for x in runs if x.get("model") == m), "runs": len(v),
                       "mean": round(sum(v) / len(v), 4), "best": max(v), "full": sum(x >= 0.999 for x in v)}
                      for m, v in by.items()), key=lambda x: (-x["mean"], -x["runs"]))
    return {"runs": len(runs), "scored": len(scored), "mean": round(sum(float(r["reward"]) for r in scored) / len(scored), 4) if scored else None,
            "full": sum(float(r["reward"]) >= 0.999 for r in scored), "histogram": hist, "by_model": models_}


@app.get("/api/tasks/{task_id}/rollouts")
def task_rollouts(task_id: str, limit: int = 50, offset: int = 0):
    """Everyone's public rollouts on one task, anonymous, with the spread of rewards."""
    runs = [r for r in store.list_runs(task_id=task_id, public=True, limit=100000) if r.get("status") not in HIDDEN_STATES]
    page = runs[offset:offset + min(limit, 200)]
    return {"stats": _stats(runs), "runs": [_public_view(r) for r in page], "total": len(runs)}


def _band(r: dict) -> str:
    rw = r.get("reward")
    if rw is None or r.get("status") != "done":
        return "unscored"
    return "full" if rw >= 0.999 else "zero" if rw <= 0 else "partial"


@app.get("/api/community")
def community(domain: str | None = None, model: str | None = None, served: str | None = None, judge: str | None = None,
              reward: str | None = None, thinking: str | None = None, q: str | None = None, sort: str = "new",
              limit: int = 50, offset: int = 0):
    """Public rollouts across all tasks, filtered every way a reader might want, and how each model does."""
    everyone = [r for r in store.list_runs(public=True, limit=100000) if r.get("status") not in HIDDEN_STATES]
    def served_by(r):
        return "own endpoint" if r.get("endpoint") else (r.get("provider") or "auto")
    def keep(r):
        if domain and r.get("domain") != domain: return False
        if model and r.get("model") != model: return False
        if served and served_by(r) != served: return False
        if judge and (r.get("judge") or "") != judge: return False
        if reward and _band(r) != reward: return False
        if thinking and ((r.get("params") or {}).get("thinking") or "default") != thinking: return False
        if q and q.lower() not in f"{r.get('title', '')} {r.get('task_id', '')}".lower(): return False
        return True
    runs = [r for r in everyone if keep(r)]
    key = {"reward_desc": lambda r: -(r.get("reward") if r.get("reward") is not None else -1),
           "reward_asc": lambda r: (r.get("reward") if r.get("reward") is not None else 2),
           "cost_asc": lambda r: (r.get("cost") or {}).get("total", 0), "cost_desc": lambda r: -(r.get("cost") or {}).get("total", 0)}.get(sort)
    if key:
        runs.sort(key=key)
    def counts(f):
        c: dict[str, int] = {}
        for r in everyone:
            v = f(r)
            if v:
                c[v] = c.get(v, 0) + 1
        return sorted(c.items(), key=lambda kv: -kv[1])
    facets = {"domain": counts(lambda r: r.get("domain")), "model": counts(lambda r: r.get("model")), "served": counts(served_by),
              "judge": counts(lambda r: r.get("judge")), "reward": counts(_band),
              "thinking": counts(lambda r: (r.get("params") or {}).get("thinking") or "default")}
    return {"stats": _stats(runs), "facets": facets, "total": len(runs), "tasks": len({r["task_id"] for r in runs}),
            "runs": [_public_view(r) for r in runs[offset:offset + min(limit, 200)]]}


@app.get("/api/community/tasks")
def community_tasks():
    """Per task: how many public rollouts, and how they scored. Tasks absent here have none yet."""
    out: dict[str, dict] = {}
    for r in store.list_runs(public=True, limit=100000):
        if r.get("status") in HIDDEN_STATES:
            continue
        t = out.setdefault(r["task_id"], {"runs": 0, "scored": 0, "sum": 0.0, "best": None, "last": 0})
        t["runs"] += 1
        t["last"] = max(t["last"], r.get("created_at") or 0)
        if r.get("reward") is not None and r.get("status") == "done":
            t["scored"] += 1; t["sum"] += float(r["reward"])
            t["best"] = max(t["best"] if t["best"] is not None else 0, float(r["reward"]))
    return {"tasks": {k: {"runs": v["runs"], "mean": round(v["sum"] / v["scored"], 4) if v["scored"] else None, "best": v["best"],
                          "last": v["last"]} for k, v in out.items()}}


@app.get("/api/runs/{run_id}/artifacts/{name}")
def artifact(run_id: str, name: str, request: Request):
    _visible(request, run_id)
    data = store.read_artifact(run_id, name)
    if data is None:
        raise HTTPException(404, "no such artifact")
    return Response(data, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                    headers={"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"})


# ── the model proxy ──────────────────────────────────────────────────────────
# On the Space, a sandbox holds no credential: OpenCode (and the General verifier's judge) call this route with a
# per-rollout capability in the path. It works only while that rollout is live, only for the rollout's own agent
# model and judge, and this server adds the user's token or endpoint key on the way out.
@app.post("/api/llm/{cap}/v1/chat/completions")
async def llm_proxy(cap: str, request: Request):
    import asyncio

    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse

    r = core.by_cap(cap)
    if r is None:
        return JSONResponse({"error": {"message": "unknown or finished rollout"}}, 404)
    raw = await request.body()
    if len(raw) > 20_000_000:
        return JSONResponse({"error": {"message": "request too large"}}, 413)
    try:
        body = json.loads(raw)
    except ValueError:
        return JSONResponse({"error": {"message": "invalid JSON"}}, 400)
    up = r.upstream(str(body.get("model") or ""))
    if up is None:
        return JSONResponse({"error": {"message": "this rollout may not call that model"}}, 403)
    base, key = up
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    ext = {}
    if r.run.get("endpoint") and base != config.ROUTER:   # the user's own endpoint: connect to the address we checked
        try:
            url, host, ext = await asyncio.to_thread(endpoints.pinned, url)
        except endpoints.EndpointError as e:
            return JSONResponse({"error": {"message": str(e)}}, 502)
        headers.update(host)
    client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30), follow_redirects=False)
    try:
        resp = await client.send(client.build_request("POST", url, content=raw, headers=headers, extensions=ext), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        return JSONResponse({"error": {"message": f"upstream unreachable: {type(e).__name__}"}}, 502)

    async def relay():
        # OpenCode reports a step only once the whole reply is in, which for a long file can take minutes. Counting
        # what streams past here lets the rollout page show the model is still writing, and how much.
        s = r.stream = {"since": time.time(), "text": 0, "thinking": 0, "tool": 0}
        buf = b""
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: {"):
                        continue
                    try:
                        choices = json.loads(line[6:]).get("choices") or []
                    except ValueError:
                        continue
                    for c in choices:
                        d = c.get("delta") or {}
                        s["text"] += len(d.get("content") or "")
                        s["thinking"] += len(d.get("reasoning_content") or d.get("reasoning") or "")
                        s["tool"] += sum(len((t.get("function") or {}).get("arguments") or "") for t in d.get("tool_calls") or [])
        finally:
            if r.stream is s:
                r.stream = None

    async def close():
        await resp.aclose()
        await client.aclose()
    return StreamingResponse(relay(), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type", "application/json"), background=BackgroundTask(close))


# ── the site ─────────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def _nf(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(exc, "detail", "not found")}, 404)
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
