"""MiMo RL Environments Explorer: browse the tasks, run a rollout on one, watch it, see it graded.

    uv run uvicorn app.main:app --reload      # local: your HF token, traces in ./.local-runs
"""

from __future__ import annotations

import mimetypes
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, catalog, config, models, previews, store
from .runner import core

app = FastAPI(title="MiMo RL Environments Explorer", docs_url="/api/docs")
app.include_router(auth.router)


@app.middleware("http")
async def _revalidate(request: Request, call_next):
    """Code and styles revalidate on every load (cheap: ETag -> 304), so a deploy is never half-cached."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/") and not request.url.path.endswith(".gz"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


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
    return {"user": auth.public(u), "local": config.LOCAL_MODE, "missing_scopes": (u or {}).get("missing_scopes", []),
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
    return v


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
class RunRequest(BaseModel):
    task_id: str
    model: str
    provider: str | None = None
    judge: str | None = None


@app.post("/api/runs")
def start_run(body: RunRequest, request: Request):
    u = auth.require_user(request)
    v = catalog.view(body.task_id)
    if not v or not v.get("runnable"):
        raise HTTPException(400, "this task cannot be run here")
    m = models.get(body.model)
    if not m or not m["tools"]:
        raise HTTPException(400, "pick a model that supports tool calling")
    provider = body.provider or m["provider"]
    need = (v.get("verify") or {}).get("needs_judge")
    judge = body.judge if need else None
    if need and not judge:
        raise HTTPException(400, "this task is graded by a model: pick a judge")
    try:
        run = core.submit(u["name"], u["token"], {"id": v["id"], "domain": v["domain"], "title": v["short_title"],
                                                  "facets": v.get("facets")}, body.model, provider, judge)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return run


@app.get("/api/runs")
def list_runs(request: Request, task_id: str | None = None):
    u = auth.current_user(request)
    if not u:
        return {"runs": []}
    return {"runs": store.list_runs(user=u["name"], task_id=task_id)}


def _own(request: Request, run_id: str) -> dict:
    u = auth.current_user(request)
    try:
        run = store.get(run_id)
    except ValueError:
        run = None
    if not run or not u or run.get("user") != u["name"]:
        raise HTTPException(404, "no such rollout")
    return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request, after: int = 0):
    run = _own(request, run_id)
    live = core.live(run_id)
    if live is not None:
        run = {**run, **live.run, "cost": live.cost_now()}
    return {"run": run, "events": core.events(run_id, after), "live": live is not None}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request):
    _own(request, run_id)
    return {"cancelled": core.cancel(run_id)}


@app.get("/api/runs/{run_id}/artifacts/{name}")
def artifact(run_id: str, name: str, request: Request):
    _own(request, run_id)
    data = store.read_artifact(run_id, name)
    if data is None:
        raise HTTPException(404, "no such artifact")
    return Response(data, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream")


# ── the site ─────────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def _nf(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(exc, "detail", "not found")}, 404)
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
