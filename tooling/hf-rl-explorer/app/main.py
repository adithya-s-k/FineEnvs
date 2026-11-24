"""HF RL Explorer: browse RL environment datasets in Harbor's format, see how each task is graded, run agents on them.

    uv run uvicorn app.main:app --reload      # local: your HF token, data in ./.local-data
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import os
from typing import Literal
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from . import auth, catalog, config, endpoints, envs, http, mcp_bridge, models, runner, runtime, search_api, seo, settings, space_files, spaces_live, store, version
from .envs import registry
from . import community as community_data

app = FastAPI(title="HF RL Explorer", docs_url="/api/docs")
app.include_router(auth.router)
app.include_router(mcp_bridge.router)
app.include_router(space_files.router)
app.include_router(search_api.router)
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)   # the listing is a few hundred KB of JSON


http.install(app)


@app.on_event("startup")
async def _runtime() -> None:
    """Timeouts on Hub calls, room in the thread pool, JSON logs, the disk janitor (app/runtime.py)."""
    if os.environ.get("RLX_TEST_TMP"):   # tests: no janitor, no log takeover
        runtime.hub_timeouts()
        runtime.more_threads()
        return
    runtime.setup()


@app.on_event("shutdown")
def _shutdown() -> None:
    """Live rollouts' events to the store, and locally the capture proxy's tunnels closed: gradio starts `frpc` as a child
    that outlives this process otherwise, and stale tunnels pile up across restarts."""
    import sys
    from . import auto_index, space_checks

    space_checks.stop()
    auto_index.stop()

    store.flush_due(force=True)
    tunneling = sys.modules.get("gradio.tunneling")
    for t in list(getattr(tunneling, "CURRENT_TUNNELS", []) or []):
        try:
            t.kill()
        except Exception:  # noqa: BLE001 - best effort, on the way out
            pass


@app.on_event("startup")
def _startup() -> None:
    import threading
    from . import auto_index, space_checks

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    space_checks.start()
    auto_index.start()
    # rollouts whose worker is gone are marked interrupted by the watcher below, once they've been silent a while
    threading.Thread(target=runner.watch_cancel_requests, daemon=True, name="cancel-requests").start()
    if os.environ.get("RLX_WARM", "1") == "1":
        threading.Thread(target=catalog.warm, daemon=True, name="warm").start()
        from .mimo import catalog as mimo_catalog   # the MiMo release's rows and file listing, so its pages open fast

        threading.Thread(target=mimo_catalog.warm, daemon=True, name="warm-mimo").start()


# ── health ───────────────────────────────────────────────────────────────────
@app.get("/healthz", include_in_schema=False)
def healthz():
    """Alive: the process answers."""
    return {"ok": True}


@app.get("/readyz", include_in_schema=False)
def readyz():
    """Ready to serve: threads to spare, disk, the store writable, how fresh the indexes are."""
    ok, checks = runtime.readiness()
    return JSONResponse({"ok": ok, **checks}, status_code=200 if ok else 503)


# ── who ──────────────────────────────────────────────────────────────────────
@app.get("/api/me")
def me(request: Request):
    u = auth.current_user(request)
    return {"user": auth.public(u), "local": config.LOCAL_MODE, "oauth": not config.LOCAL_MODE,
            "version": version.app_version(), "source": version.source_hash(),
            "missing_scopes": (u or {}).get("missing_scopes", []),
            "rollouts_enabled": settings.get("rollouts_enabled", True), "announcement": settings.get("announcement", ""),
            "billing": _billing(u),
            "storage": "local folder" if not config.STORAGE_DIR.as_posix().startswith("/data") else "private bucket"}


CREDIT_REFUSED = ("no prepaid credit", "402")


def _billing(u: dict | None) -> dict | None:
    """Whether this account looks able to pay for sandboxes, so the run panel can say so before a run, not after.
    `can_pay` is the Hub's own flag; `refused_at` is when this account's latest sandbox rollout was turned away for
    missing credit (the Hub's flag can't see an empty prepaid balance), cleared by any later sandbox that started."""
    if not u:
        return None
    try:
        b = auth.billing(u)
    except Exception:  # noqa: BLE001 - unknown: say nothing rather than something wrong
        b = {}
    refused = None
    for r in store.list_runs(user=u["name"], limit=30):
        if r.get("domain") == "music":   # no sandbox
            continue
        if any(x in (r.get("error") or "") for x in CREDIT_REFUSED):
            refused = r.get("created_at")
        break
    return {**b, "refused_at": refused}


# ── datasets ─────────────────────────────────────────────────────────────────
def _hub_errors(fn, *args):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (RepositoryNotFoundError, PermissionError) as e:
        raise HTTPException(404, "No such dataset, or it is private: sign in with an account that can read it."
                            if isinstance(e, RepositoryNotFoundError) else str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:   # busy: too many datasets indexing at once
        raise HTTPException(429, str(e))


@app.get("/api/environments")
def environments():
    """Every Harbor dataset on the Hub and the featured collections; the page sorts and filters them."""
    try:
        rows = catalog.environments()
    except Exception as e:  # noqa: BLE001 - the Hub is down or rate limiting
        raise HTTPException(502, f"could not reach the Hub: {type(e).__name__}")
    counts: dict[str, int] = {}
    for r in store.list_runs(public=True, limit=100000):   # graded public rollouts per dataset
        if _shareable(r):
            counts[r.get("dataset")] = counts.get(r.get("dataset"), 0) + 1
    return {"datasets": rows, "collections": catalog.collections(), "rollouts": counts}


@app.get("/api/spaces/{org}/{name}")
def space_page(org: str, name: str):
    """Repository metadata is distinct from verified live OpenEnv support."""
    from . import space_checks
    source = _hub_errors(catalog.space, f"{org}/{name}")
    check = space_checks.inventory().get(f"{org}/{name}", {})
    verified = space_checks.verified(check)
    return {**source, "declared_openenv": bool(source.get("openenv")), "openenv": verified,
            "framework": "openenv" if verified else "ors" if source.get("framework") == "ors" else "space",
            "badges": [b for b in source.get("badges", []) if verified or b != "OpenEnv"],
            "api_status": space_checks.status(check)}


# ── live Spaces: status, wake/restart, Task API, playground ─────────────────
def _live(fn, *args):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return fn(*args)
    except spaces_live.SpaceError as e:
        raise HTTPException(e.status, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (RepositoryNotFoundError, PermissionError):
        raise HTTPException(404, "No such Space, or it is private.")


@app.get("/api/spaces/{org}/{name}/live")
def space_live(org: str, name: str, fresh: bool = False):
    """The Space's stage now and, when it's running, what its server offers (UI, MCP tools, Task API, schemas). A server
    with no MCP tools of its own (a reset/step environment) still has tools for an agent: `bridge_tools` are exactly what
    this explorer's MCP bridge offers for it (reset, step with its typed action, state, its Task API), for the page."""
    info = _live(spaces_live.probe, f"{org}/{name}", fresh)
    if info.get("running") and (info.get("mcp") or info.get("step_api") or info.get("api") or info.get("task_api")):
        info = {**info, "bridge_tools": mcp_bridge._tools(info)}
    return info


@app.post("/api/spaces/{org}/{name}/wake")
def space_wake(org: str, name: str, request: Request):
    """Wake a sleeping Space, as its Hub page lets any visitor do."""
    if mcp_bridge._limited("web:" + http.client_ip(request)):
        raise HTTPException(429, "too many requests: slow down")
    return _live(spaces_live.wake, f"{org}/{name}")


@app.post("/api/spaces/{org}/{name}/restart")
def space_restart(org: str, name: str, request: Request):
    """Restart a Space with the visitor's own token (works when they may write to it)."""
    u = auth.require_user(request)
    if mcp_bridge._limited("restart:" + u["name"]):
        raise HTTPException(429, "too many requests: slow down")
    return _live(spaces_live.restart, f"{org}/{name}", u["token"])


@app.get("/api/spaces/{org}/{name}/tasks")
def space_tasks(org: str, name: str, split: str = "", start: int = 0, stop: int = 20, env: str = ""):
    """A page of tasks from the server's Task API (answer-like fields left out)."""
    return _live(spaces_live.tasks, f"{org}/{name}", split, start, stop, env)


@app.get("/api/spaces/{org}/{name}/task")
def space_task(org: str, name: str, split: str = "", index: int = 0, env: str = ""):
    return _live(spaces_live.task, f"{org}/{name}", split, index, env)


class PlayIn(BaseModel):
    op: Literal["start", "end", "reset", "step", "state", "call", "tools", "http"]
    session: str | None = Field(None, max_length=64)
    data: dict | None = None


@app.post("/api/spaces/{org}/{name}/play")
def space_play(org: str, name: str, body: PlayIn, request: Request):
    """The playground: start a session on the Space, then reset/step it (or call its tools), then end it."""
    u = auth.current_user(request)
    owner = "web:" + http.client_ip(request) + (":" + u["name"] if u else "")
    if mcp_bridge._limited(owner):
        raise HTTPException(429, "too many requests: slow down")
    if len(json.dumps(body.data or {})) > 256 * 1024:
        raise HTTPException(413, "that action is too large")
    spec = f"{org}/{name}"
    if body.op == "start":
        return _live(spaces_live.start, spec, owner)
    if not body.session:
        raise HTTPException(400, "start a session first")
    if body.op == "end":
        spaces_live.end(body.session, owner)
        return {"ok": True}
    s = spaces_live._sessions.get(body.session)
    if s is not None and s.spec != catalog.check_spec(spec):
        raise HTTPException(400, "that session belongs to another Space")
    return _live(spaces_live.act, body.session, owner, body.op, body.data or {})


@app.get("/api/environments/mine")
def my_environments(request: Request):
    """The signed-in visitor's own Harbor datasets and their organizations', private ones included."""
    u = auth.require_user(request)
    try:
        return {"datasets": catalog.mine(u["token"])}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach the Hub: {type(e).__name__}")


def _token(request: Request) -> str | None:
    """The visitor's token, for the private and gated datasets they can read; never the server's own."""
    return (auth.current_user(request) or {}).get("token")


@app.get("/api/random")
def random_task(request: Request, d: str = ""):
    return _hub_errors(catalog.random_task, d or None, _token(request))


# ── environments: any format, through its adapter (app/envs) ─────────────────
def _env(fn, *args, **kw):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return fn(*args, **kw)
    except (envs.ViewerError, envs.DirectError) as e:
        # the dataset viewer or the direct reader failed: say which and why in the logs (never the arguments: a token)
        logging.getLogger("rlx").warning("env %s %s: %s: %s", getattr(fn, "__name__", "?"), args[0] if args else "", type(e).__name__, str(e)[:300])
        raise HTTPException(getattr(e, "status", 502), str(e))
    except (RepositoryNotFoundError, PermissionError):
        raise HTTPException(404, "No such dataset, or it is private: sign in with an account that can read it.")
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:   # busy: too many datasets indexing at once
        raise HTTPException(429, str(e))


def _filters(f: str) -> dict[str, list[str]]:
    try:
        raw = json.loads(f) if f else {}
        if not isinstance(raw, dict) or len(raw) > 12:
            raise ValueError
        return {str(k): [str(x) for x in (v if isinstance(v, list) else [v])][:30] for k, v in raw.items()}
    except (ValueError, TypeError):
        raise HTTPException(400, "f should be a JSON object of facet: [values]")


@app.get("/api/env/{org}/{name}")
def env_summary(org: str, name: str, request: Request, subset: str = ""):
    """An environment at a glance, whatever its format: how it's read, its subsets and facets (of `subset`), how many
    tasks, and an overview (or, for a Harbor dataset being indexed, the indexing's progress)."""
    return _env(registry.summary, f"{org}/{name}", _token(request), subset or None)


@app.get("/api/env/{org}/{name}/tasks")
def env_tasks(org: str, name: str, request: Request, subset: str = "", q: str = "", f: str = "", offset: int = 0, all: bool = False):
    """Its tasks as cards: a page of them (a search, a filter by facets), or every one (inline environments)."""
    return _env(registry.tasks, f"{org}/{name}", _token(request), subset=subset or None, q=q[:200], filters=_filters(f),
                offset=max(0, offset), everything=all)


@app.get("/api/env/{org}/{name}/task")
def env_task(org: str, name: str, request: Request, ref: str):
    """One task in full: its sections (the task, grading, environment, files, ...), how it can run, links."""
    return _env(registry.task, f"{org}/{name}", ref, _token(request))


@app.get("/api/env/{org}/{name}/file")
def env_file(org: str, name: str, request: Request, ref: str, f: str = ""):
    return _env(registry.file, f"{org}/{name}", ref, f, _token(request))


@app.get("/api/env/{org}/{name}/folder")
def env_folder(org: str, name: str, request: Request, ref: str, f: str = ""):
    """One folder of a task, listed when it's opened in the file viewer."""
    return _env(registry.folder, f"{org}/{name}", ref, f, _token(request))


@app.get("/api/env/{org}/{name}/data")
def env_data(org: str, name: str, request: Request, ref: str, part: str):
    """More of a task for a custom view: a database table's rows, a file's preview (the adapter's `data`)."""
    params = {k: v[:500] for k, v in request.query_params.items() if k not in ("ref", "part")}
    return _env(registry.data, f"{org}/{name}", ref, part, params, _token(request))


@app.get("/api/env/{org}/{name}/raw")
def env_raw(org: str, name: str, request: Request, ref: str, f: str, download: bool = False):
    """One of a task's files as it is (an image, a page for a sandboxed frame). Never runs with this site's origin."""
    data, media = _env(registry.raw, f"{org}/{name}", ref, f, _token(request))
    # RFC 5987 handles Unicode and keeps control characters out of HTTP headers.
    fname = quote(f.rsplit("/", 1)[-1], safe="")
    return Response(data, media_type=media, headers={"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff",
                                                     "Content-Disposition": f"{'attachment' if download else 'inline'}; filename*=UTF-8''{fname}"})


@app.get("/api/env/{org}/{name}/random")
def env_random(org: str, name: str, request: Request, subset: str = ""):
    return {"ref": _env(registry.random, f"{org}/{name}", subset or None, _token(request))}


@app.get("/api/env/{org}/{name}/runs")
def env_runs(org: str, name: str, request: Request, ref: str):
    """A task's rollouts: yours, and everyone's public ones, wherever the same task was run (its aliases)."""
    spec, token = f"{org}/{name}", _token(request)
    env = _env(registry.resolve, "dataset", spec, token)
    keys = {(spec, ref), *_env(registry.aliases, env, ref)}
    u = auth.current_user(request)
    mine = [r for r in store.list_runs(user=u["name"], limit=2000) if (r.get("dataset"), r.get("path")) in keys] if u else []
    ids = {r["id"] for r in mine}
    public = [r for r in store.list_runs(public=True, limit=100000) if _shareable(r) and (r.get("dataset"), r.get("path")) in keys and r["id"] not in ids]
    return {"mine": [_owner_view(r) for r in mine], "public": [_public_view(r) for r in public[:100]],
            "elsewhere": [{"env": k, "ref": r} for k, r in keys if k != spec or r != ref]}


# ── rollouts ─────────────────────────────────────────────────────────────────
@app.get("/api/models")
def model_list():
    """Models on Inference Providers that can call tools, the agents, and what a sandbox costs."""
    try:
        cat = models.catalog()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not list models: {type(e).__name__}")
    return {"models": cat["agents"], "default_model": cat["default_agent"], "agents": runner.AGENTS,
            "default_agent": "opencode", "sandbox_price_per_hour": config.FLAVOR_PRICE_PER_HOUR[runner.FLAVOR],
            "prices": config.FLAVOR_PRICE_PER_HOUR, "text_judges": cat["text_judges"], "vision_judges": cat["vision_judges"]}


class EndpointIn(BaseModel):
    base_url: str = Field(max_length=300)
    model: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(None, max_length=500)
    price_in: float | None = Field(None, ge=0, le=1000)    # $ per 1M tokens, only for the cost shown
    price_out: float | None = Field(None, ge=0, le=1000)


class Params(BaseModel):
    steps: int | None = Field(None, ge=1, le=1000)
    timeout_min: int = Field(30, ge=2, le=120)


class RunRequest(BaseModel):
    dataset: str = Field(max_length=200)
    path: str = Field("", max_length=500)
    runner: str = Field("harbor", max_length=40, pattern=r"^[a-z0-9-]+$")   # how the task runs: one of the task's run options
    fields: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=20)   # that option's own inputs
    harness: str = "opencode"
    model: str | None = Field(None, max_length=200)
    endpoint: EndpointIn | None = None
    params: Params = Params()
    visibility: Literal["public", "private"] = "public"


class EndpointProbe(BaseModel):
    base_url: str = Field(max_length=300)
    api_key: str | None = Field(None, max_length=500)
    model: str | None = Field(None, max_length=200)


_probes: dict[str, list[float]] = {}


def _probe_limit(user: str) -> None:
    """This server makes the call, so cap how often one account can have it do so."""
    now = time.time()
    hits = [t for t in _probes.get(user, []) if now - t < 600]
    if len(hits) >= 30:
        raise HTTPException(429, "Too many endpoint checks. Wait a few minutes.")
    _probes[user] = hits + [now]


@app.post("/api/endpoints/models")
def endpoint_models(body: EndpointProbe, request: Request):
    import httpx

    _probe_limit(auth.require_user(request)["name"])
    try:
        return {"models": endpoints.list_models(body.base_url, body.api_key)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Couldn't reach the endpoint: {type(e).__name__}")


@app.post("/api/endpoints/test")
def endpoint_test(body: EndpointProbe, request: Request):
    import httpx

    _probe_limit(auth.require_user(request)["name"])
    try:
        return endpoints.test(endpoints.check_url(body.base_url), body.api_key, body.model or "")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Couldn't reach the endpoint: {type(e).__name__}")


@app.post("/api/runs")
def start_run(body: RunRequest, request: Request):
    u = auth.require_user(request)
    if body.harness not in runner.AGENT_IDS:
        raise HTTPException(400, "unknown agent")
    endpoint = key = provider = model = None
    if body.endpoint:
        try:
            base = endpoints.check_url(body.endpoint.base_url)
        except ValueError as e:
            raise HTTPException(400, str(e))
        endpoint, key = ({"base_url": base, "model": body.endpoint.model, "host": urlparse(base).hostname, "price_in": body.endpoint.price_in,
                          "price_out": body.endpoint.price_out}, body.endpoint.api_key)
    else:
        m = models.get(body.model or models.DEFAULT_AGENT)
        if not m or not m.get("tools"):
            raise HTTPException(400, "pick a model that can call tools")
        # the router's fastest live provider for this model: the cheapest is often the slowest, or overloaded
        model, provider = m["id"], "fastest"
    try:
        run = runner.submit(u, body.dataset, body.path, body.harness, model, provider, endpoint, key,
                            body.params.model_dump(), body.visibility, body.runner, body.fields)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except (ValueError, LookupError) as e:
        raise HTTPException(400, str(e))
    except (envs.ViewerError, envs.DirectError) as e:
        raise HTTPException(502, f"couldn't read that row: {e}")
    return {"run": _owner_view(run)}


# what anyone may see of a public rollout: no user, no sandbox, no endpoint location
PUBLIC_FIELDS = ("id", "dataset", "path", "task_id", "title", "collection", "difficulty", "category", "model", "provider",
                 "harness", "sandbox", "flavor", "image", "params", "status", "phase", "reward", "rewards", "reward_key",
                 "graded", "n_turns", "wall_s", "cost", "created_at", "started_at", "finished_at", "phase_timings", "error", "note",
                 "replayed", "row", "env", "runner", "adapter", "domain", "facets", "judge", "tokens", "reward_error", "provenance",
                 "fields")


def _shareable(r: dict) -> bool:
    """Shown to everyone: finished and graded, on a public dataset, and not taken off Community by an admin."""
    return community_data.is_public(r, set(settings.get("hidden_runs", [])))


def _public_view(r: dict) -> dict:
    v = {k: r.get(k) for k in PUBLIC_FIELDS}
    v["runner"] = community_data.runner(r)
    if r.get("endpoint"):
        v["endpoint"] = {"custom": True}
    return {**v, "visibility": "public"}


def _owner_view(r: dict) -> dict:
    return {**r, "runner": community_data.runner(r), "is_owner": True}


def _visible(request: Request, run_id: str) -> tuple[dict, bool]:
    r = store.get(run_id)
    if not r:
        raise HTTPException(404, "no such rollout")
    u = auth.current_user(request)
    if u and u.get("name") == r.get("user"):
        return r, True
    if r.get("visibility") == "public" and _shareable(r):
        return r, False
    raise HTTPException(404, "no such rollout")


def _scrub(text: str, r: dict) -> str:
    """A public trajectory, without who ran it or where their endpoint is."""
    for s, sub in ((r.get("user"), "[user]"), ((r.get("endpoint") or {}).get("host"), "[endpoint]"),
                   ((r.get("endpoint") or {}).get("base_url"), "[endpoint]")):
        if s and len(s) >= 3:
            text = text.replace(s, sub)
    return text


@app.get("/api/runs")
def my_runs(request: Request, d: str = "", p: str | None = None):
    u = auth.require_user(request)
    runs = store.list_runs(user=u["name"], task_id=f"{d}:{p}" if d and p is not None else None, limit=500)
    return {"runs": [_owner_view(r) for r in runs], "live": [r["id"] for r in runs if runner.is_live(r["id"])]}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str, request: Request, after: int = 0):
    r, owner = _visible(request, run_id)
    if community_data.runner(r) in ("mimo", "nemo-gym"):   # event-based hosted runners
        from .mimo.runner import core as mimo

        live = mimo.live(run_id)
        if live is not None:
            r = {**r, **live.run, "cost": live.cost_now()}
        events = mimo.events(run_id, max(0, after))
        stream = dict(live.stream, age=round(time.time() - live.stream["since"], 1)) if live is not None and live.stream else None
        if owner:
            return {"run": _owner_view(r), "events": events, "live": live is not None, "stream": stream}
        return {"run": _public_view(r), "events": json.loads(_scrub(json.dumps(events), r)), "live": live is not None, "stream": stream}
    raw = store.read_artifact(run_id, "trajectory.json")
    text = raw.decode() if raw else "[]"
    if not owner:
        text = _scrub(text, r)
    return {"run": _owner_view(r) if owner else _public_view(r), "trajectory": json.loads(text), "live": runner.is_live(run_id)}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request):
    r, owner = _visible(request, run_id)
    if not owner:
        raise HTTPException(403, "not your rollout")
    return {"cancelled": runner.cancel(run_id)}


class VisibilityIn(BaseModel):
    visibility: Literal["public", "private"]


@app.post("/api/runs/{run_id}/visibility")
def set_visibility(run_id: str, body: VisibilityIn, request: Request):
    r, owner = _visible(request, run_id)
    if not owner:
        raise HTTPException(403, "not your rollout")
    if body.visibility == "public" and r.get("restricted"):
        raise HTTPException(400, "rollouts on a private or gated dataset stay private")
    return {"run": _owner_view(store.update(run_id, visibility=body.visibility))}


@app.get("/api/community")
def community(dataset: str = "", group: str = "", model: str = "", served: str = "", judge: str = "", reward: str = "", thinking: str = "",
              runner: str = "", q: str = "", sort: str = "new", d: str = "", p: str | None = None, limit: int = 100, offset: int = 0):
    """Finished, graded public rollouts from any saved runner, without account identities or private endpoint locations."""
    everyone = [r for r in store.list_runs(public=True, limit=100000) if _shareable(r)]
    filters = {"dataset": dataset or d, "path": p, "group": group, "model": model, "served": served, "judge": judge,
               "reward": reward, "thinking": thinking, "runner": runner, "q": q[:200]}
    out = community_data.query(everyone, filters, sort=sort, offset=offset, limit=limit)
    out["runs"] = [{**_public_view(r), "runner": community_data.runner(r), "group": community_data.group(r),
                    "shot": r.get("domain") == "webdev" and store.has_artifact(r["id"], "screenshot.jpg")} for r in out["runs"]]
    return out


@app.get("/api/community/tasks")
def community_tasks(dataset: str):
    """One environment's tasks that have public rollouts: how many, and how they scored. Tasks absent have none yet."""
    out: dict[str, dict] = {}
    for r in store.list_runs(public=True, limit=100000):
        if not _shareable(r) or r.get("dataset") != dataset:
            continue
        t = out.setdefault(r.get("path") or "", {"runs": 0, "scored": 0, "sum": 0.0, "best": None, "last": 0, "title": r.get("title")})
        t["runs"] += 1
        t["last"] = max(t["last"], r.get("created_at") or 0)
        if r.get("reward") is not None:
            t["scored"] += 1
            t["sum"] += float(r["reward"])
            t["best"] = max(t["best"], float(r["reward"])) if t["best"] is not None else float(r["reward"])
    return {"tasks": {k: {"runs": v["runs"], "mean": round(v["sum"] / v["scored"], 4) if v["scored"] else None, "best": v["best"], "last": v["last"],
                          "title": v["title"]} for k, v in out.items()}}


@app.get("/api/runs/{run_id}/artifacts/{name}")
def run_artifact(run_id: str, name: str, request: Request):
    """A file a rollout left: its owner may have any; anyone else, only its images (a webdev page's screenshot), since
    the rest (trajectory, logs, results) is served scrubbed through /api/runs/{id}."""
    r, owner = _visible(request, run_id)
    if not re.fullmatch(r"[\w.-]{1,80}", name) or name in ("run.json", "events.jsonl"):
        raise HTTPException(404, "no such artifact")
    if not owner and not re.search(r"\.(jpe?g|png|webp)$", name, re.I):
        raise HTTPException(404, "no such artifact")
    data = store.read_artifact(run_id, name)
    if data is None:
        raise HTTPException(404, "no such artifact")
    return Response(data, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                    headers={"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"})


# ── the model proxy, for the MiMo harness ────────────────────────────────────
# On the Space, a MiMo sandbox holds no credential: OpenCode (and the General verifier's judge) call this route with a
# per-rollout capability in the path. It works only while that rollout is live, only for the rollout's own agent model
# and judge, and this server adds the visitor's token or endpoint key on the way out.
LLM_REPLY_MAX = 64 * 1024 * 1024


@app.get("/api/llm/{cap}/health")
def llm_health(cap: str):
    from .mimo.runner import core as events
    if events.by_cap(cap) is None:
        raise HTTPException(404, "unknown or finished rollout")
    return {"status": "ok"}


@app.post("/api/llm/{cap}/v1/chat/completions")
async def llm_proxy(cap: str, request: Request):
    import asyncio

    import httpx
    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse

    from .mimo.runner import core as mimo

    r = mimo.by_cap(cap)
    if r is None:
        return JSONResponse({"error": {"message": "unknown or finished rollout"}}, 404)
    raw = await request.body()
    if len(raw) > 20_000_000:
        return JSONResponse({"error": {"message": "request too large"}}, 413)
    try:
        body = json.loads(raw)
    except ValueError:
        return JSONResponse({"error": {"message": "invalid JSON"}}, 400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "JSON object required"}}, 400)
    up = r.upstream(str(body.get("model") or ""))
    if up is None:
        return JSONResponse({"error": {"message": "this rollout may not call that model"}}, 403)
    base, key = up
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    ext = {}
    if r.run.get("endpoint") and base != config.ROUTER:   # the visitor's own endpoint: connect to the address we checked
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
        # the agent reports a step only once the whole reply is in; counting what streams past lets the rollout page
        # show the model is still writing, and how much
        s = r.stream = {"since": time.time(), "text": 0, "thinking": 0, "tool": 0}
        buf = b""
        total = 0
        try:
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > LLM_REPLY_MAX:   # no model reply is this big: an endpoint streaming without end
                    break
                yield chunk
                buf += chunk
                if len(buf) > 1_000_000:   # one "line" this long isn't an SSE event: stop counting, keep relaying
                    buf = b""
                    continue
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: {"):
                        continue
                    try:
                        choices = json.loads(line[6:]).get("choices") or []
                    except ValueError:
                        continue
                    for ch in choices:
                        d = ch.get("delta") or {}
                        s["text"] += len(d.get("content") or "")
                        s["thinking"] += len(d.get("reasoning_content") or d.get("reasoning") or "")
                        s["tool"] += sum(len((t.get("function") or {}).get("arguments") or "") for t in d.get("tool_calls") or [])
        finally:
            if r.stream is s:
                r.stream = None

    async def close():
        await resp.aclose()
        await client.aclose()
    return StreamingResponse(relay(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"),
                             background=BackgroundTask(close))


# ── the page ─────────────────────────────────────────────────────────────────
# Every page at its own path, with its own title, description, canonical address, preview and structured data
# (app/seo.py); older addresses move there for good.
def _spec_or_404(org: str, name: str) -> str:
    try:
        return catalog.check_spec(f"{org}/{name}")
    except ValueError:
        raise HTTPException(404, "no such environment")


@app.get("/", include_in_schema=False)
def index(request: Request):
    return seo.home(request)


@app.get("/d/{org}/{name}", include_in_schema=False)
def page_env(org: str, name: str, request: Request):
    return seo.environment(request, _spec_or_404(org, name))


@app.get("/t/{org}/{name}/{ref:path}", include_in_schema=False)
def page_task(org: str, name: str, ref: str, request: Request):
    return seo.task(request, _spec_or_404(org, name), ref)


@app.get("/s/{org}/{name}", include_in_schema=False)
def page_space(org: str, name: str, request: Request):
    return seo.space(request, _spec_or_404(org, name))


@app.get("/community", include_in_schema=False)
@app.get("/runs", include_in_schema=False)
@app.get("/run/{run_id}", include_in_schema=False)
@app.get("/compare/{org}/{name}/{ref:path}", include_in_schema=False)
def page_other(request: Request):
    return seo.simple(request, request.url.path)


@app.get("/r/{org}/{name}", include_in_schema=False)
def page_row(org: str, name: str, c: str = "default", s: str = "train", i: int = 0):
    """A rows dataset's row, at its old address: its task page."""
    ref = f"{'' if c == 'default' else quote(c, safe='') + '/'}{quote(s, safe='')}/{max(0, i)}"
    return RedirectResponse(f"/t/{seo.enc(_spec_or_404(org, name))}/{ref}", status_code=301)


@app.get("/task/{task_id}", include_in_schema=False)
@app.get("/compare/{task_id}", include_in_schema=False)
def page_mimo_old(task_id: str, request: Request):
    """The MiMo RL Environment Explorer's addresses."""
    what = "t" if request.url.path.startswith("/task/") else "compare"
    qs = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"/{what}/{seo.MIMO}/{quote(task_id, safe='')}{qs}", status_code=301)


@app.get("/rewards", include_in_schema=False)
def page_rewards():
    return RedirectResponse(f"/d/{seo.MIMO}?rewards=1", status_code=301)


@app.get("/robots.txt", include_in_schema=False)
def robots(request: Request):
    return seo.robots(request)


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap(request: Request):
    return seo.sitemap_index(request)


@app.get("/sitemap-pages.xml", include_in_schema=False)
def sitemap_pages(request: Request):
    return seo.sitemap_pages(request)


@app.get("/sitemap-tasks-{n}.xml", include_in_schema=False)
def sitemap_tasks(n: int, request: Request):
    return seo.sitemap_tasks(request, n)


@app.get("/sitemap-space-tasks-{n}.xml", include_in_schema=False)
def sitemap_space_tasks(n: int, request: Request):
    return seo.sitemap_space_tasks(request, n)


@app.get("/og.png", include_in_schema=False)
def og_site():
    return seo.og_image("/")


@app.get("/og/{path:path}", include_in_schema=False)
def og_page(path: str, request: Request):
    if not path.endswith(".png") or len(path) > 600:
        raise HTTPException(404, "no such image")
    return seo.og_image("/" + path[:-4], request.query_params)


@app.exception_handler(404)
async def _not_found(request: Request, exc):
    """An address with nothing at it: JSON for the API and files, the app's own 404 page (not indexed) for pages."""
    path = request.url.path
    if path.startswith(("/api/", "/mcp/", "/capture")) or "." in path.rsplit("/", 1)[-1]:
        return JSONResponse({"detail": getattr(exc, "detail", "Not Found")}, 404)
    return seo.not_found(request)


if config.SPACE_HOST:   # the sandbox reaches the model through this app's own public URL
    app.mount("/capture", runner.capture_app())
app.mount("/", StaticFiles(directory=config.WEB_DIR), name="web")
