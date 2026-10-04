"""The admin dashboard's API, served by its own app (app/admin_app.py) on its own private Space, for members of the
admin organization only (see app/settings.py). Every route checks the signed-in user against the organization on
every request. Every change is written to the audit log with who made it.

The admin app shares the explorer's bucket but not its process. It reads the run records the explorer writes, and
what it changes goes into the shared settings file, which the explorer re-reads: collections, pins, hidden
environments, the rollout switches, rollouts taken off Community, and requests to stop a running rollout.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from collections import Counter
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import auth, catalog, config, runner, settings, store

STALE = 600   # a rollout whose record hasn't changed for this long is no longer live, whatever it says

router = APIRouter(prefix="/api/admin")
KEY = re.compile(r"^(space:)?[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _admin(request: Request) -> dict[str, Any]:
    u = auth.current_user(request)
    if not u:
        raise HTTPException(401, "sign in")
    if not settings.is_admin(u):
        raise HTTPException(403, f"only members of {settings.ADMIN_ORG} can open the admin dashboard")
    return u


# ── overview ─────────────────────────────────────────────────────────────────
@router.get("/overview")
def overview(request: Request):
    _admin(request)
    envs = catalog.environments(include_hidden=True)
    runs = _runs()
    now = time.time()
    day = 86400
    by_day = Counter(time.strftime("%Y-%m-%d", time.gmtime(r.get("created_at", 0))) for r in runs if now - r.get("created_at", 0) < 14 * day)
    finished = [r for r in runs if r.get("status") in ("done", "failed", "cancelled", "interrupted")]
    new = sorted((e for e in envs if e.get("created") and now - _ts(e["created"]) < 14 * day), key=lambda e: e["created"], reverse=True)
    idx_files = list(config.INDEX_DIR.glob("*.json.gz"))
    pack_files = list(config.PACK_DIR.glob("*.json.gz")) if config.PACK_DIR.is_dir() else []
    return {
        "environments": {"datasets": sum(1 for e in envs if e["kind"] == "dataset"), "spaces": sum(1 for e in envs if e["kind"] == "space"),
                         "openenv": sum(1 for e in envs if e.get("openenv")), "indexed": sum(1 for e in envs if e.get("indexed")),
                         "hidden": len(catalog.hidden()), "pinned": len(catalog.pinned()), "collections": len(catalog.collections())},
        "rollouts": {"total": len(runs), "live": sum(1 for r in runs if _live(r)), "today": sum(1 for r in runs if now - r.get("created_at", 0) < day),
                     "week": sum(1 for r in runs if now - r.get("created_at", 0) < 7 * day),
                     "public": sum(1 for r in runs if r.get("visibility") == "public"),
                     "failed": sum(1 for r in finished if r.get("status") != "done"), "finished": len(finished),
                     "users": len({r.get("user") for r in runs}),
                     "by_day": [[time.strftime("%Y-%m-%d", time.gmtime(now - i * day)), by_day.get(time.strftime("%Y-%m-%d", time.gmtime(now - i * day)), 0)]
                                for i in range(13, -1, -1)],
                     "by_model": Counter(r.get("model") or "custom endpoint" for r in runs).most_common(8),
                     "by_status": Counter(r.get("status") for r in runs).most_common()},
        "storage": {"indexes": len(idx_files), "index_bytes": sum(f.stat().st_size for f in idx_files),
                    "packs": len(pack_files), "pack_bytes": sum(f.stat().st_size for f in pack_files)},
        "recent": [_admin_view(r) for r in runs[:8]],
        "new": new[:12],
        "settings": _settings(),
    }


def _ts(iso: str) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


_avatars: dict[str, tuple[float, str | None]] = {}


def avatar(name: str | None) -> str | None:
    """A Hub user's profile picture, looked up once a day (rollouts started before avatars were recorded have none)."""
    if not name:
        return None
    hit = _avatars.get(name)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    import httpx

    url = None
    try:
        r = httpx.get(f"https://huggingface.co/api/users/{name}/overview", timeout=6)
        if r.status_code == 200:
            url = r.json().get("avatarUrl")
            url = url if isinstance(url, str) and url.startswith("https://") else None
    except Exception:  # noqa: BLE001
        pass
    _avatars[name] = (time.time(), url)
    return url


def _live(r: dict[str, Any]) -> bool:
    return r.get("status") in store.ACTIVE and time.time() - (r.get("updated_at") or 0) < STALE


def _runs() -> list[dict[str, Any]]:
    store.refresh(10)   # the explorer writes these; re-read them every few seconds
    return store.list_runs(limit=100000)


def _admin_view(r: dict[str, Any]) -> dict[str, Any]:
    """A rollout for admins: everything but where a visitor's own endpoint lives."""
    v = dict(r)
    if v.get("endpoint"):
        v["endpoint"] = {"host": v["endpoint"].get("host"), "model": v["endpoint"].get("model")}
    v["live"] = _live(r)
    v["avatar"] = r.get("avatar") or avatar(r.get("user"))
    v["moderated"] = r["id"] in set(settings.get("hidden_runs", []))
    return v


# ── rollouts ─────────────────────────────────────────────────────────────────
@router.get("/rollouts")
def rollouts(request: Request, status: str = "", q: str = "", user: str = "", limit: int = 100, offset: int = 0):
    _admin(request)
    ql = q.strip().lower()
    runs = [r for r in _runs()
            if (not status or (status == "live" and _live(r)) or r.get("status") == status)
            and (not user or r.get("user") == user)
            and (not ql or ql in f"{r.get('title', '')} {r.get('dataset', '')} {r.get('model', '')} {r.get('user', '')} {r['id']}".lower())]
    return {"total": len(runs), "runs": [_admin_view(r) for r in runs[offset: offset + max(1, min(limit, 500))]]}


@router.post("/rollouts/{run_id}/cancel")
def cancel(run_id: str, request: Request):
    """Ask the explorer to stop a running rollout: it watches the settings for these, and stops it within seconds."""
    u = _admin(request)
    r = store.get(run_id)
    if not r:
        raise HTTPException(404, "no such rollout")
    asks = {k: v for k, v in settings.get("cancel_requests", {}).items() if time.time() - v < 3600}
    asks[run_id] = time.time()
    settings.save(u["name"], {"cancel_requests": asks}, note=f"asked to stop rollout {run_id}")
    return {"cancelled": _live(r)}


class Moderate(BaseModel):
    hidden: bool


@router.post("/rollouts/{run_id}/moderate")
def moderate(run_id: str, body: Moderate, request: Request):
    """Take a public rollout off Community (or put it back). Its owner still sees it."""
    u = _admin(request)
    r = store.get(run_id)
    if not r:
        raise HTTPException(404, "no such rollout")
    gone = set(settings.get("hidden_runs", []))
    gone = (gone | {run_id}) if body.hidden else (gone - {run_id})
    settings.save(u["name"], {"hidden_runs": sorted(gone)}, note=f"{'hid' if body.hidden else 'restored'} rollout {run_id} on Community")
    return {"run": _admin_view(r)}


# ── people ───────────────────────────────────────────────────────────────────
@router.get("/people")
def people(request: Request):
    """Everyone who has run a rollout: how much, how well, on what, and when last."""
    _admin(request)
    by: dict[str, dict[str, Any]] = {}
    for r in _runs():
        name = r.get("user") or "unknown"
        p = by.setdefault(name, {"name": name, "avatar": None, "runs": 0, "done": 0, "solved": 0, "live": 0, "public": 0,
                                 "first": r.get("created_at"), "last": r.get("created_at"), "models": Counter(), "datasets": Counter()})
        p["avatar"] = p["avatar"] or r.get("avatar")
        p["runs"] += 1
        p["done"] += r.get("status") == "done"
        p["solved"] += (r.get("reward") or 0) >= 0.999
        p["live"] += _live(r)
        p["public"] += r.get("visibility") == "public"
        p["first"] = min(p["first"] or 0, r.get("created_at") or 0)
        p["last"] = max(p["last"] or 0, r.get("created_at") or 0)
        p["models"][r.get("model") or "custom endpoint"] += 1
        p["datasets"][r.get("dataset")] += 1
    out = []
    for p in sorted(by.values(), key=lambda x: -(x["last"] or 0)):
        p["avatar"] = p["avatar"] or avatar(p["name"])
        p["models"], p["datasets"] = p["models"].most_common(3), p["datasets"].most_common(3)
        out.append(p)
    return {"people": out, "me": auth.public(auth.current_user(request))}


# ── environments: pin, hide, feature, index ──────────────────────────────────
@router.get("/environments")
def environments(request: Request):
    _admin(request)
    envs = catalog.environments(include_hidden=True)
    gone = catalog.hidden()
    runs = Counter(r.get("dataset") for r in _runs())
    return {"environments": [{**e, "hidden": e["key"] in gone, "rollouts": runs.get(e["id"], 0) if e["kind"] == "dataset" else 0} for e in envs],
            "collections": catalog.collections(), "jobs": _jobs()}


def _jobs() -> dict[str, Any]:
    with catalog._jobs_lock:
        return {k: {kk: vv for kk, vv in v.items() if kk not in ("sha",)} for k, v in catalog._jobs.items()
                if v.get("state") not in ("done",)}


class EnvAction(BaseModel):
    key: str = Field(max_length=220)
    action: Literal["pin", "unpin", "hide", "unhide", "feature", "unfeature", "index"]
    collection: str | None = Field(None, max_length=40)


@router.post("/environments")
def env_action(body: EnvAction, request: Request):
    u = _admin(request)
    if not KEY.match(body.key):
        raise HTTPException(400, "not an environment key")
    key = body.key
    if body.action in ("pin", "unpin"):
        pins = [k for k in catalog.pinned() if k != key] + ([key] if body.action == "pin" else [])
        settings.save(u["name"], {"pinned": pins}, note=f"{body.action} {key}")
    elif body.action in ("hide", "unhide"):
        gone = sorted((catalog.hidden() - {key}) | ({key} if body.action == "hide" else set()))
        settings.save(u["name"], {"hidden": gone}, note=f"{body.action} {key}")
    elif body.action in ("feature", "unfeature"):
        cols = [dict(c, ids=[k for k in c["ids"] if k != key]) for c in catalog.collections()]
        if body.action == "feature":
            target = next((c for c in cols if c["id"] == body.collection), None)
            if not target:
                raise HTTPException(400, "no such collection")
            target["ids"].append(key)
        settings.save(u["name"], {"collections": cols}, note=f"{body.action} {key}" + (f" in {body.collection}" if body.collection else ""))
    elif body.action == "index":
        if key.startswith("space:"):
            raise HTTPException(400, "Spaces aren't indexed")
        st = catalog.index_status(key)   # anonymous: public datasets only
        settings.save(u["name"], {}, note=f"indexed {key}")
        return {"ok": True, "job": {k: v for k, v in st.items() if k != "index"}}
    return {"ok": True}


# ── collections ──────────────────────────────────────────────────────────────
class Collection(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,30}$")
    group: str = Field(min_length=1, max_length=40)
    about: str = Field("", max_length=80)
    icon: str
    color: str
    ids: list[str] = Field(default_factory=list, max_length=80)


class Collections(BaseModel):
    collections: list[Collection] = Field(max_length=20)


@router.get("/collections")
def get_collections(request: Request):
    _admin(request)
    return {"collections": catalog.collections(), "icons": catalog.ICONS, "colors": catalog.COLORS}


@router.put("/collections")
def put_collections(body: Collections, request: Request):
    u = _admin(request)
    ids = [c.id for c in body.collections]
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "two collections share an id")
    for c in body.collections:
        if c.icon not in catalog.ICONS or c.color not in catalog.COLORS:
            raise HTTPException(400, f"{c.group}: pick an icon and a colour from the lists")
        bad = [k for k in c.ids if not KEY.match(k)]
        if bad:
            raise HTTPException(400, f"{c.group}: not environment ids: {', '.join(bad[:3])}")
    settings.save(u["name"], {"collections": [c.model_dump() for c in body.collections]}, note="edited collections")
    return {"collections": catalog.collections()}


# ── settings ─────────────────────────────────────────────────────────────────
def _settings() -> dict[str, Any]:
    return {"rollouts_enabled": settings.get("rollouts_enabled", True),
            "max_active": settings.get("max_active", config.MAX_ACTIVE_ROLLOUTS),
            "max_per_user": settings.get("max_per_user", config.MAX_ACTIVE_PER_USER),
            "agents": settings.get("agents", [a["id"] for a in runner.AGENTS]),
            "announcement": settings.get("announcement", ""),
            "all_agents": runner.AGENTS, "admin_org": settings.ADMIN_ORG}


class SettingsIn(BaseModel):
    rollouts_enabled: bool
    max_active: int = Field(ge=1, le=200)
    max_per_user: int = Field(ge=1, le=20)
    agents: list[str] = Field(max_length=20)
    announcement: str = Field("", max_length=280)


@router.get("/settings")
def get_settings(request: Request):
    _admin(request)
    return _settings()


@router.put("/settings")
def put_settings(body: SettingsIn, request: Request):
    u = _admin(request)
    if any(a not in runner.AGENT_IDS for a in body.agents):
        raise HTTPException(400, "unknown agent")
    settings.save(u["name"], body.model_dump(), note="changed settings")
    return _settings()


# ── audit and storage ────────────────────────────────────────────────────────
@router.get("/audit")
def audit(request: Request, limit: int = 200):
    _admin(request)
    return {"entries": settings.audit(max(1, min(limit, 1000)))}


@router.get("/indexes")
def indexes(request: Request):
    _admin(request)
    out = []
    for f in sorted(config.INDEX_DIR.glob("*.json.gz")):
        head = {}
        try:
            head = json.loads(f.with_name(f.name.replace(".json.gz", ".head.json")).read_text())
        except (OSError, ValueError):
            try:
                idx = json.loads(gzip.decompress(f.read_bytes()))
                head = {"tasks": len(idx.get("tasks") or []), "version": idx.get("version"), "built": idx.get("built")}
            except (OSError, ValueError):
                pass
        pack = config.PACK_DIR / f.name
        out.append({"spec": f.name[:-8].replace("__", "/", 1), "bytes": f.stat().st_size, "pack_bytes": pack.stat().st_size if pack.exists() else None,
                    "tasks": head.get("tasks"), "version": head.get("version"), "current": head.get("version") == catalog.INDEX_VERSION,
                    "built": head.get("built")})
    return {"indexes": out, "version": catalog.INDEX_VERSION, "jobs": _jobs()}
