"""Rollout records on disk: one folder per run, holding `run.json` and an append-only `events.jsonl`.

The folder is ./.local-runs when you run locally, and the private bucket mounted at /data on the
Space. Nothing here knows which: a mounted bucket is just a directory. Tokens are never written.

A rollout this process is running keeps its events on local disk first (appends and the run page's polls stay local)
and copies them to the store every FLUSH_EVERY seconds and when it ends, so a bucket isn't rewritten per event.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from . import config

RUNS = config.STORAGE_DIR / "runs"
LOCAL = config.CACHE_DIR.parent / "live-runs"   # events of the rollouts running here, before they reach the store
FLUSH_EVERY = 20.0
_local: dict[str, float] = {}   # run id -> when its events last reached the store
_dirty: set[str] = set()        # live rollouts with events the store hasn't got yet
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _lock(run_id: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(run_id, threading.Lock())


def _dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("bad run id")
    return RUNS / run_id


def _write_json(path: Path, doc: dict) -> None:
    """Write-then-rename, so a crash mid-write never leaves a truncated run.json (the run would vanish)."""
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.replace(path)


def create(run: dict) -> dict:
    d = _dir(run["id"])
    d.mkdir(parents=True, exist_ok=False)
    run = {**run, "created_at": time.time(), "updated_at": time.time()}
    _write_json(d / "run.json", run)
    (d / "events.jsonl").touch()
    if run.get("status", "queued") in ACTIVE:   # this process runs it: its events start on local disk
        (LOCAL / run["id"]).mkdir(parents=True, exist_ok=True)
        (LOCAL / run["id"] / "events.jsonl").touch()
        with _lock(run["id"]):
            _local[run["id"]] = time.time()
    _index_put(run)
    return run


def get(run_id: str) -> dict | None:
    p = _dir(run_id) / "run.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def update(run_id: str, **fields) -> dict:
    with _lock(run_id):
        run = get(run_id) or {}
        run.update(fields, updated_at=time.time())
        _write_json(_dir(run_id) / "run.json", run)
        _index_put(run)
    if run_id in _local and run.get("status") and run["status"] not in ACTIVE:
        _end_local(run_id)   # it ended: its events go to the store now, and stay there
    return run


def _events(run_id: str) -> Path:
    return LOCAL / run_id / "events.jsonl" if run_id in _local else _dir(run_id) / "events.jsonl"


def _flush(run_id: str, force: bool = False) -> None:
    """Copy a live rollout's events to the store (whole file, write-then-rename). Call with its lock held."""
    last = _local.get(run_id)
    if last is None or run_id not in _dirty or (not force and time.time() - last < FLUSH_EVERY):
        return
    src, dst = LOCAL / run_id / "events.jsonl", _dir(run_id) / "events.jsonl"
    try:
        tmp = dst.with_name(f".events.jsonl.{threading.get_ident()}.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        _local[run_id] = time.time()
        _dirty.discard(run_id)
    except FileNotFoundError:
        pass


def _end_local(run_id: str) -> None:
    with _lock(run_id):
        if run_id not in _local:
            return
        _flush(run_id, force=True)
        _local.pop(run_id, None)
        _dirty.discard(run_id)
    shutil.rmtree(LOCAL / run_id, ignore_errors=True)


def flush_due(force: bool = False) -> None:
    """Events of live rollouts that went quiet after their last append, to the store (the runner's watchdog calls this,
    so another process reading the store, the admin Space, is never more than FLUSH_EVERY behind)."""
    for run_id in list(_local):
        with _lock(run_id):
            _flush(run_id, force=force)


def _salvage(run_id: str) -> None:
    """A rollout this process ran before it stopped: what its local events had beyond the store's, kept."""
    src = LOCAL / run_id / "events.jsonl"
    try:
        dst = _dir(run_id) / "events.jsonl"
        if src.is_file() and src.stat().st_size > (dst.stat().st_size if dst.exists() else -1):
            tmp = dst.with_name(f".events.jsonl.{threading.get_ident()}.tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(dst)
    except (OSError, ValueError):
        return
    shutil.rmtree(LOCAL / run_id, ignore_errors=True)


def append_events(run_id: str, events: list[dict]) -> None:
    if not events:
        return
    with _lock(run_id):
        with open(_events(run_id), "a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if run_id in _local:
            _dirty.add(run_id)
            _flush(run_id)


def read_events(run_id: str, after: int = 0) -> list[dict]:
    try:
        with open(_events(run_id), encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines[after:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break   # a half-written last line: pick it up on the next poll
    return out


def write_artifact(run_id: str, name: str, data: str | bytes) -> None:
    p = _dir(run_id) / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def has_artifact(run_id: str, name: str) -> bool:
    if "/" in name or ".." in name:
        return False
    try:
        return (_dir(run_id) / name).stat().st_size > 0
    except (FileNotFoundError, ValueError):
        return False


def read_artifact(run_id: str, name: str) -> bytes | None:
    if "/" in name or ".." in name:
        return None
    try:
        return (_dir(run_id) / name).read_bytes()
    except FileNotFoundError:
        return None


# An in-memory index of every run record. On the Space the store is a mounted bucket, where reading every run.json
# per request would be slow, so the index is read once and kept current: every write here updates it, and runs that
# can still change (running ones, or written in the last minutes) are re-read when their file changes, so a record
# another process writes (a test script, an older worker) never shows stale.
_index: dict[str, dict] | None = None
_mtimes: dict[str, float] = {}
_index_lock = threading.Lock()
_synced_at = 0.0
_full_at = 0.0
FRESH = 900   # seconds a finished run's record may still change; older ones are re-checked once a minute


def _mtime(run_id: str) -> float:
    try:
        return (_dir(run_id) / "run.json").stat().st_mtime
    except (OSError, ValueError):
        return 0.0


def _load_index() -> dict[str, dict]:
    global _index
    with _index_lock:
        if _index is None:
            idx = {}
            if RUNS.is_dir():
                for d in RUNS.iterdir():
                    r = get(d.name) if d.is_dir() and not d.name.startswith(".") else None
                    if r and r.get("id"):
                        idx[r["id"]] = r
                        _mtimes[r["id"]] = _mtime(r["id"])
            _index = idx
        return _index


def _sync(every: float = 3.0) -> None:
    """Pick up runs written by another process: new folders, and changed files of runs that can still change."""
    global _synced_at, _full_at
    now = time.time()
    if now - _synced_at < every:
        return
    _synced_at = now
    full = now - _full_at > 60
    if full:
        _full_at = now
    idx = _load_index()
    try:
        names = [d.name for d in RUNS.iterdir() if not d.name.startswith(".")] if RUNS.is_dir() else []
    except OSError:
        return
    for name in names:
        r = idx.get(name)
        if (not full and r is not None and r.get("status") not in ACTIVE and r.get("status") != "interrupted"
                and now - (r.get("updated_at") or 0) > FRESH):
            continue
        m = _mtime(name)
        if r is None or m != _mtimes.get(name):
            fresh = get(name)
            if fresh and fresh.get("id"):
                with _index_lock:
                    idx[name] = fresh
                    _mtimes[name] = m


def refresh(max_age: float) -> None:
    """Re-read every run from disk if the index is older than `max_age` seconds: for a process that only reads runs
    another one writes (the admin Space)."""
    global _index, _synced_at
    if time.time() - _synced_at > max_age:
        with _index_lock:
            _index = None
        _load_index()
        _synced_at = time.time()


def _index_put(run: dict) -> None:
    if run.get("id"):
        idx = _load_index()
        with _index_lock:
            idx[run["id"]] = run
            _mtimes[run["id"]] = _mtime(run["id"])


def list_runs(user: str | None = None, task_id: str | None = None, limit: int = 200, public: bool | None = None,
              domain: str | None = None, model: str | None = None) -> list[dict]:
    _sync()
    runs = [r for r in list(_load_index().values())
            if (not user or r.get("user") == user) and (not task_id or r.get("task_id") == task_id)
            and (public is None or (r.get("visibility") == "public") == public)
            and (not domain or r.get("domain") == domain) and (not model or r.get("model") == model)]
    runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return runs[:limit]


STALE = 180   # a running rollout's record is touched every 20 s; one silent for this long has lost its worker


def mark_interrupted(live: set[str] | None = None) -> int:
    """Rollouts whose worker is gone (the process stopped): still 'active' on disk, not running here, and silent for
    STALE seconds. Checked continuously, not only at startup, and never for a rollout another process is running."""
    n = 0
    now = time.time()
    for r in list_runs(limit=100000):
        if r.get("status") in ACTIVE and r["id"] not in (live or set()) and now - (r.get("updated_at") or 0) > STALE:
            fresh = get(r["id"]) or r   # the file, not the index: another process may have just touched it
            if fresh.get("status") in ACTIVE and now - (fresh.get("updated_at") or 0) > STALE:
                if r["id"] not in _local:
                    _salvage(r["id"])
                update(r["id"], status="interrupted", error="The server restarted while this rollout was running, so it stopped. Run it again.")
                n += 1
    return n


ACTIVE = {"queued", "starting", "setup", "running", "verifying"}
