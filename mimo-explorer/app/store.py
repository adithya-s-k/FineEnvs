"""Rollout records on disk: one folder per run, holding `run.json` and an append-only `events.jsonl`.

The folder is ./.local-runs when you run locally, and the private bucket mounted at /data on the
Space. Nothing here knows which: a mounted bucket is just a directory. Tokens are never written.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import config

RUNS = config.STORAGE_DIR / "runs"
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
        return run


def append_events(run_id: str, events: list[dict]) -> None:
    if not events:
        return
    with _lock(run_id), open(_dir(run_id) / "events.jsonl", "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def read_events(run_id: str, after: int = 0) -> list[dict]:
    try:
        with open(_dir(run_id) / "events.jsonl", encoding="utf-8") as f:
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


def read_artifact(run_id: str, name: str) -> bytes | None:
    if "/" in name or ".." in name:
        return None
    try:
        return (_dir(run_id) / name).read_bytes()
    except FileNotFoundError:
        return None


# An in-memory index of every run record, read from disk once. On the Space the store is a mounted bucket, where
# reading every run.json per request would be slow; every write above also updates the index.
_index: dict[str, dict] | None = None
_index_lock = threading.Lock()


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
            _index = idx
        return _index


def _index_put(run: dict) -> None:
    if run.get("id"):
        idx = _load_index()
        with _index_lock:
            idx[run["id"]] = run


def list_runs(user: str | None = None, task_id: str | None = None, limit: int = 200, public: bool | None = None,
              domain: str | None = None, model: str | None = None) -> list[dict]:
    runs = [r for r in list(_load_index().values())
            if (not user or r.get("user") == user) and (not task_id or r.get("task_id") == task_id)
            and (public is None or (r.get("visibility") == "public") == public)
            and (not domain or r.get("domain") == domain) and (not model or r.get("model") == model)]
    runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return runs[:limit]


def mark_interrupted() -> int:
    """At startup: anything still 'active' lost its worker when the process died."""
    n = 0
    for r in list_runs(limit=100000):
        if r.get("status") in ACTIVE:
            update(r["id"], status="interrupted", error="The server restarted while this rollout was running, so it stopped. Run it again.")
            n += 1
    return n


ACTIVE = {"queued", "starting", "setup", "running", "verifying"}
