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


def create(run: dict) -> dict:
    d = _dir(run["id"])
    d.mkdir(parents=True, exist_ok=False)
    run = {**run, "created_at": time.time(), "updated_at": time.time()}
    (d / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))
    (d / "events.jsonl").touch()
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
        (_dir(run_id) / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))
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


def list_runs(user: str | None = None, task_id: str | None = None, limit: int = 200) -> list[dict]:
    if not RUNS.is_dir():
        return []
    runs = []
    for d in RUNS.iterdir():
        r = get(d.name) if d.is_dir() else None
        if not r or (user and r.get("user") != user) or (task_id and r.get("task_id") != task_id):
            continue
        runs.append(r)
    runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return runs[:limit]


def mark_interrupted() -> int:
    """At startup: anything still 'active' lost its worker when the process died."""
    n = 0
    for r in list_runs(limit=100000):
        if r.get("status") in ACTIVE:
            update(r["id"], status="interrupted", error="The Space restarted while this rollout was running.")
            n += 1
    return n


ACTIVE = {"queued", "starting", "setup", "running", "verifying"}
