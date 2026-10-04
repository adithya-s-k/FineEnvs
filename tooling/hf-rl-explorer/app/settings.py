"""What admins can change without a deploy, and who changed what.

One JSON file in the bucket (STORAGE_DIR/admin/settings.json) holds the collections, pinned and hidden
environments, the rollout switches and an announcement; every change is appended to admin/audit.jsonl with who
made it. Values not set here fall back to the defaults in code.

The separate private admin application writes these settings; this public module
only supplies the shared storage contract.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import config

DIR = config.STORAGE_DIR / "admin"
PATH = DIR / "settings.json"
AUDIT = DIR / "audit.jsonl"

_lock = threading.Lock()
_cache: tuple[float, dict[str, Any]] | None = None
_checked = 0.0


def load() -> dict[str, Any]:
    """The saved settings (re-read when the file changes, as another replica may write it)."""
    global _cache, _checked
    if _cache and time.time() - _checked < 2:   # a stat on the bucket mount per call adds up: every 2 s is enough
        return _cache[1]
    try:
        mtime = PATH.stat().st_mtime
    except OSError:
        return {}
    _checked = time.time()
    with _lock:
        if _cache and _cache[0] == mtime:
            return _cache[1]
        try:
            data = json.loads(PATH.read_text())
        except (OSError, ValueError):
            data = {}
        _cache = (mtime, data)
        return data


def get(key: str, default: Any = None) -> Any:
    v = load().get(key)
    return default if v is None else v


def save(user: str, changes: dict[str, Any], note: str = "") -> dict[str, Any]:
    """Write `changes` over the saved settings, and record who did it."""
    global _cache
    with _lock:
        try:
            data = json.loads(PATH.read_text())
        except (OSError, ValueError):
            data = {}
        before = {k: data.get(k) for k in changes}
        data.update(changes)
        DIR.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_name(f".{PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(PATH)
        _cache = None
        with AUDIT.open("a") as fh:
            fh.write(json.dumps({"at": time.time(), "user": user, "note": note, "before": before, "after": changes}) + "\n")
    return data


def audit(limit: int = 200) -> list[dict[str, Any]]:
    try:
        lines = AUDIT.read_text().splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def path_for(name: str) -> Path:
    return DIR / name
