"""What admins can change without a deploy, and who changed what.

One JSON file in the bucket (STORAGE_DIR/admin/settings.json) holds the collections, pinned and hidden
environments, the rollout switches and an announcement; every change is appended to admin/audit.jsonl with who
made it. Values not set here fall back to the defaults in code.

Admins are the members of one Hub organization (RLX_ADMIN_ORG, FineEnvs by default), checked against the Hub,
plus any usernames in RLX_ADMINS.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import config

ADMIN_ORG = os.environ.get("RLX_ADMIN_ORG", "FineEnvs")
EXTRA_ADMINS = {u.strip() for u in os.environ.get("RLX_ADMINS", "").split(",") if u.strip()}
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


# ── who is an admin ──────────────────────────────────────────────────────────
_members: tuple[float, set[str]] | None = None
_members_lock = threading.Lock()


def _org_members() -> set[str]:
    """The admin organization's members, from the Hub, refreshed every 10 minutes."""
    global _members
    with _members_lock:
        if _members and time.time() - _members[0] < 600:
            return _members[1]
    from huggingface_hub import HfApi

    try:
        names = {m.username for m in HfApi(token=False).list_organization_members(ADMIN_ORG)}
    except Exception:  # noqa: BLE001 - the Hub is down: keep the last list, else nobody
        with _members_lock:
            return _members[1] if _members else set()
    with _members_lock:
        _members = (time.time(), names)
    return names


def is_admin(user: dict[str, Any] | None) -> bool:
    """Whether this signed-in user may open the admin dashboard. The name comes from the encrypted session, which
    the Hub set at sign-in, so it can't be claimed by a request."""
    if not user or not user.get("name"):
        return False
    # the org's public member list, or the memberships the Hub reported when this user signed in (private ones too)
    return user["name"] in EXTRA_ADMINS or user["name"] in _org_members() or ADMIN_ORG in (user.get("orgs") or [])


def path_for(name: str) -> Path:
    return DIR / name
