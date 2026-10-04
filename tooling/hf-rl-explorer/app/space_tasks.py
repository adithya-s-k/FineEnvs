"""Bounded discovery of the public OpenEnv Task API, without starting episodes.

These are server-reported catalog sizes, not downloaded task records or proof that
an episode works. Keep their own timestamp so a health check cannot freshen them.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

MAX_COUNT = 1_000_000_000
FRESH = 3600


def count(value):
    return value if type(value) is int and 0 <= value <= MAX_COUNT else None


def discover(rec, paths, envs=None):
    from . import spaces_live as live

    if "/{env_name}/splits" not in paths:
        return None
    if envs is None:
        envs = live._get_json(rec, "/list_environments")
    if not isinstance(envs, list):
        return None
    names = list(dict.fromkeys(e for e in envs if isinstance(e, str) and re.fullmatch(r"[\w.-]{1,80}", e) and e not in (".", "..")))

    def read(name):
        raw = live._get_json(rec, f"/{name}/splits")
        splits, seen = [], set()
        for s in raw[:40] if isinstance(raw, list) else []:
            if not isinstance(s, dict) or not isinstance(s.get("name"), str) or len(s["name"]) > 200 or s["name"] in seen:
                continue
            seen.add(s["name"])
            n = count(s.get("num_tasks"))
            if n is None and "/{env_name}/num_tasks" in paths and len(splits) < 12:
                try:
                    r = live._client.post(live._url(rec, f"/{name}/num_tasks"), json={"split": s["name"]}, timeout=6)
                    doc = live._json(r) if r.status_code == 200 else None
                    n = count(doc.get("num_tasks")) if isinstance(doc, dict) else None
                except Exception:
                    pass
            splits.append({"name": s["name"], "type": s.get("type"), "num_tasks": n, "default": s.get("default") is True})
        return {"env": name, "splits": splits, "complete": isinstance(raw, list) and len(raw) <= 40 and len(seen) == len(raw)}

    with ThreadPoolExecutor(4) as pool:
        environments = list(pool.map(read, names[:16]))
    if not environments:
        return None
    return {"env": environments[0]["env"], "splits": environments[0]["splits"], "environments": environments,
            "complete": len(names) <= 16 and all(e["complete"] for e in environments), "checked_at": time.time()}


def summary(task_api):
    if not isinstance(task_api, dict):
        return None
    envs = task_api.get("environments") or [task_api]
    entries, unknown, seen = [], 0, set()
    for env in envs[:16]:
        for split in env.get("splits", [])[:40]:
            key = (env.get("env"), split.get("name"))
            if key in seen:
                continue
            seen.add(key)
            n = count(split.get("num_tasks"))
            if n is None:
                unknown += 1
            else:
                entries.append(n)
    return {"tasks": sum(entries), "counted_splits": len(entries), "unknown_splits": unknown,
            "complete": bool(task_api.get("complete", True)) and not unknown,
            "checked_at": task_api.get("checked_at") or time.time()}


def census(records, visible, now=None):
    now = time.time() if now is None else now
    out = {"tasks": 0, "spaces": 0, "partial_spaces": 0, "oldest_check": None}
    for spec in visible:
        rec = records.get(spec, {})
        s = rec.get("task_catalog") or {}
        at = s.get("checked_at", 0)
        if rec.get("stage") != "RUNNING" or not isinstance(at, (int, float)) or not 0 <= now - at < FRESH or not s.get("counted_splits"):
            continue
        n = s.get("tasks")
        if type(n) is not int or not 0 <= n <= MAX_COUNT * 640:
            continue
        out["tasks"] += n
        out["spaces"] += 1
        out["partial_spaces"] += not s.get("complete")
        out["oldest_check"] = min(out["oldest_check"] or at, at)
    return out
