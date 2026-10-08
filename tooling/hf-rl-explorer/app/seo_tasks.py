"""Anonymous, bounded task reads for server-rendered public task pages.

No episode, index build, credentials or model call. Only advertised Task API
coordinates and public row datasets may be read. Answers use the UI's withholding.
"""
from __future__ import annotations

import re
import threading
from urllib.parse import urlencode

from . import catalog, spaces_live

_reads = threading.BoundedSemaphore(4)


class Unavailable(Exception):
    def __init__(self, status=503):
        self.status = status


def read(key, fn):
    def fetch():
        if not _reads.acquire(blocking=False):
            raise Unavailable()
        try:
            return fn()
        finally:
            _reads.release()
    return catalog._cached(("seo-task", *key), 600, fetch)


def space_path(spec, env, split, index):
    return f"/s/{spec}?" + urlencode({"env": env, "split": split, "task": index})


def ranges(task_api):
    if not isinstance(task_api, dict):
        return []
    result, seen = [], set()
    for e in (task_api.get("environments") or [task_api])[:16]:
        name = e.get("env")
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]{1,80}", name) or name in (".", ".."):
            continue
        for s in e.get("splits", [])[:40]:
            split, n = s.get("name"), s.get("num_tasks")
            if not isinstance(split, str) or not 0 < len(split) <= 200 or type(n) is not int or not 0 < n <= 1_000_000_000:
                continue
            if (name, split) not in seen:
                result.append([name, split, n])
                seen.add((name, split))
    return result


def space(spec, env, split, index):
    """Validate against the last public probe before making a read-only task request."""
    known = ranges((spaces_live.last_seen(spec) or {}).get("task_api"))
    if not known:
        known = ranges(spaces_live.probe(spec).get("task_api"))
    match = next((r for r in known if r[0] == env and r[1] == split), None)
    if not match or not 0 <= index < match[2]:
        raise Unavailable(404)

    def fetch():
        try:
            raw = spaces_live.task(spec, split, index, env).get("task")
        except spaces_live.SpaceError as exc:
            raise Unavailable(404 if exc.status == 404 else 503) from None
        if not isinstance(raw, dict):
            raise Unavailable(404)
        # Explicit text fields only; do not serialize arbitrary task metadata.
        title = next((raw[k] for k in ("task_name", "title", "task_id", "id") if isinstance(raw.get(k), str)), f"Task {index + 1}")
        prompt = next((raw[k] for k in ("prompt", "instruction", "description", "question") if isinstance(raw.get(k), str)), "")
        fields = {k: raw[k] for k in ("category", "difficulty", "language", "language_name", "family", "mime", "duration_seconds",
                                     "sampling_rate", "n_frames", "provider", "sequence_id", "offline_ready", "media_ready")
                  if isinstance(raw.get(k), (str, int, float, bool))}
        return {"title": title[:400], "brief": prompt[:16000], "total": match[2], "fields": fields}
    return read(("space", spec, env, split, index), fetch)


def row(spec, ref):
    from .envs import rows

    try:
        config, split, index = rows.parse_ref(ref)
    except (ValueError, LookupError):
        raise Unavailable(404) from None
    if index < 0:
        raise Unavailable(404)

    def fetch():
        try:
            view = rows.task(spec, config, split, index, token=None)
        except (LookupError, PermissionError):
            raise Unavailable(404) from None
        except Exception:
            raise Unavailable() from None
        if view.get("restricted"):
            raise Unavailable(404)
        texts = []
        for section in view.get("sections", []):
            if section.get("id") not in ("task", "prompt", "messages", "instruction"):
                continue
            body = section.get("body")
            if isinstance(body, str):
                texts.append(body)
            elif section.get("kind") == "messages" and isinstance(body, list):
                texts.extend(m["content"] for m in body if isinstance(m, dict)
                             and m.get("role") in ("system", "user") and isinstance(m.get("content"), str))
            elif section.get("kind") == "blocks" and isinstance(body, list):
                texts.extend(b["text"] for b in body if isinstance(b, dict)
                             and b.get("type") in ("markdown", "custom", "note") and isinstance(b.get("text"), str))
        return {"title": str(view.get("title") or ref)[:400], "brief": "\n\n".join(texts)[:16000],
                "path": ref, "total": view.get("total")}
    return read(("row", spec, ref), fetch)
