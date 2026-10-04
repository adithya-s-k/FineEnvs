"""How the app runs in production: called once at startup (app/main.py).

  timeouts   every Hub call gets one (huggingface_hub's own client waits forever, and one hung call holds a thread)
  threads    room for slow calls: sync routes and to_thread share one pool, 40 threads by default
  logs       one JSON line per event on stdout, capability tokens and session ids masked out of access logs
  janitor    local disk kept under a budget: task folders, trial folders, downloaded files, old files
  health     /healthz (alive) and /readyz (threads, disk, the store, the newest index), for the platform and for us
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("rlx")
HUB_TIMEOUT = float(os.environ.get("RLX_HUB_TIMEOUT", 120))      # seconds without a byte before a Hub call gives up
THREADS = int(os.environ.get("RLX_THREADS", 200))
DISK_BUDGET = int(os.environ.get("RLX_DISK_BUDGET_GB", 40)) * 1024**3   # local caches, all together
TRIALS_KEEP = 2 * 86400                                               # a rollout's local trial folder, after it ends
_started = time.time()


# ── timeouts and threads ─────────────────────────────────────────────────────
def hub_timeouts() -> None:
    import httpx
    from huggingface_hub.utils import _http

    def factory() -> httpx.Client:
        return httpx.Client(event_hooks={"request": [_http.hf_request_event_hook]}, follow_redirects=True,
                            timeout=httpx.Timeout(HUB_TIMEOUT, connect=15.0))

    _http.set_client_factory(factory)


def more_threads() -> None:
    import anyio.to_thread

    try:
        anyio.to_thread.current_default_thread_limiter().total_tokens = THREADS
    except RuntimeError:   # not inside an event loop yet: done at startup instead
        pass


# ── logs ─────────────────────────────────────────────────────────────────────
SECRET_PATH = re.compile(r"(/(?:api|rlx)/llm/)[^/\s]+|((?:session|cap|token)=)[^&\s]+", re.I)


class _Json(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        msg = SECRET_PATH.sub(lambda m: (m.group(1) or m.group(2)) + "[hidden]", r.getMessage())
        doc: dict[str, Any] = {"t": round(r.created, 3), "level": r.levelname.lower(), "logger": r.name, "msg": msg}
        if r.exc_info:
            doc["exc"] = self.formatException(r.exc_info)[-4000:]
        return json.dumps(doc, ensure_ascii=False)


def logs() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Json())
    for name in ("rlx", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = [h]
        lg.propagate = False
        lg.setLevel(logging.INFO)


# ── the janitor ──────────────────────────────────────────────────────────────
def _size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p, onerror=lambda e: None):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def sweep() -> dict[str, Any]:
    """Free local disk: trial folders of finished rollouts after two days, then the least recently used of everything
    else in the cache until it fits the budget. Never touches STORAGE_DIR (the bucket)."""
    from . import runner

    cache = config.CACHE_DIR
    freed = 0
    trials = getattr(runner, "TRIALS_DIR", cache.parent / "trials")
    live = {r.get("id") for r in runner.live()}
    if trials.is_dir():
        for d in trials.iterdir():
            try:
                if d.name not in live and time.time() - d.stat().st_mtime > TRIALS_KEEP:
                    freed += _size(d)
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
    if cache.is_dir():
        entries = []
        for d in cache.iterdir():
            if d.name in ("hub",):   # huggingface_hub's own cache: its blobs are shared, cleaned below by age
                entries += [(x, x.stat().st_atime) for x in d.iterdir() if x.is_dir()]
            else:
                try:
                    entries.append((d, d.stat().st_atime))
                except OSError:
                    pass
        sizes = {p: _size(p) for p, _ in entries}
        total = sum(sizes.values())
        for p, used in sorted(entries, key=lambda e: e[1]):
            if total <= DISK_BUDGET:
                break
            if time.time() - max(used, p.stat().st_mtime if p.exists() else 0) < 3600:
                continue   # in use (a rollout's task folder, a page's files): never the last hour's
            total -= sizes[p]
            freed += sizes[p]
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    if freed:
        log.info("janitor freed %d MB", freed // (1024 * 1024))
    return {"freed": freed}


def janitor(every: float = 1800) -> None:
    def loop() -> None:
        while True:
            try:
                sweep()
            except Exception:  # noqa: BLE001 - try again next round
                log.exception("janitor failed")
            time.sleep(every)

    threading.Thread(target=loop, daemon=True, name="janitor").start()


# ── health ───────────────────────────────────────────────────────────────────
def readiness() -> tuple[bool, dict[str, Any]]:
    import anyio.to_thread

    from . import runner

    checks: dict[str, Any] = {"uptime_s": round(time.time() - _started)}
    try:
        lim = anyio.to_thread.current_default_thread_limiter()
        checks["threads"] = {"busy": lim.borrowed_tokens, "total": lim.total_tokens}
    except RuntimeError:
        pass
    try:
        du = shutil.disk_usage(config.CACHE_DIR if config.CACHE_DIR.exists() else "/")
        checks["disk_free_gb"] = round(du.free / 1024**3, 1)
    except OSError:
        checks["disk_free_gb"] = None
    try:
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.STORAGE_DIR / ".ready"
        probe.write_text(str(time.time()))
        checks["store"] = "ok"
    except OSError as e:
        checks["store"] = f"unwritable: {type(e).__name__}"
    try:
        newest = max((p.stat().st_mtime for p in config.INDEX_DIR.glob("*.json.gz")), default=None)
        checks["newest_index_age_h"] = round((time.time() - newest) / 3600, 1) if newest else None
    except OSError:
        checks["newest_index_age_h"] = None
    checks["rollouts_live"] = len(runner.live())
    from . import snapshot

    snap = snapshot.status()
    checks["snapshot"] = {k: snap[k] for k in ("source", "db", "built_at", "error")}
    ok = checks["store"] == "ok" and (checks.get("disk_free_gb") or 1) > 0.5 and \
        (not checks.get("threads") or checks["threads"]["busy"] < checks["threads"]["total"] * 0.95)
    return ok, checks


def setup() -> None:
    logs()
    hub_timeouts()
    more_threads()
    if os.environ.get("RLX_JANITOR", "1") == "1":
        janitor()
