"""Bounded automatic public catalog maintenance, independent of page visits.

The existing indexer runs in a child process so stalled downloads cannot leave
builder threads behind. A process lease prevents duplicate app workers from
publishing concurrently. No HF Jobs, inference, or episode execution is used.
Deployments with an external indexer can set RLX_AUTO_INDEX=0.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
import threading
import time

from . import catalog, config

INTERVAL = 600
LISTING_REFRESH = 3600
MAX_SECONDS = 600
_stop = threading.Event()
_worker = None
log = logging.getLogger("rlx")


def status():
    try:
        record = json.loads((config.STORAGE_DIR / "auto-index.json").read_bytes())
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(record):
    catalog.atomic_write(config.STORAGE_DIR / "auto-index.json", json.dumps(record).encode())


def cached_listing(now):
    """Reuse only a recent complete public discovery; never extend its age."""
    try:
        with gzip.open(config.STORAGE_DIR / "listing.json.gz", "rt") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            return None, None
        at = doc.get("at", 0)
        if doc.get("full") and isinstance(doc.get("rows"), list) and doc["rows"] and 0 <= now - at < LISTING_REFRESH:
            return doc["rows"], at
    except (OSError, ValueError, TypeError):
        pass
    return None, None


def cycle():
    """One lease-protected cycle; publication remains atomic in indexer.run."""
    import fcntl
    from . import indexer, snapshot

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with (config.CACHE_DIR / "auto-index.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        now = time.time()
        previous = status()
        if previous.get("next_at", 0) > now:
            return
        record = {"state": "indexing", "started_at": now, "last_success": previous.get("last_success"),
                  "failures": previous.get("failures", 0), "next_at": now + MAX_SECONDS + INTERVAL}
        _save(record)
        try:
            rows, at = cached_listing(now)
            report = indexer.run(snapshot.open_store(), rows=rows, listing_at=at, budget_s=90, max_builds=4)
            _save({"state": "idle", "last_success": time.time(), "next_at": time.time() + INTERVAL,
                   "failures": 0, "counts": report["counts"]})
        except Exception as exc:
            failures = record["failures"] + 1
            _save({**record, "state": "retrying", "failures": failures, "error": type(exc).__name__,
                   "next_at": time.time() + min(3600, INTERVAL * 2 ** min(failures - 1, 3))})
            log.warning("Automatic catalog update failed: %s", type(exc).__name__)


def _child():
    # A public index must not accidentally depend on a developer's HF login.
    env = {**os.environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    child = subprocess.Popen([sys.executable, "-m", "app.auto_index"], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + MAX_SECONDS
    try:
        while child.poll() is None:
            if _stop.wait(1) or time.monotonic() >= deadline:
                break
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def start():
    global _worker
    if os.environ.get("RLX_TEST_TMP") or os.environ.get("RLX_AUTO_INDEX", "1") != "1" or (_worker and _worker.is_alive()):
        return
    _stop.clear()
    def watch():
        while not _stop.is_set():
            try:
                _child()
            except Exception as exc:
                log.warning("Automatic catalog worker failed: %s", type(exc).__name__)
            _stop.wait(60)
    _worker = threading.Thread(target=watch, daemon=True, name="auto-index")
    _worker.start()


def stop():
    _stop.set()
    if _worker:
        _worker.join(timeout=5)


if __name__ == "__main__":
    from . import runtime
    runtime.hub_timeouts()
    cycle()
