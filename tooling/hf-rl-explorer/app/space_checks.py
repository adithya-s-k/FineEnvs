"""Recent, non-executing OpenEnv API evidence, separate from the immutable catalog.

No wake, reset, step or tool invocation. Only public running Spaces are probed.
Per-Space atomic files work on both local storage and the mounted bucket. Search
reads a bounded cached inventory; it never waits for the network.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor

import httpx
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from . import catalog, config

FRESH = 3600
REFRESH = 1800
PASS = "API checked"
UNKNOWN = "Not checked"
_cache = (None, 0.0, {})
_lock = threading.Lock()
_write_lock = threading.Lock()
_refreshing = {}
_stop = threading.Event()
_worker = None
log = logging.getLogger("rlx")


def _path(spec):
    return config.STORAGE_DIR / "openenv-checks" / f"{catalog._slug(catalog.check_spec(spec))}.json"


def save(rec):
    global _cache
    path = _path(rec["id"])
    # Bucket I/O must not hold the catalog's reader lock.
    with _write_lock:
        try:
            old = json.loads(path.read_bytes())
            if old.get("checked_at", 0) > rec["checked_at"]:
                return
        except (OSError, ValueError):
            pass
        catalog.atomic_write(path, json.dumps(rec, separators=(",", ":")).encode())
        with _lock:
            directory = path.parent
            records = dict(_cache[2]) if _cache[0] == directory else {}
            records[rec["id"]] = rec
            _cache = (directory, _cache[1] if _cache[0] == directory else 0.0, records)


def _read_inventory(directory):
    """Public evidence only; invalid or oversized files are ignored, never trusted."""
    records = {}
    for path in sorted(directory.glob("*.json"))[:20000]:
        try:
            if path.stat().st_size > 32 * 1024:
                continue
            r = json.loads(path.read_bytes())
            spec = catalog.check_spec(r["id"])
            if r.get("schema") == 1 and isinstance(r.get("checked_at"), (float, int)):
                records[spec] = r
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def _refresh_inventory(directory, done):
    global _cache
    try:
        records = _read_inventory(directory)
        with _lock:
            if _cache[0] == directory:
                # A check may finish during the read. Its newer evidence,
                # including a failure superseding a pass, must win.
                for spec, rec in _cache[2].items():
                    if rec.get("checked_at", 0) > records.get(spec, {}).get("checked_at", 0):
                        records[spec] = rec
                _cache = (directory, time.time(), records)
    except Exception:
        log.exception("Could not read stored OpenEnv checks")
    finally:
        with _lock:
            _refreshing.pop(directory, None)
        done.set()


def inventory(*, wait=False):
    """Serve cached evidence immediately; refresh bucket files off the request path.

    Cold readers see no verified Spaces until the first read completes. Expiry
    still uses checked_at. Only the background checker waits for storage so it
    does not re-probe everything on boot.
    """
    global _cache
    directory = config.STORAGE_DIR / "openenv-checks"
    with _lock:
        if _cache[0] != directory:
            _cache = (directory, 0.0, {})
        if time.time() - _cache[1] < 60:
            return _cache[2]
        done = _refreshing.get(directory)
        if done is None:
            done = _refreshing[directory] = threading.Event()
            threading.Thread(target=_refresh_inventory, args=(directory, done), daemon=True,
                             name="read-openenv-checks").start()
        records = _cache[2]
    if not wait:
        return records
    done.wait()
    with _lock:
        return _cache[2]


def status(rec, now=None):
    if not rec:
        return UNKNOWN
    age = (time.time() if now is None else now) - rec.get("checked_at", 0)
    if age < 0 or age >= FRESH:
        return "Check expired"
    return rec.get("status") if rec.get("status") in (PASS, "Checks failed", "Not running", "Check unavailable") else UNKNOWN


def verified(rec):
    """Only a fresh successful check of a running server establishes API support."""
    return bool(rec and rec.get("stage") == "RUNNING" and status(rec) == PASS)


def browseable(rec):
    """A live, recognized environment API; this does not certify reward quality."""
    return verified(rec) or bool(rec and rec.get("stage") == "RUNNING"
                                and status(rec) == "Checks failed"
                                and rec.get("interface") in ("nemo-gym", "ors"))


def environment_interface(api):
    """Other supported server contracts, kept distinct from OpenEnv evidence."""
    if not isinstance(api, dict) or not isinstance(api.get("info"), dict) or not api["info"].get("version"):
        return None
    paths = api.get("paths")
    if not isinstance(paths, dict):
        return None
    for name, routes in (("nemo-gym", ("/seed_session", "/verify")), ("ors", ("/create_session", "/{env_name}/call"))):
        if all(isinstance(paths.get(p), dict) and isinstance(paths[p].get("post"), dict) for p in routes):
            return name
    return None


def tag_version(tags):
    for tag in tags or []:
        match = re.fullmatch(r"openenv-(\d[^\s]{0,70})", str(tag), re.I)
        if match:
            try:
                return {"value": str(Version(match[1])), "source": "Hub tag"}
            except InvalidVersion:
                pass
    return {"value": "Unknown", "source": "Unknown"}


def requirement_version(text):
    try:
        req = Requirement(text)
    except InvalidRequirement:
        return None
    if req.name.lower().replace("_", "-") not in ("openenv", "openenv-core"):
        return None  # current OpenEnv and the earlier openenv-core distribution
    if req.marker:
        return {"value": "Conditional", "source": "Dependency constraint", "requirement": str(req)[:240]}
    if req.url:
        ref = re.search(r"@([^/#?\s]+)(?:#.*)?$", req.url)
        return {"value": "git:" + ref[1][:64] if ref else "Unpinned source", "source": "Source reference"}
    specs = list(req.specifier)
    if len(specs) == 1 and specs[0].operator in ("==", "===") and "*" not in specs[0].version:
        try:
            return {"value": str(Version(specs[0].version)), "source": "Dependency pin"}
        except InvalidVersion:
            pass
    return {"value": str(req.specifier) or "Unpinned", "source": "Dependency constraint"}


def parse_version(text, path):
    """Repository evidence, never an assertion about the installed runtime."""
    if path.endswith(".toml") or path.endswith("uv.lock"):
        try:
            doc = tomllib.loads(text)
        except (ValueError, RecursionError):
            return None
        if path.endswith("uv.lock"):
            pins = []
            for pkg in doc.get("package", []):
                if not isinstance(pkg, dict) or pkg.get("name", "").replace("_", "-") not in ("openenv", "openenv-core"):
                    continue
                source = pkg.get("source") or {}
                if source.get("git"):
                    ref = source["git"].rsplit("#", 1)
                    pins.append({"value": "git:" + ref[-1][:64] if len(ref) == 2 else "Unpinned source", "source": "Lockfile"})
                elif pkg.get("version"):
                    pins.append(requirement_version(f"openenv-core=={pkg['version']}"))
            pins = [p for p in pins if p]
            if len({p["value"] for p in pins}) == 1:
                return {**pins[0], "source": "Lockfile"}
            return {"value": "Multiple locked versions", "source": "Lockfile"} if pins else None
        sources = doc.get("tool", {}).get("uv", {}).get("sources", {})
        source = sources.get("openenv") or sources.get("openenv-core") or sources.get("openenv_core")
        if isinstance(source, dict) and source.get("git"):
            ref = source.get("rev") or source.get("tag") or source.get("branch")
            return {"value": "git:" + str(ref)[:64] if ref else "Unpinned source", "source": "Source reference"}
        if source:
            return {"value": "Custom source", "source": "Source reference"}
        deps = doc.get("project", {}).get("dependencies", [])
    else:
        deps = [line.split(" #", 1)[0].strip() for line in text.splitlines()]
    for dep in deps:
        if isinstance(dep, str) and (v := requirement_version(dep)):
            return v
    return None


def declared_version(spec, tags):
    from . import space_files as files
    fallback = tag_version(tags)
    try:
        info = files._info(spec)
        listing, _, _ = files.listing(info)
        paths = [f["path"] for f in listing]
        manifest = files.manifest_path(paths, "Dockerfile" if "Dockerfile" in paths else None, spec)
        root = manifest.rsplit("/", 1)[0] if manifest and "/" in manifest else ""
        keys = files.key_files(paths, root, manifest)
        sizes = {f["path"]: f.get("size", 0) for f in listing}
        reqs = [f"{root}/requirements.txt" if root else "requirements.txt", "requirements.txt"]
        for path in dict.fromkeys([keys.get("lock"), keys.get("pyproject"), *reqs]):
            if not path or path not in sizes or sizes[path] > 400 * 1024:
                continue
            text = files._fetch_text(spec, info["sha"], path, sizes[path])
            if text and (v := parse_version(text, path)):
                return {**v, "path": path, "revision": info["sha"]}
    except Exception:  # incomplete version evidence must not change API health
        pass
    return fallback


def assess(api, health, metadata, schema, tools):
    api = api if isinstance(api, dict) else {}
    paths = api.get("paths") if isinstance(api.get("paths"), dict) else {}
    def method(path, verb):
        return isinstance(paths.get(path), dict) and isinstance(paths[path].get(verb), dict)
    checks = {
        "OpenAPI": isinstance(api.get("info"), dict) and isinstance(api["info"].get("version"), str),
        "Health": isinstance(health, dict) and health.get("status") == "healthy",
        "Metadata": isinstance(metadata, dict) and all(isinstance(metadata.get(k), str) for k in ("name", "description")),
        "Schemas": isinstance(schema, dict) and all(isinstance(schema.get(k), dict) for k in ("action", "observation", "state")),
    }
    simulation = any(p in paths for p in ("/reset", "/step", "/state"))
    mode = "Simulation" if simulation else "MCP service"
    checks["Control routes" if simulation else "MCP tools"] = (
        all(method(p, v) for p, v in (("/reset", "post"), ("/step", "post"), ("/state", "get")))
        if simulation else method("/mcp", "post") and bool(tools))
    return {"status": PASS if all(checks.values()) else "Checks failed", "mode": mode if all(checks.values()) else None,
            "failed": [k for k, ok in checks.items() if not ok], "tools": len(tools or [])}


def _task_splits(task_api):
    from .seo_tasks import ranges

    out = []
    for row in ranges(task_api):
        if len(json.dumps(out + [row]).encode()) > 12000:
            break
        out.append(row)
    return out


def check(spec):
    from . import spaces_live as live
    rec = {"schema": 1, "id": catalog.check_spec(spec), "checked_at": time.time(), "status": "Check unavailable"}
    previous = inventory().get(spec, {})
    rec["version"] = previous.get("version") or {"value": "Unknown", "source": "Unknown"}
    try:
        hub = live.record(spec, fresh=True)
        rec["stage"] = hub["stage"]
        if hub["stage"] != "RUNNING":
            rec["status"] = "Not running"
        elif hub.get("host"):
            urls = ("/openapi.json", "/health", "/metadata", "/schema")
            def get(path):
                try:
                    with live.SpaceClient(timeout=httpx.Timeout(6, connect=4), follow_redirects=False) as client:
                        response = client.get(live._url(hub, path))
                        return response.json() if response.status_code == 200 and "json" in response.headers.get("content-type", "") else None
                except (httpx.HTTPError, ValueError, live.SpaceError):
                    return None
            with ThreadPoolExecutor(4) as pool:
                api, health, meta, schema = list(pool.map(get, urls))
            # MCP discovery is separate from typed-action OpenEnv compatibility.
            paths = api.get("paths", {}) if isinstance(api, dict) else {}
            typed = isinstance(schema, dict) and all(isinstance(schema.get(k), dict) for k in ("action", "observation", "state"))
            tools, _, _ = live._tools({**hub, "mcp_standard": not typed}) if isinstance(paths, dict) and "/mcp" in paths else (None, None, None)
            rec.update(assess(api, health, meta, schema, tools))
            rec["interface"] = environment_interface(api)
            from . import space_tasks
            task_api = space_tasks.discover(hub, paths)
            rec["task_catalog"] = space_tasks.summary(task_api)
            rec["task_splits"] = _task_splits(task_api)
            if rec["status"] == PASS:
                rec["version"] = declared_version(spec, hub.get("tags"))
            elif rec["version"]["value"] == "Unknown":
                rec["version"] = tag_version(hub.get("tags"))
    except Exception as exc:
        rec["error"] = type(exc).__name__  # no upstream response/secret in public inventory
    save(rec)
    return rec


def observe(spec, info):
    """Opening/rechecking a Space also updates discovery, including negative results."""
    try:
        old = inventory().get(spec, {})
        rec = {"schema": 1, "id": spec, "checked_at": info.get("checked") or time.time(), "stage": info.get("stage"),
               "version": old.get("version") or {"value": "Unknown", "source": "Unknown"}}
        if not info.get("running"):
            rec["status"] = "Not running"
        else:
            rec.update(assess({"info": {"version": info.get("openapi_version")}, "paths": info.get("endpoint_methods")},
                              info.get("health"), info.get("metadata"), info.get("schema"), info.get("mcp")))
            rec["interface"] = environment_interface({"info": {"version": info.get("openapi_version")}, "paths": info.get("endpoint_methods")})
            from . import space_tasks
            rec["task_catalog"] = space_tasks.summary(info.get("task_api"))
            rec["task_splits"] = _task_splits(info.get("task_api"))
        info["openenv_verified"] = verified(rec)
        save(rec)
    except Exception:
        log.warning("Could not save OpenEnv check for %s", spec)


def scan():
    # App workers and an operator's CLI refresh share one bounded probe queue.
    # Use local cache storage for this lock, never a mounted object-store bucket.
    try:
        import fcntl
    except ImportError:  # Windows development: the app still has one worker thread
        return _scan()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with (config.CACHE_DIR / "openenv-checks.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        return _scan()


def _scan():
    from . import snapshot
    old = inventory(wait=True)
    with snapshot.use() as (_, conn):
        candidates = list(conn.execute("SELECT id, stage FROM envs WHERE kind = 'space' ORDER BY trending DESC, likes DESC"))
    due = pending(candidates, old)
    def visit(spec):
        if not _stop.is_set():
            return check(spec)
    with ThreadPoolExecutor(4) as pool:
        return [r for r in pool.map(visit, due) if r]


def pending(candidates, old, now=None):
    """Bound the queue and reserve capacity for inactive candidates too.

    Never probing sleeping hosts, but refreshing their Hub metadata lets a
    newly started Space enter the catalog without someone opening its page.
    """
    now = time.time() if now is None else now
    active, inactive = [], []
    for spec, stage in candidates:
        rec = old.get(spec, {})
        running = rec.get("stage", stage) == "RUNNING"
        at = rec.get("checked_at", 0)
        interval = REFRESH if running else 6 * 3600
        if not rec or now - at >= interval or (running and status(rec, now) == PASS and "task_catalog" not in rec):
            (active if running else inactive).append((spec, at))
    # Once checked, older entries take precedence so a failing popular Space
    # cannot starve the long tail. FineEnvs goes first within each generation.
    key = lambda item: (int(item[1] // REFRESH), item[0].split("/", 1)[0].lower() != "fineenvs", item[1])
    active.sort(key=key)
    inactive.sort(key=key)
    return [s for s, _ in active[:96] + inactive[:32]]


def start():
    global _worker
    if os.environ.get("RLX_TEST_TMP") or os.environ.get("RLX_SPACE_CHECKS", "1") != "1" or (_worker and _worker.is_alive()):
        return
    _stop.clear()
    def watch():
        while not _stop.is_set():
            try:
                scan()
            except Exception:
                log.exception("OpenEnv API inventory refresh failed")
            _stop.wait(60)
    _worker = threading.Thread(target=watch, daemon=True, name="openenv-checks")
    _worker.start()


def stop():
    _stop.set()


if __name__ == "__main__":
    from collections import Counter
    from . import runtime
    runtime.hub_timeouts()
    print(dict(Counter(r["status"] for r in scan())), flush=True)
