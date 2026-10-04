"""The catalog's offline indexer: keeps the store's listing, dataset indexes and SQLite snapshot fresh, so the app
never builds them itself. Run hourly as an HF scheduled Job (scripts/schedule_indexer.py), or by hand:

    uv run python -m app.indexer --store .local-data --budget-min 10 --max-builds 5     # a folder as the bucket
    uv run python -m app.indexer --store .local-data --dry-run                          # build, don't publish
    python -m app.indexer --store /data --budget-min 40 --use-token                     # the Job: the bucket at /data

One run:

  1. listing   every environment on the Hub (catalog._listing_fetch, a couple of minutes for every Space with its
               files), refused when it is much smaller than the last snapshot's (a Hub listing that failed halfway
               must not empty the site); written to STORAGE_DIR/listing.json.gz for the app's own code paths.
  2. indexes   datasets whose Hub revision changed since their index was built (or never indexed, or indexed in an
               older format) are rebuilt with catalog's own builders, featured ones first, within a time and count
               budget; a dataset that failed is not retried at the same revision for a day (indexer-state.json).
  3. snapshot  one SQLite file built on local disk (app/snapshot.py: envs, tasks, FTS, indexes, ANALYZE, VACUUM,
               quick_check), streamed one dataset's index at a time.
  4. publish   the file to `snapshots/catalog-<UTC ts>.db`, checked where it landed, then the pointer
               `snapshots/LATEST.v<SCHEMA>.json` written last, atomically. A run that fails anywhere before that
               publishes nothing: the app keeps the snapshot it has.
  5. prune     the newest --keep snapshots stay, and any a pointer names (an older schema's too); older ones go.

Rerunning is safe: indexes already current are skipped, and each run publishes a complete snapshot of its own.
Exit status 0 when a snapshot was published (or built, with --dry-run), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("rlx")

SHA_MARGIN_MS = 3600 * 1000   # a dataset changed up to an hour before its index was built gets its revision checked
RETRY_FAILED_AFTER = 24 * 3600
STATE = "indexer-state.json"


class IndexerError(Exception):
    """A run that must not publish."""


def _ev(event: str, **fields: Any) -> None:
    """One structured log line: `indexer.<event> key=value ...`."""
    log.info("indexer.%s %s", event, " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in fields.items()))


# ── 1. listing ───────────────────────────────────────────────────────────────
def fetch_listing(attempts: int = 2) -> list[dict[str, Any]]:
    from . import catalog

    last: Exception | None = None
    for i in range(attempts):
        try:
            t0 = time.time()
            rows = catalog._listing_fetch(full=True)
            _ev("listing", rows=len(rows), seconds=round(time.time() - t0, 1))
            return rows
        except Exception as e:  # noqa: BLE001 - retried once, then the run fails
            last = e
            _ev("listing_failed", attempt=i + 1, error=f"{type(e).__name__}: {str(e)[:200]}")
            time.sleep(10 * (i + 1))
    raise IndexerError(f"listing failed: {last}")


def guard(rows: list[dict[str, Any]], prev: dict[str, Any] | None, min_ratio: float) -> None:
    """Refuse a listing much smaller than the last snapshot's, overall or for datasets or Spaces alone: one of the
    Hub's listings failing quietly (catalog skips a tag that errors) must not take environments off the site."""
    if not rows:
        raise IndexerError("the listing is empty")
    before = (prev or {}).get("counts") or {}
    now = {"envs": len(rows), "datasets": sum(1 for r in rows if r.get("kind") == "dataset"),
           "spaces": sum(1 for r in rows if r.get("kind") == "space")}
    for k, n in now.items():
        if before.get(k) and n < min_ratio * before[k]:
            raise IndexerError(f"the listing has {n:,} {k}, the last snapshot {before[k]:,}: refusing to publish "
                               f"(a Hub listing probably failed; --force to publish anyway)")


# ── 2. indexes ───────────────────────────────────────────────────────────────
def _state_path() -> Path:
    from . import config

    return config.STORAGE_DIR / STATE


def load_state() -> dict[str, Any]:
    try:
        st = json.loads(_state_path().read_text())
    except (OSError, ValueError):
        st = {}
    st.setdefault("failed", {})
    st.setdefault("not_harbor", {})
    st.setdefault("rebuild", [])
    return st


def save_state(st: dict[str, Any]) -> None:
    from . import catalog

    catalog.atomic_write(_state_path(), json.dumps(st, indent=1, sort_keys=True).encode())


def _head(spec: str) -> dict[str, Any] | None:
    from . import catalog

    try:
        return json.loads(catalog._head_path(spec).read_text())
    except (OSError, ValueError):
        return None


def plan(rows: list[dict[str, Any]], state: dict[str, Any], *, workers: int = 8, info=None) -> list[dict[str, Any]]:
    """The datasets to (re)build, in order: featured first, then a changed revision, an old format, an index left
    partial, never indexed; each by trending. A dataset whose index looks current is skipped without a Hub call;
    one changed around or after its build has its revision checked (`info`, catalog.info by default)."""
    from . import catalog, snapshot

    info = info or catalog.info
    featured = set(catalog.featured_datasets())
    items: list[dict[str, Any]] = []
    check: list[dict[str, Any]] = []
    for r in rows:
        if r.get("kind") != "dataset" or r.get("private") or r.get("gated"):
            continue
        spec = r["id"]
        head = _head(spec)
        tagged = r.get("framework") == "harbor"
        if not (tagged or (head and head.get("tasks")) or spec in featured):
            continue   # rows datasets are read row by row (app/envs): nothing to index
        it = {"spec": spec, "featured": spec in featured, "tagged": tagged, "trending": r.get("trending") or 0,
              "downloads": r.get("downloads") or 0, "head": head}
        if head is None:
            it["reason"] = "new"
        elif head.get("version") != catalog.INDEX_VERSION:
            it["reason"] = "format"
        elif spec in state.get("rebuild", []):
            it["reason"] = "partial"
        elif snapshot.iso_ms(r.get("updated")) >= (head.get("built") or 0) * 1000 - SHA_MARGIN_MS:
            it["reason"] = "changed?"
            check.append(it)
            continue
        else:
            continue
        items.append(it)
    if check:   # one Hub call each, in parallel: only for datasets touched around or after their build
        def sha_of(it):
            try:
                return info(it["spec"])
            except Exception as e:  # noqa: BLE001 - deleted or gated since: skip it
                _ev("info_failed", spec=it["spec"], error=f"{type(e).__name__}: {str(e)[:120]}")
                return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for it, meta in zip(check, pool.map(sha_of, check)):
                if meta and meta.get("sha") != (it["head"] or {}).get("sha"):
                    items.append({**it, "reason": "changed", "meta": meta})
    order = {"changed": 0, "format": 1, "partial": 2, "new": 3}
    items.sort(key=lambda it: (not it["featured"], order[it["reason"]], -it["trending"], -it["downloads"], it["spec"]))
    return items


def _build_one(spec: str, meta: dict[str, Any]) -> int:
    """catalog's own build, as app.precache runs it: the index and the pack, written to STORAGE_DIR."""
    from . import catalog

    idx, pack = catalog.build_index(spec, meta, {})
    if pack is not None:
        catalog._write_pack(spec, meta["sha"], pack)
    catalog._write_index(idx)
    return len(idx.get("tasks") or [])


def refresh_indexes(items: list[dict[str, Any]], state: dict[str, Any], *, budget_s: float, max_builds: int,
                    build=None, info=None, looks_harbor=None) -> dict[str, Any]:
    """Build what `plan` found, until the time or the count runs out. A build that outlives the budget is left to
    finish in the background (its files are written atomically) and the run moves on."""
    from . import catalog

    build, info = build or _build_one, info or catalog.info
    looks_harbor = looks_harbor or catalog.looks_harbor
    deadline = time.time() + budget_s
    out: dict[str, Any] = {"built": [], "failed": [], "skipped": [], "deferred": 0, "abandoned": []}
    now = time.time()
    for it in items:
        spec = it["spec"]
        if len(out["built"]) + len(out["failed"]) >= max_builds or time.time() >= deadline:
            out["deferred"] += 1
            continue
        try:
            meta = it.get("meta") or info(spec)
        except Exception as e:  # noqa: BLE001 - gone, gated: not this run
            out["skipped"].append({"spec": spec, "why": f"info: {type(e).__name__}"})
            continue
        sha = meta.get("sha")
        if meta.get("restricted") or meta.get("private") or meta.get("gated"):
            out["skipped"].append({"spec": spec, "why": "not public"})
            continue
        failed = state["failed"].get(spec)
        if failed and failed.get("sha") == sha and now - failed.get("at", 0) < RETRY_FAILED_AFTER:
            out["skipped"].append({"spec": spec, "why": "failed at this revision less than a day ago"})
            continue
        if not it.get("tagged") and not (it.get("head") or {}).get("tasks"):
            if state["not_harbor"].get(spec) == sha:
                continue
            if not looks_harbor(spec, sha):
                state["not_harbor"][spec] = sha
                continue
        t0 = time.time()
        result: dict[str, Any] = {}

        def work(spec: str = spec, meta: dict[str, Any] = meta, result: dict[str, Any] = result) -> None:
            # bound now: a build left running past the budget must not write into a later one's result
            try:
                result["tasks"] = build(spec, meta)
            except BaseException as e:  # noqa: BLE001 - reported below
                result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

        th = threading.Thread(target=work, daemon=True, name=f"index-{spec}")
        th.start()
        th.join(timeout=max(60.0, deadline - time.time() + 120))
        if th.is_alive():
            out["abandoned"].append(spec)
            _ev("build_abandoned", spec=spec, seconds=round(time.time() - t0))
            continue
        if "error" in result:
            state["failed"][spec] = {"sha": sha, "at": time.time(), "error": result["error"]}
            out["failed"].append({"spec": spec, "error": result["error"]})
            _ev("build_failed", spec=spec, reason=it["reason"], error=result["error"])
        else:
            state["failed"].pop(spec, None)
            if spec in state["rebuild"]:
                state["rebuild"].remove(spec)
            out["built"].append({"spec": spec, "tasks": result.get("tasks"), "reason": it["reason"],
                                 "seconds": round(time.time() - t0, 1)})
            _ev("built", spec=spec, reason=it["reason"], tasks=result.get("tasks"), seconds=round(time.time() - t0, 1))
    return out


# ── 3–5. snapshot, publish, prune ────────────────────────────────────────────
def env_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Listing rows as the snapshot keeps them: with their index's headline, and Harbor whenever that index found task
    folders (catalog.environments does the same). Collections, pins and hidden ones are left to query time."""
    from . import catalog

    out = []
    for r in rows:
        indexed = catalog.headline(r["id"]) if r.get("kind") == "dataset" else None
        fw = "harbor" if indexed and indexed.get("tasks") else r.get("framework")
        out.append({**r, "framework": fw, "indexed": indexed})
    return out


def build_snapshot(rows: list[dict[str, Any]], path: Path, meta: dict[str, Any], seen: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import snapshot

    envs = env_rows(rows)
    public = [r["id"] for r in envs if r.get("kind") == "dataset" and not r.get("private") and not r.get("gated")]
    return snapshot.build_db(path, envs, snapshot.iter_index_tasks(public, mimo=snapshot.MIMO in public, seen=seen), meta)


def publish(store, path: Path, counts: dict[str, Any], *, extra: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    """The database into the store under a new name, checked where it landed, then the pointer: last, in one write."""
    from . import snapshot

    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    name, n = f"{snapshot.SNAP_DIR}/catalog-{stamp}.db", 2
    while store.size(name) is not None:   # two runs in one second: never overwrite a published file
        name, n = f"{snapshot.SNAP_DIR}/catalog-{stamp}-{n}.db", n + 1
    digest, size = snapshot.sha256_file(path), path.stat().st_size
    t0 = time.time()
    store.put_file(path, name)
    landed = store.size(name)
    if landed != size:
        raise IndexerError(f"{name}: {landed} bytes in the store, {size} written")
    back = store.digest(name)
    if back is not None and back != digest:
        raise IndexerError(f"{name}: sha256 in the store differs from the one built")
    ptr = {"db": name, "sha256": digest, "size": size, "built_at": counts.get("built_at"), "schema": snapshot.SCHEMA,
           "counts": {k: v for k, v in counts.items() if k != "built_at"}, **(extra or {})}
    store.put_bytes(snapshot.POINTER, json.dumps(ptr, indent=1).encode())
    _ev("published", db=name, bytes=size, sha256=digest[:16], seconds=round(time.time() - t0, 1))
    return ptr


def prune(store, keep: int) -> list[str]:
    """Old snapshots out: the newest `keep` stay, and every one a pointer names (any schema's). Stray temp files from a
    crashed run go after a day."""
    from . import snapshot

    names = store.list(snapshot.SNAP_DIR)
    protected = set()
    for n in names:
        if Path(n).name.startswith("LATEST.v") and n.endswith(".json"):
            try:
                protected.add(json.loads(store.read_bytes(n) or b"{}").get("db"))
            except ValueError:
                continue
    dbs = sorted((n for n in names if snapshot.DB_NAME.match(n)), reverse=True)
    gone = [n for n in dbs[keep:] if n not in protected]
    gone += [n for n in names if Path(n).name.startswith(".") and n.endswith(".tmp") and store.age(n) > 86400]
    if gone:
        store.delete(gone)
        _ev("pruned", files=gone)
    return gone


def run(store, *, rows: list[dict[str, Any]] | None = None, budget_s: float = 1800, max_builds: int = 100, keep: int = 5,
        min_ratio: float = 0.8, force: bool = False, index: bool = True, dry_run: bool = False,
        out: Path | None = None) -> dict[str, Any]:
    """One indexer run (see the module docstring). `rows` stands in for the Hub listing (tests). Raises IndexerError
    (or anything else) without publishing."""
    from . import catalog, config, snapshot

    t0 = time.time()
    try:
        prev = snapshot.read_pointer(store)
    except snapshot.SnapshotError as e:
        _ev("pointer_unreadable", error=str(e))
        prev = None
    rows = rows if rows is not None else fetch_listing()
    rows = [r for r in rows if not r.get("private") and not r.get("gated")]
    if not force:
        guard(rows, prev, min_ratio)
    # catalog's own readers (environments, featured lookups) see this listing, and the app's fallback path finds it
    with catalog._listing_lock:
        catalog._env_list.update(rows=rows, at=time.time(), full=True)
    if not dry_run:
        import gzip

        catalog.atomic_write(config.STORAGE_DIR / "listing.json.gz",
                             gzip.compress(json.dumps({"at": time.time(), "full": True, "rows": rows}, default=str).encode()))
    report: dict[str, Any] = {"listing": len(rows)}
    state = load_state()
    if index and not dry_run:
        items = plan(rows, state)
        _ev("plan", builds=len(items), featured=sum(1 for i in items if i["featured"]),
            reasons={r: sum(1 for i in items if i["reason"] == r) for r in ("changed", "format", "partial", "new")})
        report["indexes"] = refresh_indexes(items, state, budget_s=budget_s, max_builds=max_builds)
        save_state(state)
    # 3. the database, on local disk
    if dry_run and out is None:
        out = snapshot.local_dir() / "dry-run-catalog.db"
        out.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="rlx-snapshot-"))
    path = out or tmpdir / "catalog.db"
    seen: dict[str, Any] = {}
    try:
        t1 = time.time()
        counts = build_snapshot(rows, path, {"index_version": catalog.INDEX_VERSION, "source": os.environ.get("JOB_ID") or "indexer"}, seen)
        # an index left partial (a walk cut short after its task list) is rebuilt next run
        state["rebuild"] = sorted(s for s, v in seen.items() if v.get("partial"))
        if not dry_run:
            save_state(state)
        _ev("snapshot_built", seconds=round(time.time() - t1, 1), bytes=path.stat().st_size, **{k: v for k, v in counts.items() if k != "built_at"})
        if counts["envs"] == 0:
            raise IndexerError("the snapshot has no environments")
        report["counts"] = counts
        if dry_run:
            report["dry_run"] = str(path)
            return report
        # 4. publish (pointer last), 5. prune
        report["pointer"] = publish(store, path, counts, extra={"index_version": catalog.INDEX_VERSION})
        report["pruned"] = prune(store, keep)
    finally:
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        tmpdir.rmdir()
    report["seconds"] = round(time.time() - t0, 1)
    _ev("done", seconds=report["seconds"], db=report["pointer"]["db"], envs=counts["envs"], tasks=counts["tasks"],
        built=len((report.get("indexes") or {}).get("built", [])), failed=len((report.get("indexes") or {}).get("failed", [])))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None,
                    help="a folder (local dev, or the bucket mounted at /data) or hf://buckets/<org>/<name>; "
                         "default RLX_SNAPSHOT_STORE, else STORAGE_DIR")
    ap.add_argument("--budget-min", type=float, default=30, help="minutes for rebuilding indexes (default 30)")
    ap.add_argument("--max-builds", type=int, default=100, help="at most this many index rebuilds a run (default 100)")
    ap.add_argument("--keep", type=int, default=5, help="snapshots kept in the store (default 5)")
    ap.add_argument("--min-ratio", type=float, default=0.8, help="refuse a listing smaller than this share of the last")
    ap.add_argument("--force", action="store_true", help="publish even when the listing shrank a lot")
    ap.add_argument("--no-index", action="store_true", help="don't rebuild any index, only the snapshot")
    ap.add_argument("--dry-run", action="store_true", help="build the snapshot locally, publish nothing, write nothing")
    ap.add_argument("--out", type=Path, default=None, help="with --dry-run: where to keep the database")
    ap.add_argument("--use-token", action="store_true", help="download public dataset files with your HF token")
    args = ap.parse_args(argv)
    # before app.config is imported: a folder store is STORAGE_DIR too, so indexes and the listing land beside it
    if args.store and not str(args.store).startswith("hf://"):
        os.environ["STORAGE_DIR"] = str(Path(args.store).resolve())
    if args.use_token:
        from huggingface_hub import get_token

        os.environ["RLX_INDEX_TOKEN"] = get_token() or ""
    from . import runtime, snapshot

    runtime.logs()
    runtime.hub_timeouts()
    try:
        store = snapshot.open_store(args.store)
        _ev("start", store=str(store), budget_min=args.budget_min, max_builds=args.max_builds, dry_run=args.dry_run)
        report = run(store, budget_s=args.budget_min * 60, max_builds=args.max_builds, keep=args.keep,
                     min_ratio=args.min_ratio, force=args.force, index=not args.no_index, dry_run=args.dry_run,
                     out=args.out)
    except Exception:   # nothing was published; the app keeps the snapshot it has
        log.exception("indexer.failed")
        return 1
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
