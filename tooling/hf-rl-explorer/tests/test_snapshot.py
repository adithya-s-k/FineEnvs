"""Catalog snapshots (app/snapshot.py) and the indexer that publishes them (app/indexer.py), with no network.

What is checked: a snapshot built from listing rows holds only public environments and their tasks' public fields
(no private or gated dataset, no index built for a signed-in visitor, no metadata values); publishing writes the
database first and the pointer last, and the app swaps to it while a query on the old one finishes; a pointer whose
database is corrupt, truncated or not the file it names is rejected and the app keeps what it had (or falls back to
the live catalog); a newer schema's pointer is left alone; the indexer rebuilds only indexes whose dataset changed,
within its budget, doesn't retry a failure at the same revision, refuses a listing that shrank, publishes nothing
when any step fails, reruns safely and prunes old snapshots; the same works through the bucket API.
"""

from __future__ import annotations

import gzip
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import catalog, config, indexer, snapshot


def ds(i: int, **kw) -> dict:
    return {"id": f"snaptest/d{i}", "key": f"snaptest/d{i}", "kind": "dataset", "framework": "harbor", "heading": f"D{i}",
            "brief": "tasks", "tags": ["t"], "badges": [], "downloads": i, "likes": 0, "trending": i,
            "updated": "2026-09-01T00:00:00+00:00", **kw}


def sp(i: int, **kw) -> dict:
    return {"id": f"snaptest/s{i}", "key": f"space:snaptest/s{i}", "kind": "space", "framework": "openenv", "openenv": True,
            "heading": None, "brief": "", "tags": [], "badges": ["OpenEnv"], "downloads": 0, "likes": i, "trending": 0, **kw}


ROWS = [ds(0), ds(1), ds(2, private=True), ds(3, gated=True), ds(4), sp(0), sp(1), sp(2)]
SECRET = "SECRET-ANSWER-4242"


def task(path: str, title: str = "A task") -> dict:
    return {"path": path, "title": title, "brief": "Do the thing.", "category": "c", "difficulty": "easy", "tags": ["x"],
            "meta": {"gold_answer": SECRET, "note": SECRET}, "withheld": ["gold_answer"], "verifier": {"kind": "txt", "env": ["OPENAI_KEY"]},
            "env": {"image": "img", "gpus": 0}, "run": "image"}


def write_index(spec: str, tasks: list[dict], *, restricted: bool = False, version: int | None = None, sha: str = "sha1",
                built: float = 1e9, partial: bool = False) -> None:
    idx = {"spec": spec, "sha": sha, "built": built, "version": version or catalog.INDEX_VERSION, "info": {"restricted": restricted},
           "tasks": tasks, "summary": {}, **({"partial": True} if partial else {})}
    catalog.atomic_write(catalog._index_path(spec), gzip.compress(json.dumps(idx).encode()))
    head = {"tasks": len(tasks), "graded": {"txt": len(tasks)}, "image": len(tasks), "built": built, "sha": sha,
            "version": version or catalog.INDEX_VERSION}
    catalog.atomic_write(catalog._head_path(spec), json.dumps(head).encode())
    catalog._heads.pop(spec, None)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("RLX_SNAPSHOT_LOCAL", str(tmp_path / "local"))
    monkeypatch.setenv("RLX_SNAPSHOT_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(catalog, "environments", lambda include_hidden=False: ROWS)
    monkeypatch.setattr(catalog, "hidden", lambda: set())
    write_index("snaptest/d0", [task("tasks/a", "Alpha"), task("tasks/b", "Beta")])
    write_index("snaptest/d2", [task("tasks/p")])                    # a private dataset's index (a visitor built it)
    write_index("snaptest/d4", [task("tasks/r")], restricted=True)   # an index built with a visitor's token
    snapshot.reset()
    listing = dict(catalog._env_list)
    had_listing = (config.STORAGE_DIR / "listing.json.gz").exists()
    yield
    snapshot.reset()
    for p in list(config.INDEX_DIR.glob("snaptest__*")):
        p.unlink(missing_ok=True)
    (config.STORAGE_DIR / indexer.STATE).unlink(missing_ok=True)
    catalog._env_list.clear()
    catalog._env_list.update(listing)   # the indexer sets catalog's listing: other tests mustn't see these rows
    if not had_listing:
        (config.STORAGE_DIR / "listing.json.gz").unlink(missing_ok=True)


def store_of(tmp_path: Path) -> snapshot.LocalStore:
    return snapshot.LocalStore(tmp_path / "store")


def build(tmp_path: Path, rows=ROWS, name: str = "build.db") -> tuple[Path, dict]:
    db = tmp_path / name
    counts = indexer.build_snapshot(rows, db, {"test": True})
    return db, counts


# ── what a snapshot holds ────────────────────────────────────────────────────
def test_snapshot_holds_public_rows_and_public_fields_only(tmp_path):
    db, counts = build(tmp_path)
    assert counts["envs"] == 6 and counts["datasets"] == 3 and counts["spaces"] == 3
    assert counts["tasks"] == 2 and counts["task_envs"] == 1   # d0's; not the private one's, nor the restricted index's
    raw = db.read_bytes()
    assert SECRET.encode() not in raw and b"gold_answer" not in raw and b"OPENAI_KEY" not in raw
    import sqlite3

    c = sqlite3.connect(db)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    assert {r[0] for r in c.execute("SELECT key FROM envs")} == {"snaptest/d0", "snaptest/d1", "snaptest/d4", "space:snaptest/s0",
                                                                 "space:snaptest/s1", "space:snaptest/s2"}
    assert c.execute("SELECT tasks, framework FROM envs WHERE key = 'snaptest/d0'").fetchone() == (2, "harbor")
    assert json.loads(c.execute("SELECT v FROM meta WHERE k = 'schema'").fetchone()[0]) == snapshot.SCHEMA
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"envs_trending", "envs_downloads", "envs_likes", "envs_tasks", "envs_updated", "envs_created", "tasks_env"} <= names
    assert c.execute("SELECT COUNT(*) FROM sqlite_stat1").fetchone()[0] > 0   # ANALYZE ran


def test_harbor_whenever_an_index_found_tasks(tmp_path):
    rows = [ds(0, framework="verifiers"), ds(1, framework="verifiers")]
    db, _ = build(tmp_path, rows)
    import sqlite3

    got = dict(sqlite3.connect(db).execute("SELECT id, framework FROM envs"))
    assert got == {"snaptest/d0": "harbor", "snaptest/d1": "verifiers"}


# ── publishing and swapping ──────────────────────────────────────────────────
class Recording(snapshot.LocalStore):
    def __init__(self, root):
        super().__init__(root)
        self.ops: list[tuple[str, str]] = []

    def put_file(self, src, rel):
        self.ops.append(("file", rel))
        super().put_file(src, rel)

    def put_bytes(self, rel, data):
        self.ops.append(("bytes", rel))
        super().put_bytes(rel, data)


def test_database_first_pointer_last(tmp_path):
    store = Recording(tmp_path / "store")
    db, counts = build(tmp_path)
    ptr = indexer.publish(store, db, counts)
    assert store.ops == [("file", ptr["db"]), ("bytes", snapshot.POINTER)]
    assert snapshot.read_pointer(store) == ptr and ptr["sha256"] == snapshot.sha256_file(tmp_path / "store" / ptr["db"])
    assert ptr["counts"]["envs"] == 6 and ptr["schema"] == snapshot.SCHEMA


def test_app_swaps_to_a_new_snapshot_while_a_query_finishes(tmp_path):
    store = store_of(tmp_path)
    db, counts = build(tmp_path)
    first = indexer.publish(store, db, counts)
    assert snapshot.check(store) and snapshot.current().name == first["db"]
    with snapshot.use() as (old, conn):
        assert conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] == 6
        db2, counts2 = build(tmp_path, ROWS + [ds(9)], "build2.db")
        second = indexer.publish(store, db2, counts2, now=snapshot.datetime(2030, 1, 1, tzinfo=snapshot.UTC))
        assert snapshot.check(store)                                   # swapped under a running query
        assert snapshot.current().name == second["db"] and old.name == first["db"]
        assert conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] == 6   # the old one still answers
        assert not old._closed
    assert old._closed                                                 # closed once its last query was done
    with snapshot.use() as (new, conn):
        assert new.name == second["db"] and conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] == 7
    assert not snapshot.check(store)                                   # nothing new: no work


def test_concurrent_queries_during_swaps(tmp_path):
    store = store_of(tmp_path)
    db, counts = build(tmp_path)
    indexer.publish(store, db, counts)
    snapshot.check(store)
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                with snapshot.use() as (_, conn):
                    assert conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] in (6, 7)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    for i in range(6):
        rows = ROWS + ([ds(9)] if i % 2 == 0 else [])
        dbi, ci = build(tmp_path, rows, f"b{i}.db")
        indexer.publish(store, dbi, ci, now=snapshot.datetime(2030, 1, 1, 0, 0, i, tzinfo=snapshot.UTC))
        assert snapshot.check(store)
    stop.set()
    for t in threads:
        t.join()
    assert not errors


def test_corrupt_or_mismatched_snapshots_are_rejected(tmp_path):
    store = store_of(tmp_path)
    good_db, counts = build(tmp_path)
    good = indexer.publish(store, good_db, counts)
    assert snapshot.check(store)
    root = tmp_path / "store"

    def point(name: str, data: bytes, sha: str | None = None, **extra) -> None:
        (root / name).write_bytes(data)
        ptr = {"db": name, "sha256": sha or snapshot.hashlib.sha256(data).hexdigest(), "size": len(data), "built_at": "x",
               "schema": snapshot.SCHEMA, "counts": counts, **extra}
        store.put_bytes(snapshot.POINTER, json.dumps(ptr).encode())

    raw = good_db.read_bytes()
    point("snapshots/catalog-20300101T000001Z.db", raw, sha="0" * 64)               # not the file the pointer names
    assert not snapshot.check(store) and "sha256" in snapshot.status()["error"]
    point("snapshots/catalog-20300101T000002Z.db", b"not a database" * 100)          # garbage with a matching sha
    assert not snapshot.check(store)
    point("snapshots/catalog-20300101T000003Z.db", raw[: len(raw) // 2])             # truncated
    assert not snapshot.check(store)
    point("snapshots/catalog-20300101T000004Z.db", raw, counts={**counts, "envs": 99})   # not what the pointer says
    assert not snapshot.check(store)
    store.put_bytes(snapshot.POINTER, b"{not json")
    assert not snapshot.check(store)
    store.put_bytes(snapshot.POINTER, json.dumps({"db": "../../etc/passwd", "sha256": "0" * 64, "schema": 1}).encode())
    assert not snapshot.check(store)
    assert snapshot.current().name == good["db"]                                    # the good one stayed in use
    assert not list((tmp_path / "local").glob("catalog-20300101T00000[234]Z.db"))   # rejected files aren't kept


def test_no_usable_snapshot_falls_back_to_the_live_catalog(tmp_path):
    store = store_of(tmp_path)
    (tmp_path / "store" / "snapshots").mkdir(parents=True)
    (tmp_path / "store" / "snapshots" / "catalog-20300101T000000Z.db").write_bytes(b"garbage")
    store.put_bytes(snapshot.POINTER, json.dumps({"db": "snapshots/catalog-20300101T000000Z.db", "sha256": "1" * 64,
                                                  "schema": snapshot.SCHEMA, "counts": {}}).encode())
    with snapshot.use() as (snap, conn):
        assert snap.source == "live" and conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
    assert snapshot.status()["source"] == "live" and "rejected" in snapshot.status()["error"]


def test_a_newer_schemas_pointer_is_not_read(tmp_path):
    store = store_of(tmp_path)
    store.put_bytes(f"snapshots/LATEST.v{snapshot.SCHEMA + 1}.json", b'{"db": "snapshots/catalog-1Z.db"}')
    assert snapshot.read_pointer(store) is None and not snapshot.check(store)


# ── the indexer ──────────────────────────────────────────────────────────────
def test_plan_rebuilds_only_what_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "featured_datasets", lambda: ["snaptest/f1"])
    built = 1_790_000_000.0   # 2026-09-21
    write_index("snaptest/same", [task("t")], sha="s1", built=built)
    write_index("snaptest/touched", [task("t")], sha="s1", built=built)
    write_index("snaptest/moved", [task("t")], sha="old", built=built)
    write_index("snaptest/oldfmt", [task("t")], version=catalog.INDEX_VERSION - 1, built=built)
    write_index("snaptest/part", [task("t")], built=built)
    rows = [{**ds(0), "id": i, "key": i, "updated": u, "trending": t} for i, u, t in [
        ("snaptest/same", "2026-09-01T00:00:00+00:00", 9),       # changed before its index was built: no Hub call
        ("snaptest/touched", "2026-09-30T00:00:00+00:00", 8),    # changed after, same revision: checked, skipped
        ("snaptest/moved", "2026-09-30T00:00:00+00:00", 1),      # a new revision
        ("snaptest/oldfmt", "2026-09-01T00:00:00+00:00", 7),
        ("snaptest/part", "2026-09-01T00:00:00+00:00", 6),
        ("snaptest/new", "2026-09-01T00:00:00+00:00", 5),
        ("snaptest/f1", "2026-09-01T00:00:00+00:00", 0)]]
    rows.append({**ds(0), "id": "snaptest/rows", "key": "snaptest/rows", "framework": "verl"})   # read row by row: never indexed
    asked: list[str] = []

    def info(spec):
        asked.append(spec)
        return {"sha": "new" if spec == "snaptest/moved" else "s1"}

    items = indexer.plan(rows, {"rebuild": ["snaptest/part"]}, info=info)
    assert sorted(asked) == ["snaptest/moved", "snaptest/touched"]
    assert [(i["spec"], i["reason"]) for i in items] == [("snaptest/f1", "new"), ("snaptest/moved", "changed"),
                                                        ("snaptest/oldfmt", "format"), ("snaptest/part", "partial"),
                                                        ("snaptest/new", "new")]


def test_refresh_keeps_to_its_budget_and_doesnt_retry_failures(tmp_path):
    items = [{"spec": f"snaptest/x{i}", "reason": "new", "tagged": True, "featured": False, "trending": 0, "downloads": 0, "head": None}
             for i in range(5)]
    calls: list[str] = []

    def build(spec, meta):
        calls.append(spec)
        if spec == "snaptest/x1":
            raise ValueError("more than 20,000 tasks")
        return 3

    state = indexer.load_state()
    info = lambda spec: {"sha": "r1"}
    out = indexer.refresh_indexes(items, state, budget_s=60, max_builds=3, build=build, info=info, looks_harbor=lambda s, h: True)
    assert calls == ["snaptest/x0", "snaptest/x1", "snaptest/x2"] and out["deferred"] == 2
    assert [b["spec"] for b in out["built"]] == ["snaptest/x0", "snaptest/x2"] and out["failed"][0]["spec"] == "snaptest/x1"
    calls.clear()
    out = indexer.refresh_indexes(items[1:2], state, budget_s=60, max_builds=3, build=build, info=info, looks_harbor=lambda s, h: True)
    assert calls == [] and out["skipped"][0]["why"].startswith("failed at this revision")
    out = indexer.refresh_indexes(items[1:2], state, budget_s=60, max_builds=3, build=build, info=lambda s: {"sha": "r2"},
                                  looks_harbor=lambda s, h: True)
    assert calls == ["snaptest/x1"]                                    # a new revision is tried again
    out = indexer.refresh_indexes(items, state, budget_s=0, max_builds=3, build=build, info=info)
    assert out["deferred"] == 5                                        # out of time: nothing started


def test_indexer_run_publishes_and_prunes(tmp_path, monkeypatch):
    store = Recording(tmp_path / "store")
    monkeypatch.setattr(catalog, "featured_datasets", list)
    monkeypatch.setattr(catalog, "info", lambda spec: {"sha": "zz"})
    builds: list[str] = []

    def fake_build(spec, meta):
        builds.append(spec)
        write_index(spec, [task("tasks/new")], sha=meta["sha"], built=time.time())
        return 1

    monkeypatch.setattr(indexer, "_build_one", fake_build)
    reports = []
    for i in range(4):
        reports.append(indexer.run(store, rows=ROWS, budget_s=60, max_builds=10, keep=2))
        time.sleep(1.05)   # one snapshot a second at most: names are UTC seconds
    # d4 and d0 changed revision (the Hub says zz), d1 was never indexed; after that, all current: no more builds
    assert builds == ["snaptest/d4", "snaptest/d0", "snaptest/d1"]
    assert [r["indexes"]["built"] for r in reports[1:]] == [[], [], []]
    ptr = snapshot.read_pointer(store)
    assert ptr == reports[-1]["pointer"] and ptr["counts"]["tasks"] == 3
    assert store.ops[-2:] == [("file", ptr["db"]), ("bytes", snapshot.POINTER)]
    dbs = [n for n in store.list("snapshots") if n.endswith(".db")]
    assert len(dbs) == 2 and ptr["db"] in dbs                          # pruned to --keep
    assert (config.STORAGE_DIR / "listing.json.gz").exists()
    assert snapshot.check(store) and snapshot.current().counts["envs"] == 6


def test_cached_listing_publication_preserves_discovery_timestamp(tmp_path):
    at = time.time() - 1800
    indexer.run(snapshot.LocalStore(tmp_path / "store"), rows=ROWS, index=False, listing_at=at)
    data = json.loads(gzip.decompress((config.STORAGE_DIR / "listing.json.gz").read_bytes()))
    assert data["at"] == at
    assert catalog._env_list["at"] == at


def test_timed_out_build_backs_off_so_later_batches_can_progress(monkeypatch):
    class Stalled:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def join(self, **kwargs): pass
        def is_alive(self): return True
    monkeypatch.setattr(indexer.threading, "Thread", Stalled)
    items = [{"spec": "snaptest/large", "reason": "new", "tagged": True}]
    state = {"failed": {}, "not_harbor": {}, "rebuild": []}
    info = lambda spec: {"sha": "same-revision"}
    first = indexer.refresh_indexes(items, state, budget_s=90, max_builds=4, info=info)
    assert first["abandoned"] == ["snaptest/large"]
    second = indexer.refresh_indexes(items, state, budget_s=90, max_builds=4, info=info)
    assert second["abandoned"] == [] and second["skipped"][0]["spec"] == "snaptest/large"


def test_a_failed_run_publishes_nothing(tmp_path, monkeypatch):
    store = store_of(tmp_path)
    first = indexer.run(store, rows=ROWS, index=False)["pointer"]
    time.sleep(1.05)
    monkeypatch.setattr(snapshot, "build_db", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        indexer.run(store, rows=ROWS, index=False)
    assert snapshot.read_pointer(store) == first


def test_a_listing_that_shrank_is_refused(tmp_path):
    store = store_of(tmp_path)
    first = indexer.run(store, rows=ROWS, index=False)["pointer"]
    with pytest.raises(indexer.IndexerError, match="refusing to publish"):
        indexer.run(store, rows=ROWS[:2], index=False)               # the Spaces' listing failed
    assert snapshot.read_pointer(store) == first
    time.sleep(1.05)
    assert indexer.run(store, rows=ROWS[:2], index=False, force=True)["pointer"]["counts"]["envs"] == 2


def test_dry_run_writes_nothing(tmp_path):
    store = store_of(tmp_path)
    out = tmp_path / "dry.db"
    report = indexer.run(store, rows=ROWS, dry_run=True, out=out)
    assert report["counts"]["envs"] == 6 and out.exists()
    assert snapshot.read_pointer(store) is None and not (tmp_path / "store").exists()


def test_main_exit_status(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(config.STORAGE_DIR))   # main() points it at --store: restored after
    monkeypatch.setattr(indexer, "fetch_listing", lambda attempts=2: (_ for _ in ()).throw(indexer.IndexerError("Hub down")))
    monkeypatch.setattr("app.runtime.logs", lambda: None)
    assert indexer.main(["--store", str(tmp_path / "store"), "--no-index"]) == 1
    assert not (tmp_path / "store" / snapshot.POINTER).exists()


# ── through the bucket API ───────────────────────────────────────────────────
class FakeBucketApi:
    """The four bucket calls the store uses, over a dict."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.batches: list[dict] = []

    def batch_bucket_files(self, bucket_id, *, add=None, delete=None, copy=None, token=None):
        self.batches.append({"add": [p for _, p in add or []], "delete": list(delete or [])})
        for src, path in add or []:
            self.files[path] = src if isinstance(src, bytes) else Path(src).read_bytes()
        for path in delete or []:
            self.files.pop(path, None)

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        for path, dst in files:
            if path not in self.files:
                if raise_on_missing_files:
                    raise FileNotFoundError(path)
                continue
            Path(dst).write_bytes(self.files[path])

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        return [SimpleNamespace(type="file", path=p, size=len(b)) for p, b in sorted(self.files.items()) if p.startswith(prefix or "")]

    def get_bucket_paths_info(self, bucket_id, paths, *, token=None):
        return [SimpleNamespace(type="file", path=p, size=len(self.files[p])) for p in paths if p in self.files]


def test_the_same_through_the_bucket_api(tmp_path):
    api = FakeBucketApi()
    store = snapshot.BucketStore("FineEnvs/rl-explorer-data", api=api)
    report = indexer.run(store, rows=ROWS, index=False, keep=1)
    ptr = report["pointer"]
    assert [b["add"] for b in api.batches] == [[ptr["db"]], [snapshot.POINTER]]   # the database, then the pointer
    assert snapshot.check(store) and snapshot.current().counts["envs"] == 6
    time.sleep(1.05)
    indexer.run(store, rows=ROWS, index=False, keep=1)
    assert api.batches[-1]["delete"] == [ptr["db"]]                                 # pruned through the API too
    assert str(snapshot.open_store("hf://buckets/FineEnvs/rl-explorer-data/sub")).endswith("rl-explorer-data/sub")


def teardown_module(module):
    shutil.rmtree(Path(config.STORAGE_DIR) / "snapshots", ignore_errors=True)


def test_a_snapshot_whose_file_vanished_is_dropped_and_fetched_again(tmp_path):
    store = store_of(tmp_path)
    db, counts = build(tmp_path)
    indexer.publish(store, db, counts)
    assert snapshot.check(store)
    snap = snapshot.current()
    snap._close()                    # its pooled connections gone, as after a while idle
    snap._closed = False
    snap.path.unlink()               # a disk cleaner took the file
    with snapshot.use() as (s, conn):   # this request is still answered: the snapshot is fetched again at once
        assert conn.execute("SELECT COUNT(*) FROM envs").fetchone()[0] == 6
    assert s is not snap and s.name == snap.name and s.path.exists() and snapshot.current() is s


def test_fallback_files_of_dead_processes_are_cleaned(tmp_path):
    local = tmp_path / "local"
    local.mkdir(parents=True)
    dead, mine = local / "fallback-999999-1.db", local / f"fallback-{snapshot.os.getpid()}-77.db"
    dead.write_bytes(b"x")
    mine.write_bytes(b"x")
    with snapshot.use() as (snap, _):
        assert snap.source == "live"
    assert not dead.exists() and mine.exists()
