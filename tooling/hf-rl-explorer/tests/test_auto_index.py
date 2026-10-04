import gzip
import json
import time

import pytest

from app import auto_index, config, indexer, snapshot


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("RLX_SNAPSHOT_STORE", str(tmp_path))


def test_recent_full_listing_is_reused_without_extending_discovery_age(monkeypatch):
    at = time.time() - 300
    rows = [{"id": "owner/env", "kind": "dataset"}]
    (config.STORAGE_DIR / "listing.json.gz").write_bytes(gzip.compress(json.dumps({"at": at, "full": True, "rows": rows}).encode()))
    called = []
    def run(store, **kwargs):
        called.append(kwargs)
        return {"counts": {"envs": 1, "tasks": 2}}
    monkeypatch.setattr(indexer, "run", run)
    auto_index.cycle()
    assert called == [{"rows": rows, "listing_at": at, "budget_s": 90, "max_builds": 4}]
    assert auto_index.status()["state"] == "idle"
    assert auto_index.status()["last_success"] >= at
    auto_index.cycle()
    assert len(called) == 1  # app restarts and duplicate workers honor cadence
    assert auto_index.cached_listing(at + 3601) == (None, None)
    assert auto_index.cached_listing(at - 1) == (None, None)


def test_locked_worker_cannot_publish_duplicate_snapshot(monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    config.CACHE_DIR.mkdir()
    monkeypatch.setattr(indexer, "run", lambda *a, **k: pytest.fail("duplicate indexer"))
    with (config.CACHE_DIR / "auto-index.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auto_index.cycle()
    assert auto_index.status() == {}


def test_failure_preserves_published_pointer_and_backs_off(monkeypatch):
    store = snapshot.open_store()
    store.put_bytes(snapshot.POINTER, b"last-good-snapshot")
    def fail(*a, **k):
        raise ConnectionError("upstream response must not enter public status")
    monkeypatch.setattr(indexer, "run", fail)
    auto_index.cycle()
    first = auto_index.status()
    assert first["state"] == "retrying" and first["error"] == "ConnectionError"
    assert first["next_at"] > time.time() + 590
    assert store.read_bytes(snapshot.POINTER) == b"last-good-snapshot"
    auto_index._save({**first, "next_at": 0})
    auto_index.cycle()
    assert auto_index.status()["failures"] == 2
    assert auto_index.status()["next_at"] > time.time() + 1190


def test_missing_or_partial_listing_requires_fresh_discovery():
    assert auto_index.cached_listing(time.time()) == (None, None)
    doc = {"at": time.time(), "full": False, "rows": [{"id": "owner/partial"}]}
    (config.STORAGE_DIR / "listing.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
    assert auto_index.cached_listing(time.time()) == (None, None)


def test_shutdown_terminates_only_owned_indexer_child(monkeypatch):
    calls = []
    class Child:
        def poll(self): return None
        def terminate(self): calls.append("terminate")
        def wait(self, **kwargs): calls.append("wait")
    def popen(command, **kwargs):
        assert command[-2:] == ["-m", "app.auto_index"]
        assert kwargs["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        return Child()
    monkeypatch.setattr(auto_index.subprocess, "Popen", popen)
    auto_index._stop.set()
    try:
        auto_index._child()
    finally:
        auto_index._stop.clear()
    assert calls == ["terminate", "wait"]
