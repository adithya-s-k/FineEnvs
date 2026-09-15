import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nayana_ocr.corpus_training import BlockTaskStream
from nayana_ocr.data.cache import DiskCache
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.index import build_index, sha256_file
from nayana_ocr.data.schema import REVISION, canonical_json
from nayana_ocr.fixtures import fixture_rows
from nayana_ocr.models import NayanaAction
from nayana_ocr.server.environment import NayanaEnvironment


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "source"
    files = []
    for language in ("en", "ar"):
        rows = list(fixture_rows(language))
        for index in range(2):
            path = root / language / f"train-{index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows[index * 3 : (index + 1) * 3]),
                path,
                row_group_size=2,
            )
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "xet_hash": sha256_file(path),
                }
            )
    inventory = {
        "bucket_id": "fixture/corpus",
        "source": "synthetic-fixture",
        "revision": REVISION,
        "source_license": "synthetic-fixture",
        "files": files,
        "source_bytes": sum(f["size"] for f in files),
        "parquet_files": len(files),
    }
    inventory["inventory_id"] = hashlib.sha256(
        canonical_json(inventory).encode()
    ).hexdigest()
    output = tmp_path / "index"
    with monkeypatch.context() as context:
        # A complete index must not decode or materialize any page image.
        context.setattr(
            "nayana_ocr.data.tasks.Image.open",
            lambda *a, **kw: pytest.fail("Index touched image decoder"),
        )
        manifest = build_index(
            inventory, output, languages=("en", "ar"), source_root=root, workers=2
        )
    catalog = CorpusCatalog(
        output,
        tmp_path / "cache",
        source_root=root,
        local_source=True,
        group_cache_bytes=2_000_000,
        max_group_bytes=1_000_000,
        asset_cache_bytes=100_000,
    )
    yield catalog, manifest, root, output, inventory
    catalog.close()


def test_complete_index_global_and_group_random_access_without_media(corpus):
    catalog, manifest, _, output, inventory = corpus
    assert sum(manifest["pages"].values()) == 12
    assert sum(c["tasks"] for c in manifest["counts"]) == 48
    assert catalog.count("test") == 16
    for split in ("train", "validation", "test"):
        rows = catalog.task_range(split, 0, catalog.count(split))
        assert len({r["task_id"] for r in rows}) == 16
        assert all("reference" not in r and r["asset_path"] is None for r in rows)
        last = catalog.at(split, catalog.count(split) - 1)
        assert catalog.get(last["task_id"]) == last
        for language in ("en", "ar"):
            task = catalog.group_at(split, language, "page_ocr", 1)
            assert (
                catalog.group_position(task["task_id"], split, language, "page_ocr")
                == 1
            )
    assert catalog.stats()["groups"].get("loads", 0) == 0
    assert catalog.stats()["assets"].get("loads", 0) == 0
    with pytest.raises(IndexError):
        catalog.at("train", catalog.count("train"))
    with pytest.raises(KeyError):
        catalog.get(
            catalog.at("train", 0)["task_id"].replace(manifest["snapshot_id"], "0" * 64)
        )
    assert (
        build_index(inventory, output, languages=("ar", "en"), source_root=corpus[2])[
            "snapshot_id"
        ]
        == manifest["snapshot_id"]
    )


def test_prefetch_and_repeated_rollouts_share_group_and_render(corpus):
    catalog, _, _, _, _ = corpus
    task = catalog.group_at("train", "en", "page_ocr", 0)
    catalog.prefetch(task_ids=[task["task_id"], task["task_id"]])
    deadline = time.monotonic() + 5
    while catalog.stats()["groups"]["pending_groups"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert catalog.stats()["groups"]["loads"] == 1
    a, b = NayanaEnvironment(catalog), NayanaEnvironment(catalog)
    first = a.reset(task_id=task["task_id"])
    second = b.reset(task_id=task["task_id"])
    assert first.model_dump() == second.model_dump()
    assert catalog.stats()["assets"]["loads"] == 1
    assert catalog.stats()["groups"]["loads"] == 1
    assert a.step(NayanaAction(answer="")).reward == 0
    assert b.step(NayanaAction(answer=task["reference"])).reward == 1
    data, mime = catalog.asset_bytes(first.asset_sha256, first.task_id)
    assert (
        hashlib.sha256(data).hexdigest() == first.asset_sha256 and mime == "image/png"
    )
    assert catalog.stats()["groups"]["loads"] == 1
    with pytest.raises(KeyError):
        catalog.asset_bytes("0" * 64, first.task_id)


def test_block_plan_covers_all_selected_tasks_once_and_source_changes_fail(corpus):
    catalog, _, root, _, _ = corpus
    blocks = catalog.blocks("train", ["ar", "en"], ["section_ocr", "page_ocr"])
    rows = [
        r
        for block in blocks
        for r in catalog.block_tasks(
            block["block_id"], "train", ["section_ocr", "page_ocr"]
        )
    ]
    assert len(rows) == len({r["task_id"] for r in rows}) == 8
    selected = catalog.sample("test", ["en", "ar"], ["page_ocr"], 2, 42)
    assert selected == catalog.sample("test", ["en", "ar"], ["page_ocr"], 2, 42)
    task = catalog.get(rows[-1]["task_id"])
    path = root / task["_source_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source changed"):
        catalog.materialize(task)


def test_cache_single_flight_eviction_pins_and_failure_cleanup(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=20)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def loader(path):
        calls.append(path)
        entered.set()
        assert release.wait(5)
        path.write_bytes(b"a" * 10)

    def read():
        with cache.lease("a", loader) as path:
            return path.read_bytes()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read)
        assert entered.wait(5)
        second = pool.submit(read)
        release.set()
        assert first.result() == second.result() == b"a" * 10
    assert len(calls) == 1
    with cache.lease("a", loader) as pinned:
        with cache.lease("b", lambda p: p.write_bytes(b"b" * 10)):
            pass
        with cache.lease("c", lambda p: p.write_bytes(b"c" * 10)):
            pass
        assert pinned.read_bytes() == b"a" * 10
        assert cache.stats()["bytes"] == 20 and cache.stats()["evictions"] == 1

    def broken(path):
        path.write_bytes(b"partial")
        raise OSError("interrupted")

    with pytest.raises(OSError):
        with cache.lease("broken", broken):
            pass
    assert not list(tmp_path.glob("*.part"))
    with cache.lease("broken", lambda p: p.write_bytes(b"ok")) as path:
        assert path.read_bytes() == b"ok"


def test_index_content_change_is_rejected(corpus):
    catalog, manifest, _, output, _ = corpus
    path = output / "en.sqlite"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="index changed"):
        catalog.group_at("train", "en", "page_ocr", 0)
    assert len(manifest["snapshot_id"]) == 64


def test_reindex_reuses_verified_annotations_without_source_reads(
    corpus, tmp_path, monkeypatch
):
    catalog, manifest, _, output, inventory = corpus
    monkeypatch.setattr(
        "nayana_ocr.data.index._read_metadata",
        lambda *a, **kw: pytest.fail("Reindex read source Parquet"),
    )
    rebuilt = build_index(
        inventory, tmp_path / "rebuilt", languages=("en", "ar"), reuse_index=output
    )
    assert rebuilt["snapshot_id"] == manifest["snapshot_id"]
    assert rebuilt["counts"] == manifest["counts"]


def test_full_epoch_cursor_is_bounded_replayable_and_disjoint_across_ranks(corpus):
    catalog, _, _, _, _ = corpus
    options = dict(
        split="train",
        languages=["en", "ar"],
        families=["page_ocr", "section_ocr"],
        seed=13,
        prefetch_blocks=0,
        chunk_size=2,
    )
    full = list(BlockTaskStream(catalog, **options))
    assert len(full) == len({r["task_id"] for r in full}) == 8
    assert full == list(BlockTaskStream(catalog, **options))
    stream = BlockTaskStream(catalog, **options)
    iterator = iter(stream)
    prefix = [next(iterator) for _ in range(3)]
    saved = stream.state_dict()
    resumed = BlockTaskStream(catalog, **options)
    resumed.load_state_dict(saved)
    assert prefix + list(resumed) == full
    ranks = [
        list(BlockTaskStream(catalog, **options, rank=r, world_size=2))
        for r in range(2)
    ]
    ids = [{row["task_id"] for row in rows} for rows in ranks]
    assert not ids[0] & ids[1] and ids[0] | ids[1] == {r["task_id"] for r in full}
    with pytest.raises(ValueError, match="plan"):
        BlockTaskStream(catalog, **{**options, "seed": 14}).load_state_dict(saved)
    assert catalog.stats()["groups"].get("loads", 0) == 0


def test_full_corpus_http_websocket_and_trl_dataloader(corpus, tmp_path):
    import requests
    from nayana_ocr.corpus_training import CorpusAPI, build_corpus_dataset
    from nayana_ocr.runtime import local_server
    from nayana_ocr.smoke import probe
    from test_training import assert_trl_groups

    catalog, manifest, root, output, _ = corpus
    with local_server(
        output, source_root=root, local_source=True, cache_dir=tmp_path / "server-cache"
    ) as url:
        api = CorpusAPI(url, manifest["snapshot_id"])
        try:
            rows = list(BlockTaskStream(api, prefetch_blocks=2))
            assert len(rows) == len({r["task_id"] for r in rows}) == 16
            assert (
                requests.post(
                    url + "/data/blocks", json={"snapshot_id": "wrong"}
                ).status_code
                == 409
            )
            assert (
                requests.post(
                    url + "/data/block-tasks",
                    json={
                        "snapshot_id": manifest["snapshot_id"],
                        "block_id": "en:1",
                        "limit": 1001,
                    },
                ).status_code
                == 422
            )
            result = probe(url, catalog)
            assert result["exact_reference_checked"] and len(result["coverage"]) == 8
            dataset = build_corpus_dataset(
                url,
                manifest["snapshot_id"],
                ["en", "ar"],
                ["section_ocr", "mcq_vqa", "page_ocr"],
            )
            # The actual VLM runner obtains prompts from environment.reset. A tiny
            # CPU text model exercises the installed TRL data loader without weights.
            assert_trl_groups(
                tmp_path, dataset.map(lambda row: {**row, "prompt": "hello"})
            )
        finally:
            api.close()


def test_asset_reconstruction_after_eviction(corpus):
    catalog, _, _, _, _ = corpus
    task = catalog.materialize(catalog.group_at("train", "en", "page_ocr", 0))
    raw, _ = catalog.asset_bytes(task["asset_sha256"], task["task_id"])
    for path in catalog.asset_cache.directory.glob("*.bin"):
        path.unlink()
    rebuilt, _ = catalog.asset_bytes(task["asset_sha256"], task["task_id"])
    assert rebuilt == raw
    assert catalog.stats()["assets"]["loads"] == 2
    assert catalog.stats()["groups"]["loads"] == 1


def test_http_range_reader_seek_and_fail_closed():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from nayana_ocr.data.cache import RangeReader

    data = b"0123456789abcdefghij"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, stop = map(
                int, self.headers["Range"].removeprefix("bytes=").split("-")
            )
            payload = data[start : stop + 1]
            self.send_response(200 if self.path == "/ignored" else 206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{stop}/{len(data) if self.path != '/changed' else 99}",
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        received = []
        with RangeReader(url, len(data), on_read=received.append) as source:
            assert source.read(3) == b"012"
            assert source.seek(-4, 2) == 16
            assert source.read(100) == b"ghij"
            assert source.read() == b""
            source.seek(5)
            target = bytearray(2)
            assert source.readinto(target) == 2 and target == b"56"
        assert sum(received) == 9
        for path in ("ignored", "changed"):
            with RangeReader(url + "/" + path, len(data)) as source:
                with pytest.raises(ValueError):
                    source.read(3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cache_lock_stripe_collision_can_evict_old_entry(tmp_path):
    seen = {}
    for i in range(10000):
        key = str(i)
        prefix = hashlib.sha256(key.encode()).hexdigest()[:3]
        if prefix in seen:
            first, second = seen[prefix], key
            break
        seen[prefix] = key
    else:
        pytest.fail("Could not find a lock stripe collision")
    cache = DiskCache(tmp_path, max_bytes=10)
    with cache.lease(first, lambda p: p.write_bytes(b"a" * 6)):
        pass
    with cache.lease(second, lambda p: p.write_bytes(b"b" * 6)) as path:
        assert path.read_bytes() == b"b" * 6
    assert cache.stats()["entries"] == 1 and cache.stats()["evictions"] == 1
    assert len(list(tmp_path.glob("stripe-*.lock"))) == 1


def test_cache_restart_cleans_orphans_but_preserves_active_staging(tmp_path):
    orphan = tmp_path / ("a" * 64 + ".old.part")
    orphan.write_bytes(b"interrupted")
    cache = DiskCache(tmp_path, max_bytes=20)
    assert not orphan.exists()
    writing, release = threading.Event(), threading.Event()

    def loader(path):
        path.write_bytes(b"live")
        writing.set()
        assert release.wait(5)

    def worker():
        with cache.lease("active", loader) as path:
            return path.read_bytes()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        assert writing.wait(5)
        DiskCache(tmp_path, max_bytes=20)
        assert len(list(tmp_path.glob("*.part"))) == 1
        release.set()
        assert future.result() == b"live"


def test_manifest_cannot_change_source_hashes_without_changing_identity(
    corpus, tmp_path
):
    import copy
    import json

    _, manifest, _, _, _ = corpus
    for field, value in (("xet_hash", "a" * 64), ("sha256", "b" * 64), ("size", 1)):
        changed = copy.deepcopy(manifest)
        changed["source_files"][0][field] = value
        path = tmp_path / "modified-manifest.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="source inventory identity mismatch"):
            CorpusCatalog(path, tmp_path / "invalid-cache")
