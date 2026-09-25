"""Fetch bucket row groups into a byte-bounded local cache. Audio stays in the bucket.

Mirrors 05-multilingual-ocr's storage: source files are never copied, a read is a range
request or a mounted read, and what lands on disk has an explicit budget. The one number
that differs matters a lot — FLEURS publishes **one row group per file**, 310 MB for a
test shard and up to 1.5 GB for a train shard, against Nayana's ~75-116 MB groups. A cold
read is therefore expensive and a warm one is free, so task order decides throughput:
serve a language/split together and the shard is fetched once.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token

from .cache import DiskCache, RangeReader


def runtime_bucket_id(manifest):
    """Where to read the bucket right now, which need not be its recorded name.

    ``manifest["bucket_id"]`` is provenance and is hashed into the snapshot identity, so
    an organization rename must not rewrite it. HTTP reads survive a rename through a
    redirect and a mount does not, so FLEURS_BUCKET_ID addresses the live bucket without
    disturbing the recorded identity.
    """
    return os.environ.get("FLEURS_BUCKET_ID") or manifest["bucket_id"]


class CorpusStorage:
    def __init__(
        self,
        manifest,
        cache_dir,
        *,
        source_root=None,
        local_source=False,
        group_cache_bytes=8_000_000_000,
        max_group_bytes=2_000_000_000,
        prefetch_workers=2,
        prefetch_pending=4,
    ):
        if not 1 <= prefetch_workers <= prefetch_pending <= 32:
            raise ValueError("Use 1 <= prefetch_workers <= prefetch_pending <= 32")
        self.manifest = manifest
        self.source_root = Path(source_root) if source_root else None
        self.local_source = local_source
        # A single FLEURS shard can be 1.5 GB, so the per-entry ceiling is far above
        # Nayana's: a budget that cannot hold one train shard can never serve one.
        self.cache = DiskCache(
            Path(cache_dir) / "groups", group_cache_bytes, max_group_bytes
        )
        self.api = HfApi()
        token = get_token()
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._pending = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=prefetch_workers)
        self._pending_limit = prefetch_pending
        self._verified = {}

    def _verify(self, spec):
        """Refuse a source that is not the one the index was built from."""
        if self.source_root is not None:
            path = self.source_root / spec["path"]
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            if self._verified.get(spec["path"]) == signature:
                return
            if signature[0] != spec["size"]:
                raise ValueError(
                    f"{spec['path']}: local source is {signature[0]} bytes, indexed at "
                    f"{spec['size']}"
                )
            self._verified[spec["path"]] = signature
            return
        if spec["path"] in self._verified:
            return
        rows = list(
            self.api.get_bucket_paths_info(
                runtime_bucket_id(self.manifest), [spec["path"]]
            )
        )
        if len(rows) != 1 or rows[0].size != spec["size"]:
            raise ValueError(
                "Bucket source changed from the indexed corpus; refusing a stale task identity"
            )
        self._verified[spec["path"]] = True

    @contextmanager
    def _open(self, spec):
        if self.source_root:
            with (self.source_root / spec["path"]).open("rb") as source:
                yield source
        else:
            url = (
                f"https://huggingface.co/buckets/{runtime_bucket_id(self.manifest)}"
                f"/resolve/{quote(spec['path'])}"
            )
            with RangeReader(
                url,
                spec["size"],
                headers=self.headers,
                on_read=lambda n: self.cache.increment("remote_bytes", n),
            ) as source:
                yield source

    @staticmethod
    def key(spec, row_group):
        return f"{spec['path']}#{row_group}"

    @contextmanager
    def group(self, spec, row_group):
        """Yield the cached Arrow table for one row group, fetching it if cold."""
        self._verify(spec)

        def load(path):
            import pyarrow as pa

            with self._open(spec) as source:
                table = pq.ParquetFile(source).read_row_group(row_group)
            with pa.OSFile(str(path), "wb") as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)

        with self.cache.lease(self.key(spec, row_group), load) as path:
            import pyarrow as pa

            with pa.memory_map(str(path), "rb") as handle:
                yield pa.ipc.open_file(handle).read_all()

    def audio(self, spec, row_group, offset, expected_id):
        """Return one utterance's bytes, checking it is the row the index promised."""
        with self.group(spec, row_group) as table:
            row = table.slice(offset, 1).to_pylist()[0]
            if row.get("id") != expected_id:
                raise ValueError(
                    f"{spec['path']}#{row_group}[{offset}] is utterance {row.get('id')}, "
                    f"indexed as {expected_id}"
                )
            media = (row.get("audio") or {}).get("bytes")
            if not isinstance(media, bytes) or not media:
                raise ValueError(f"utterance {expected_id} carries no audio bytes")
            return media

    def prefetch(self, items):
        """Warm upcoming shards without blocking the caller."""
        started = 0
        for spec, row_group in items:
            key = self.key(spec, row_group)
            with self._lock:
                if key in self._pending or len(self._pending) >= self._pending_limit:
                    continue
                future = self._pool.submit(self._warm, spec, row_group)
                self._pending[key] = future
            future.add_done_callback(lambda f, k=key: self._finished(k, f))
            started += 1
        return {"started": started, "pending": len(self._pending)}

    def _warm(self, spec, row_group):
        with self.group(spec, row_group):
            return True

    def _finished(self, key, future):
        with self._lock:
            self._pending.pop(key, None)
        future.exception()  # Surface a failure at the next real read, not silently.

    def stats(self):
        return {
            **self.cache.stats(),
            "pending_groups": len(self._pending),
            "transport": "mounted-filesystem" if self.source_root else "http-range",
        }

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def load_manifest(value):
    """Resolve a manifest path or hf:// URI to (manifest, index directory)."""
    if str(value).startswith("hf://"):
        raise ValueError("Pass a local manifest path; index files are fetched lazily")
    path = Path(value)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    directory = str(path.parent.resolve())
    if manifest.get("storage") == "bucket-parquet" and not all(
        (path.parent / info["path"]).is_file() for info in manifest["indexes"].values()
    ):
        directory = (
            f"hf://buckets/{runtime_bucket_id(manifest)}"
            f"/openenv/indexes/{manifest['snapshot_id']}"
        )
    return manifest, directory
