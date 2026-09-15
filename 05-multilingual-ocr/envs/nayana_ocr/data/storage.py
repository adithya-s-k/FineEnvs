"""Fetch bucket row groups into a byte-bounded local Arrow cache."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token

from .cache import DiskCache, RangeReader
from .index import sha256_file


class CorpusStorage:
    def __init__(
        self,
        manifest,
        cache_dir,
        *,
        source_root=None,
        local_source=False,
        group_cache_bytes=4_000_000_000,
        max_group_bytes=512_000_000,
        prefetch_workers=2,
        prefetch_pending=4,
    ):
        if not 1 <= prefetch_workers <= prefetch_pending <= 32:
            raise ValueError("Use 1 <= prefetch_workers <= prefetch_pending <= 32")
        self.manifest = manifest
        self.source_root = Path(source_root) if source_root else None
        self.local_source = local_source
        self.cache = DiskCache(
            Path(cache_dir) / "groups", group_cache_bytes, max_group_bytes
        )
        self.api = HfApi()
        token = get_token()
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.executor = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="nayana-prefetch"
        )
        self._load_slots = threading.BoundedSemaphore(prefetch_workers)
        self.pending = {}
        self._pending_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(prefetch_pending)
        self._local_verified = {}
        self._verified_lock = threading.Lock()

    def _verify(self, spec):
        if self.local_source:
            if self.source_root is None or not spec.get("sha256"):
                raise ValueError(
                    "Offline sources need a root directory and SHA-256 inventory"
                )
            path = self.source_root / spec["path"]
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            with self._verified_lock:
                if self._local_verified.get(spec["path"]) != signature:
                    if (
                        signature[0] != spec["size"]
                        or sha256_file(path) != spec["sha256"]
                    ):
                        raise ValueError("Local source changed from indexed corpus")
                    self._local_verified[spec["path"]] = signature
        else:
            rows = list(
                self.api.get_bucket_paths_info(
                    self.manifest["bucket_id"], [spec["path"]]
                )
            )
            if (
                len(rows) != 1
                or rows[0].xet_hash != spec["xet_hash"]
                or rows[0].size != spec["size"]
            ):
                raise ValueError(
                    "Bucket source changed from indexed corpus; refusing stale task identity"
                )

    @contextmanager
    def _open(self, spec):
        if self.source_root:
            with (self.source_root / spec["path"]).open("rb") as source:
                yield source
        else:
            # The documented bucket resolve endpoint supports HTTP Range.
            url = f"https://huggingface.co/buckets/{self.manifest['bucket_id']}/resolve/{quote(spec['path'])}"
            with RangeReader(
                url,
                spec["size"],
                headers=self.headers,
                on_read=lambda n: self.cache.increment("remote_bytes", n),
            ) as source:
                yield source

    def key(self, spec, row_group):
        return f"{spec['xet_hash']}:{row_group}:jpg-and-page-id-v1"

    @contextmanager
    def group(self, spec, row_group):
        key = self.key(spec, row_group)

        def load(path):
            with self._load_slots:
                self._verify(spec)
                with self._open(spec) as source:
                    parquet = pq.ParquetFile(source, pre_buffer=True)
                    group = parquet.metadata.row_group(row_group)
                    if group.total_byte_size > self.cache.max_entry_bytes:
                        raise ValueError(
                            "Source row group exceeds the configured entry budget"
                        )
                    table = parquet.read_row_group(
                        row_group, columns=["jpg", "image_id.txt"], use_threads=False
                    )
                    if table.nbytes > self.cache.max_entry_bytes:
                        raise ValueError(
                            "Decoded row group exceeds the configured entry budget"
                        )
                    with pa.OSFile(str(path), "wb") as sink:
                        with pa.ipc.new_file(sink, table.schema) as writer:
                            writer.write_table(table)
                self._verify(spec)  # Detect source mutation during range/mount reads.

        with self.cache.lease(key, load) as path:
            with pa.memory_map(str(path), "r") as source:
                yield pa.ipc.open_file(source).read_all()

    def image(self, spec, row_group, offset, expected_page):
        with self.group(spec, row_group) as table:
            row = table.slice(offset, 1).to_pylist()
            if len(row) != 1 or row[0]["image_id.txt"] != expected_page:
                raise ValueError("Image row does not match indexed page identity")
            return row[0]["jpg"]["bytes"]

    def prefetch(self, items):
        accepted, busy = 0, 0
        for spec, row_group in items:
            key = self.key(spec, row_group)
            with self._pending_lock:
                if key in self.pending:
                    accepted += 1
                    continue
                if not self._slots.acquire(blocking=False):
                    busy += 1
                    continue
                future = self.executor.submit(self._warm, spec, row_group)
                self.pending[key] = future
            future.add_done_callback(lambda f, k=key: self._finished(k, f))
            accepted += 1
        return {"accepted_groups": accepted, "busy_groups": busy}

    def _warm(self, spec, row_group):
        with self.group(spec, row_group):
            pass

    def _finished(self, key, future):
        if future.exception() is not None:
            self.cache.increment("prefetch_failures")
        else:
            self.cache.increment("prefetch_completed")
        with self._pending_lock:
            self.pending.pop(key, None)
        self._slots.release()

    def stats(self):
        with self._pending_lock:
            pending = len(self.pending)
        return {
            **self.cache.stats(),
            "pending_groups": pending,
            "transport": "mounted-filesystem" if self.source_root else "http-range",
        }

    def close(self):
        self.executor.shutdown(wait=True)


def load_manifest(value):
    """Manifests are small; indexes and data are fetched separately, on demand."""
    import requests

    if str(value).startswith("hf://buckets/"):
        path = str(value).removeprefix("hf://buckets/")
        namespace, bucket, key = path.split("/", 2)
        token = get_token()
        response = requests.get(
            f"https://huggingface.co/buckets/{namespace}/{bucket}/resolve/{quote(key)}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=120,
        )
        response.raise_for_status()
        raw = response.content
        if len(raw) > 16_000_000:
            raise ValueError("Corpus manifest exceeds 16 MB")
        return json.loads(raw), str(value).rsplit("/", 1)[0]
    path = Path(value)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    directory = str(path.parent.resolve())
    if manifest.get("storage") == "bucket-parquet" and not all(
        (path.parent / info["path"]).is_file() for info in manifest["indexes"].values()
    ):
        directory = f"hf://buckets/{manifest['bucket_id']}/openenv/indexes/{manifest['snapshot_id']}"
    return manifest, directory
