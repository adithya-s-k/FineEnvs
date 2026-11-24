"""Bounded disk caches with single-flight loads, atomic writes, and reader leases."""

import fcntl
import hashlib
import io
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import requests


class DiskCache:
    def __init__(self, directory, max_bytes, max_entry_bytes=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.max_entry_bytes = min(int(max_entry_bytes or max_bytes), self.max_bytes)
        if self.max_entry_bytes < 1:
            raise ValueError("Cache budgets must be positive")
        self.counters = Counter()
        self._stats_lock = threading.Lock()
        # Recover staging space after a process was killed before its finally block.
        # A live writer owns this stripe, including writers in another process.
        for path in self.directory.glob("*.part"):
            stripe = path.name[:3]
            with (self.directory / f"stripe-{stripe}.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                path.unlink(missing_ok=True)

    def increment(self, key, value=1):
        with self._stats_lock:
            self.counters[key] += value

    def stats(self):
        with self._stats_lock:
            result = dict(self.counters)
        files = list(self.directory.glob("*.bin"))
        sizes = []
        for path in files:
            try:
                sizes.append(path.stat().st_size)
            except FileNotFoundError:
                pass
        return {
            **result,
            "bytes": sum(sizes),
            "entries": len(sizes),
            "max_bytes": self.max_bytes,
            "max_entry_bytes": self.max_entry_bytes,
        }

    @contextmanager
    def lease(self, key, loader):
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = self.directory / f"{digest}.bin"
        # Fixed lock stripes avoid millions of permanent lock files over full epochs.
        # A stripe covers both loading and reading; collisions only serialize work.
        stripe = digest[:3]
        with (self.directory / f"stripe-{stripe}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                self.increment("hits")
            else:
                self.increment("misses")
                temporary = self.directory / f"{digest}.{uuid.uuid4().hex}.part"
                try:
                    loader(temporary)
                    size = temporary.stat().st_size
                    if size > self.max_entry_bytes:
                        raise ValueError(
                            f"Cache entry {size} exceeds {self.max_entry_bytes} byte limit"
                        )
                    with (self.directory / ".eviction.lock").open("a+b") as eviction:
                        fcntl.flock(eviction, fcntl.LOCK_EX)
                        self._make_room(size, stripe)
                        os.replace(temporary, path)
                    self.increment("loads")
                    self.increment("loaded_bytes", size)
                except BaseException:
                    self.increment("load_failures")
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
            os.utime(path, None)
            yield path

    def _make_room(self, needed, owned_stripe):
        entries = [
            (p.stat().st_mtime_ns, p.stat().st_size, p)
            for p in self.directory.glob("*.bin")
        ]
        used = sum(size for _, size, _ in entries)
        for _, size, path in sorted(entries):
            if used + needed <= self.max_bytes:
                return
            stripe = path.stem[:3]
            if stripe == owned_stripe:
                # Our exclusive stripe already proves no other reader holds this entry.
                path.unlink(missing_ok=True)
                used -= size
                self.increment("evictions")
                continue
            with (self.directory / f"stripe-{stripe}.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                path.unlink(missing_ok=True)
                used -= size
                self.increment("evictions")
        if used + needed > self.max_bytes:
            raise RuntimeError(
                "Cache budget is occupied by active readers; reduce prefetch or increase cache bytes"
            )


class RangeReader(io.RawIOBase):
    """Seekable HTTP object: require exact 206 responses and meter actual payload bytes."""

    def __init__(self, url, size, *, headers=None, on_read=None, timeout=120):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.headers = headers or {}
        self.on_read = on_read or (lambda amount: None)
        self.timeout = timeout
        self.session = requests.Session()
        self._lock = threading.RLock()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        with self._lock:
            position = (
                offset
                if whence == 0
                else self.position + offset
                if whence == 1
                else self.size + offset
            )
            if position < 0 or whence not in (0, 1, 2):
                raise ValueError("Invalid seek")
            self.position = position
            return position

    def read(self, size=-1):
        with self._lock:
            start = self.position
            stop = self.size if size < 0 else min(start + size, self.size)
            if start >= stop:
                return b""
            for attempt in range(3):
                try:
                    with self.session.get(
                        self.url,
                        headers={**self.headers, "Range": f"bytes={start}-{stop - 1}"},
                        timeout=(15, self.timeout),
                        stream=True,
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise ValueError(
                                "Storage ignored byte range; refusing a whole-shard download"
                            )
                        if (
                            response.headers.get("Content-Range")
                            != f"bytes {start}-{stop - 1}/{self.size}"
                        ):
                            raise ValueError(
                                "Storage returned an unexpected byte range or object size"
                            )
                        data = bytearray()
                        for chunk in response.iter_content(1024 * 1024):
                            self.on_read(len(chunk))
                            data.extend(chunk)
                            if len(data) > stop - start:
                                raise ValueError(
                                    "Storage returned more bytes than requested"
                                )
                        if len(data) != stop - start:
                            raise requests.ConnectionError(
                                "Incomplete byte-range response"
                            )
                    self.position = stop
                    return bytes(data)
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 2:
                        raise
                    time.sleep(0.25 * (attempt + 1))

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[: len(value)] = value
        return len(value)

    def close(self):
        self.session.close()
        super().close()
