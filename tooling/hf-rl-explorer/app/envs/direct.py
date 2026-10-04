"""Rows read straight from a dataset's files, for when the dataset viewer can't (its conversion failed, or it's off):
JSON Lines, JSON (a list, or a dict of lists), parquet and CSV.

Parquet is read a row group at a time over HTTP ranges, so a big file costs only the groups a page needs. A JSON
Lines file is downloaded once (to the Hub cache) and indexed by line offsets, so any row is one seek. Search and
filters scan the first rows only (up to SCAN), and say so.
"""

from __future__ import annotations

import json
import posixpath
import re
import threading
import time
from array import array
from typing import Any

from .. import catalog, config

EXT = re.compile(r"\.(jsonl|ndjson|json|parquet|csv|tsv)(\.gz)?$", re.I)
SPLIT = re.compile(r"(^|[/_.\-])(train|training|test|testing|eval|evaluation|valid|validation|dev|val)([/_.\-]|$)", re.I)
MAX_FILE = 600 * 2**20     # a JSON Lines / JSON / CSV file this explorer will download
SCAN = 20_000              # rows a search or a filter looks through
_lock = threading.Lock()
_lines: dict[str, tuple[str, array]] = {}      # local path -> (path, line start offsets)


class DirectError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


HEAD = 512 * 2**10            # bytes of a JSON Lines file read first, so its first pages show before the whole file is in
_fetching: dict[str, dict[str, Any]] = {}     # path key -> {"state": running|done|error, "local", "error"}


def _head(spec: str, sha: str, path: str, token: str | None) -> tuple[list[dict[str, Any]], bool]:
    """The rows in the first HEAD bytes of a JSON Lines file (one range request), and whether that was all of it."""
    import httpx
    from urllib.parse import quote

    def fetch():
        r = httpx.get(f"https://huggingface.co/datasets/{spec}/resolve/{sha}/{quote(path)}", follow_redirects=True, timeout=httpx.Timeout(45, connect=10),
                      headers={"Range": f"bytes=0-{HEAD - 1}", **({"Authorization": f"Bearer {token}"} if token else {})})
        if r.status_code not in (200, 206):
            raise DirectError(f"couldn't read {path} (HTTP {r.status_code})")
        data = r.content
        whole = r.status_code == 200 or len(data) < HEAD
        lines = data.split(b"\n")
        if not whole:
            lines = lines[:-1]   # the last line was cut by the range
        out = []
        for ln in lines:
            if ln.strip():
                try:
                    v = json.loads(ln)
                except ValueError:
                    continue
                out.append(v if isinstance(v, dict) else {"value": v})
        return out, whole

    return catalog._cached(("direct-head", spec, sha, path), 3600, fetch)


def _background(spec: str, sha: str, path: str, size: int | None, token: str | None) -> dict[str, Any]:
    """The whole file, downloaded in the background once; its state now."""
    key = f"{spec}@{sha}/{path}"
    with _lock:
        st = _fetching.get(key)
        if st and (st["state"] != "error" or time.time() - st.get("at", 0) < 120):
            return st
        st = _fetching[key] = {"state": "running", "at": time.time()}

    def run():
        try:
            st["local"] = _local(spec, sha, path, size, token)
            _line_index(st["local"])
            st["state"] = "done"
        except Exception as e:  # noqa: BLE001 - reported to the page
            st.update(state="error", error=str(e)[:200], at=time.time())

    threading.Thread(target=run, daemon=True, name=f"direct-{path}").start()
    return st


def data_files(spec: str, meta: dict[str, Any], token: str | None) -> dict[str, dict[str, list[str]]]:
    """{config: {split: [paths]}} from the card's `configs`, or the data files the repository holds."""
    from fnmatch import fnmatch

    def listing():
        out = []
        for n, e in enumerate(catalog._api(token).list_repo_tree(spec, repo_type="dataset", revision=meta["sha"], recursive=True)):
            if hasattr(e, "size") and EXT.search(e.path) and not posixpath.basename(e.path).startswith("."):
                out.append((e.path, e.size))
            if n > 50_000:
                break
        return out

    files = catalog._cached(("data-files", spec, meta["sha"]), 3600, listing)
    names = [p for p, _ in files]
    configs: dict[str, dict[str, list[str]]] = {}

    def card_data():
        info = catalog._api(token).dataset_info(spec, revision=meta["sha"], expand=["cardData"])
        return info.card_data.to_dict() if info.card_data else {}

    card = catalog._cached(("card", spec, meta["sha"]), 3600, card_data)
    for c in (card.get("configs") or []) if isinstance(card, dict) else []:
        df = c.get("data_files")
        name = c.get("config_name") or "default"
        specs = [{"split": "train", "path": df}] if isinstance(df, (str, list)) and not (isinstance(df, list) and df and isinstance(df[0], dict)) else (df or [])
        for d in specs:
            pats = d["path"] if isinstance(d.get("path"), list) else [d.get("path")]
            hits = sorted(p for p in names if any(pat and fnmatch(p, pat) for pat in pats))
            if hits:
                configs.setdefault(name, {})[d.get("split") or "train"] = hits
    if not configs and names:
        by: dict[str, list[str]] = {}
        for p in sorted(names):
            m = SPLIT.search(p)
            split = {"training": "train", "testing": "test", "evaluation": "eval", "valid": "validation", "val": "validation"}.get(
                m.group(2).lower(), m.group(2).lower()) if m else "train"
            by.setdefault(split, []).append(p)
        configs["default"] = by
    return configs


def _local(spec: str, sha: str, path: str, size: int | None, token: str | None) -> str:
    from huggingface_hub import hf_hub_download

    if size is not None and size > MAX_FILE:
        raise DirectError(f"{path} is {size / 2**20:.0f} MB: too large to read here")
    return hf_hub_download(spec, path, repo_type="dataset", revision=sha, token=token or False,
                           cache_dir=str(config.CACHE_DIR / "direct"))


def _line_index(local: str) -> array:
    with _lock:
        hit = _lines.get(local)
    if hit:
        return hit[1]
    offs = array("Q")
    with open(local, "rb") as f:
        pos = 0
        for line in f:
            if line.strip():
                offs.append(pos)
            pos += len(line)
    with _lock:
        _lines[local] = (local, offs)
        while len(_lines) > 12:
            _lines.pop(next(iter(_lines)))
    return offs


class Source:
    """One split's rows, across its files, in order."""

    def __init__(self, spec: str, meta: dict[str, Any], paths: list[str], token: str | None):
        self.spec, self.meta, self.paths, self.token = spec, meta, paths[:200], token
        self._counts: list[int] | None = None

    def _size(self, path: str) -> int | None:
        files = catalog._cached(("data-files", self.spec, self.meta["sha"]), 3600, lambda: [])
        return next((s for p, s in files if p == path), None)

    def _parquet(self, path: str):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=self.token or False)
        return pq.ParquetFile(fs.open(f"datasets/{self.spec}@{self.meta['sha']}/{path}", "rb", block_size=2**20))

    def count(self, path: str) -> int:
        def fetch():
            low = path.lower()
            if low.endswith(".parquet"):
                return self._parquet(path).metadata.num_rows
            if low.endswith((".jsonl", ".ndjson")):
                st = _background(self.spec, self.meta["sha"], path, self._size(path), self.token)
                if st["state"] != "done":
                    rows, whole = _head(self.spec, self.meta["sha"], path, self.token)
                    return len(rows) if whole else None
                return len(_line_index(st["local"]))
            return len(self._whole(path))

        n = catalog._cached(("direct-count", self.spec, self.meta["sha"], path), 3600, fetch)
        if n is None:   # not known until the file is in: don't keep that answer
            catalog._memo.pop(("direct-count", self.spec, self.meta["sha"], path), None)
        return n

    def _whole(self, path: str) -> list[dict[str, Any]]:
        def fetch():
            local = _local(self.spec, self.meta["sha"], path, self._size(path), self.token)
            low = path.lower()
            if low.endswith((".csv", ".tsv")):
                import csv

                with open(local, newline="", encoding="utf-8", errors="replace") as f:
                    return list(csv.DictReader(f, delimiter="\t" if low.endswith(".tsv") else ","))
            with open(local, encoding="utf-8", errors="replace") as f:
                doc = json.load(f)
            if isinstance(doc, dict):   # {"data": [...]} or a dict of columns
                lists = [v for v in doc.values() if isinstance(v, list)]
                if len(doc) == 1 and lists:
                    doc = lists[0]
                elif lists and all(isinstance(v, list) and len(v) == len(lists[0]) for v in doc.values()):
                    doc = [dict(zip(doc, vals)) for vals in zip(*doc.values())]
                else:
                    doc = [doc]
            return [r if isinstance(r, dict) else {"value": r} for r in doc] if isinstance(doc, list) else []

        return catalog._cached(("direct-whole", self.spec, self.meta["sha"], path), 1800, fetch, big=True)

    def total(self) -> int | None:
        """Rows in the split; None while a file is still being read (the first rows are served meanwhile)."""
        if self._counts is None:
            self._counts = [self.count(p) for p in self.paths]
        return None if any(n is None for n in self._counts) else sum(self._counts)

    def read(self, offset: int, length: int) -> list[tuple[int, dict[str, Any]]]:
        self.total()
        if self._counts and self._counts[0] is None:   # the first file is still coming: its head serves the first pages
            rows, _ = _head(self.spec, self.meta["sha"], self.paths[0], self.token)
            if offset + length <= len(rows) or offset < len(rows):
                return [(offset + k, r) for k, r in enumerate(rows[offset:offset + length])]
            raise DirectError("this file is still being read; these rows show in a minute", 409)
        out: list[tuple[int, dict[str, Any]]] = []
        base = 0
        for path, n in zip(self.paths, self._counts or []):
            if n is None:
                raise DirectError("this file is still being read; these rows show in a minute", 409)
            if offset + length <= base:
                break
            if offset < base + n:
                lo, hi = max(0, offset - base), min(n, offset + length - base)
                out += [(base + lo + k, r) for k, r in enumerate(self._slice(path, lo, hi))]
            base += n
        return out

    def _slice(self, path: str, lo: int, hi: int) -> list[dict[str, Any]]:
        low = path.lower()
        if low.endswith(".parquet"):
            pf = self._parquet(path)
            rows, start = [], 0
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                if start + n > lo and start < hi:
                    tbl = pf.read_row_group(g).to_pylist()
                    rows += tbl[max(0, lo - start):hi - start]
                start += n
                if start >= hi:
                    break
            return [_plain(r) for r in rows]
        if low.endswith((".jsonl", ".ndjson")):
            st = _background(self.spec, self.meta["sha"], path, self._size(path), self.token)
            if st["state"] != "done":
                raise DirectError("this file is still being read; these rows show in a minute", 409)
            local = st["local"]
            offs = _line_index(local)
            out = []
            with open(local, "rb") as f:
                for k in range(lo, min(hi, len(offs))):
                    f.seek(offs[k])
                    try:
                        r = json.loads(f.readline())
                    except ValueError:
                        r = {"_unreadable_line": k}
                    out.append(r if isinstance(r, dict) else {"value": r})
            return out
        return self._whole(path)[lo:hi]


def _plain(r: dict[str, Any]) -> dict[str, Any]:
    """Parquet values the page can take: bytes as base64 (a packed task), the rest as they are."""
    import base64

    out = {}
    for k, v in r.items():
        out[k] = base64.b64encode(v).decode() if isinstance(v, (bytes, bytearray)) else v
    return out


def scan(src: Source, match, limit: int = SCAN) -> tuple[list[tuple[int, dict[str, Any]]], bool]:
    """Rows that match, from the first `limit` rows; and whether it stopped early."""
    hits = []
    total = src.total()
    if total is None:
        raise DirectError("search works once the whole file is read; try again in a minute", 409)
    for start in range(0, min(total, limit), 500):
        for i, r in src.read(start, 500):
            if match(r):
                hits.append((i, r))
    return hits, total > limit
