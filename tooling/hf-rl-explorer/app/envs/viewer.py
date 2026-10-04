"""The Hub's dataset viewer API (datasets-server), which reads rows of any format the Hub can convert (parquet, JSON
Lines, JSON, CSV, ...) a page at a time: splits, rows, filters, per-column statistics and full-text search. Rows-based
environments are browsed through it, so nothing has to be downloaded or indexed first.

Private datasets are read with the visitor's token (the viewer serves them to accounts that may read them); public
ones without a token, so their answers are cached for everyone.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import httpx

BASE = "https://datasets-server.huggingface.co"
_client = httpx.Client(timeout=httpx.Timeout(30, connect=8), headers={"User-Agent": "hf-rl-explorer"},
                       limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
_memo: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


class ViewerError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _get(path: str, params: dict[str, Any], token: str | None = None, ttl: float = 600, timeout: float = 30) -> dict[str, Any]:
    # by account (a hash of the whole token): two visitors whose tokens share an ending never share a private answer
    key = (path, tuple(sorted((k, str(v)) for k, v in params.items())), token and hashlib.sha256(token.encode()).hexdigest()[:24])
    now = time.time()
    with _lock:
        hit = _memo.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    r = None
    tries = 3 if timeout >= 20 else 2   # the viewer has brief hiccups (a 502, a slow answer); search and filter fall back sooner
    for attempt in range(tries):
        last = attempt == tries - 1
        try:
            r = _client.get(BASE + path, params=params, headers={"Authorization": f"Bearer {token}"} if token else {},
                            timeout=httpx.Timeout(timeout, connect=8))
        except httpx.HTTPError as e:
            if last or timeout < 20:
                raise ViewerError(f"couldn't reach the dataset viewer: {type(e).__name__}") from None
            continue
        if (r.status_code < 500 and r.status_code != 429) or last:   # rate limited: wait as asked (capped), then again
            break
        ra = r.headers.get("Retry-After", "")
        time.sleep(min(4.0, float(ra)) if ra.replace(".", "", 1).isdigit() else 0.6 * (attempt + 1))
    try:
        doc = r.json()
    except ValueError:
        raise ViewerError(f"the dataset viewer answered HTTP {r.status_code}") from None
    if r.status_code != 200:
        msg = doc.get("error") if isinstance(doc, dict) else None
        raise ViewerError(str(msg or f"the dataset viewer answered HTTP {r.status_code}")[:300], 404 if r.status_code in (401, 404) else 502)
    with _lock:
        _memo[key] = (now, doc)
        if len(_memo) > 4000:
            for k in sorted(_memo, key=lambda k: _memo[k][0])[:1000]:
                _memo.pop(k, None)
    return doc


def valid(spec: str, token: str | None = None) -> dict[str, bool]:
    try:
        return _get("/is-valid", {"dataset": spec}, token, ttl=1800)
    except ViewerError:
        return {}


def splits(spec: str, token: str | None = None) -> list[dict[str, str]]:
    doc = _get("/splits", {"dataset": spec}, token, ttl=1800)
    return [{"config": s["config"], "split": s["split"]} for s in doc.get("splits") or []]


def rows(spec: str, config: str, split: str, offset: int, length: int, token: str | None = None) -> dict[str, Any]:
    doc = _get("/rows", {"dataset": spec, "config": config, "split": split, "offset": max(0, offset), "length": max(1, min(100, length))}, token)
    return {"rows": [(r["row_idx"], r["row"], r.get("truncated_cells") or []) for r in doc.get("rows") or []],
            "total": doc.get("num_rows_total"), "features": doc.get("features") or [], "partial": bool(doc.get("partial"))}


def search(spec: str, config: str, split: str, query: str, offset: int, length: int, token: str | None = None) -> dict[str, Any]:
    doc = _get("/search", {"dataset": spec, "config": config, "split": split, "query": query[:200], "offset": max(0, offset),
                           "length": max(1, min(100, length))}, token, ttl=300, timeout=12)
    return {"rows": [(r["row_idx"], r["row"], r.get("truncated_cells") or []) for r in doc.get("rows") or []],
            "total": doc.get("num_rows_total"), "features": doc.get("features") or []}


def filter_rows(spec: str, config: str, split: str, where: str, offset: int, length: int, token: str | None = None) -> dict[str, Any]:
    doc = _get("/filter", {"dataset": spec, "config": config, "split": split, "where": where, "offset": max(0, offset),
                           "length": max(1, min(100, length))}, token, ttl=300, timeout=12)
    return {"rows": [(r["row_idx"], r["row"], r.get("truncated_cells") or []) for r in doc.get("rows") or []],
            "total": doc.get("num_rows_total"), "features": doc.get("features") or []}


def statistics(spec: str, config: str, split: str, token: str | None = None) -> list[dict[str, Any]]:
    try:
        return _get("/statistics", {"dataset": spec, "config": config, "split": split}, token, ttl=3600).get("statistics") or []
    except ViewerError:
        return []
