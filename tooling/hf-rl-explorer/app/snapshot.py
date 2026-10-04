"""Immutable SQLite snapshots of the catalog: built offline (app/indexer.py, an HF scheduled Job), read here.

A snapshot is one SQLite file, `snapshots/catalog-<UTC ts>.db` in the store (the bucket), published by writing the
pointer `snapshots/LATEST.v<SCHEMA>.json` last:

    {"db": "snapshots/catalog-20261003T230000Z.db", "sha256": "...", "size": 123, "built_at": "...", "counts": {...},
     "schema": 1}

Every app process looks at the pointer on startup and every CHECK_EVERY seconds. A new one is copied from the store to
local disk (SQLite never opens a file on the bucket mount), its sha256 checked, opened read-only and immutable, checked
again (schema, row counts, quick_check), then swapped in: queries already running finish on the old one, which is
closed when the last of them is done. Bumping SCHEMA means a new pointer name, so a running app of an older version
keeps reading the snapshots made for it.

With no snapshot (local dev, the first boot, a corrupt file) the same database is built here, from the catalog's own
listing (`catalog.environments()`) and the indexes on disk: nothing goes dark, and one query path serves both.

What is in it is only what the explorer already shows publicly: public environments' cards, and their indexed tasks'
titles, briefs and facets (no metadata values, no file texts, no answers). What admins change between runs (hidden
environments, pins, collections) is applied when it is queried (app/search_api.py), not baked in.

Tables: envs (one row per environment), envs_fts (trigram FTS5: substring search, as the page has always done),
tasks (one row per indexed task), tasks_fts (FTS5, token prefixes), meta.
"""

from __future__ import annotations

import calendar
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import config

log = logging.getLogger("rlx")

SCHEMA = 1
SNAP_DIR = "snapshots"
POINTER = f"{SNAP_DIR}/LATEST.v{SCHEMA}.json"
DB_NAME = re.compile(r"^snapshots/catalog-[0-9TZ]+(?:-\d+)?\.db$")
CHECK_EVERY = float(os.environ.get("RLX_SNAPSHOT_CHECK", "60"))
FIRST_WAIT = float(os.environ.get("RLX_SNAPSHOT_FIRST_WAIT", "8"))   # a request waits this long for the first check
POOL = 8                                                            # idle read connections kept per snapshot
MIMO = "XiaomiMiMo/MiMo-V2.6-RL-oss"


def local_dir() -> Path:
    """Where snapshots are opened from: local disk, never the bucket mount."""
    return Path(os.environ.get("RLX_SNAPSHOT_LOCAL") or config.CACHE_DIR / "snapshots")


class SnapshotError(Exception):
    """A snapshot that can't be used: missing, corrupt, of another schema, or not what its pointer says."""


# ── stores: a folder (local dev, or a bucket mounted at /data), or a bucket through the Hub's API ─────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tmp_name(p: Path) -> Path:
    return p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")


class LocalStore:
    """A folder as the store: `.local-data` locally, or the bucket mounted at /data (a Space's or a Job's volume).
    Writes go to a temp name of this process's own and are renamed into place, so a reader sees the old file or the
    new one (on a bucket mount, a rename is one batch of the bucket API: add the new path, delete the old)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def __str__(self) -> str:
        return str(self.root)

    def _p(self, rel: str) -> Path:
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"not a store path: {rel!r}")
        return self.root / rel

    def read_bytes(self, rel: str) -> bytes | None:
        try:
            return self._p(rel).read_bytes()
        except FileNotFoundError:
            return None

    def fetch(self, rel: str, dst: Path) -> str:
        """Copy a file out of the store to `dst`; its sha256."""
        h = hashlib.sha256()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(self._p(rel), "rb") as src, open(dst, "wb") as out:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        return h.hexdigest()

    def put_file(self, src: Path, rel: str) -> None:
        dst = self._p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_name(dst)
        try:
            with open(src, "rb") as fh, open(tmp, "wb") as out:
                shutil.copyfileobj(fh, out, 1 << 20)
                out.flush()
                os.fsync(out.fileno())
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)

    def put_bytes(self, rel: str, data: bytes) -> None:
        dst = self._p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_name(dst)
        try:
            with open(tmp, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)

    def size(self, rel: str) -> int | None:
        try:
            return self._p(rel).stat().st_size
        except FileNotFoundError:
            return None

    def digest(self, rel: str) -> str | None:
        """The stored file's sha256, read back (a folder can be re-read cheaply; a bucket can't)."""
        try:
            return sha256_file(self._p(rel))
        except FileNotFoundError:
            return None

    def list(self, prefix: str) -> list[str]:
        d = self._p(prefix)
        if not d.is_dir():
            return []
        return sorted(f"{prefix.rstrip('/')}/{p.name}" for p in d.iterdir() if p.is_file())

    def age(self, rel: str) -> float:
        try:
            return time.time() - self._p(rel).stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def delete(self, rels: list[str]) -> None:
        for rel in rels:
            self._p(rel).unlink(missing_ok=True)


class BucketStore:
    """A bucket through the Hub's bucket API (`hf://buckets/<org>/<name>[/<prefix>]`), for a writer or reader without
    the mount. Each put is one batch call, so a file appears whole; the pointer is its own, later call."""

    def __init__(self, bucket_id: str, prefix: str = "", api: Any = None, token: str | None = None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.bucket, self.prefix, self.api = bucket_id, prefix.strip("/"), api

    def __str__(self) -> str:
        return f"hf://buckets/{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")

    def _p(self, rel: str) -> str:
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"not a store path: {rel!r}")
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def read_bytes(self, rel: str) -> bytes | None:
        tmp = local_dir() / f".read.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)
        try:
            self.api.download_bucket_files(self.bucket, [(self._p(rel), str(tmp))], raise_on_missing_files=False)
            return tmp.read_bytes() if tmp.exists() else None
        finally:
            tmp.unlink(missing_ok=True)

    def fetch(self, rel: str, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.api.download_bucket_files(self.bucket, [(self._p(rel), str(dst))], raise_on_missing_files=True)
        return sha256_file(dst)

    def put_file(self, src: Path, rel: str) -> None:
        self.api.batch_bucket_files(self.bucket, add=[(str(src), self._p(rel))])

    def put_bytes(self, rel: str, data: bytes) -> None:
        self.api.batch_bucket_files(self.bucket, add=[(data, self._p(rel))])

    def size(self, rel: str) -> int | None:
        for f in self.api.get_bucket_paths_info(self.bucket, [self._p(rel)]):
            return f.size
        return None

    def digest(self, rel: str) -> str | None:
        return None   # not re-downloaded: the size is checked instead

    def list(self, prefix: str) -> list[str]:
        out = []
        for f in self.api.list_bucket_tree(self.bucket, prefix=self._p(prefix), recursive=True):
            if getattr(f, "type", "file") == "file":
                out.append(f.path[len(self.prefix) + 1:] if self.prefix else f.path)
        return sorted(out)

    def age(self, rel: str) -> float:
        return 0.0

    def delete(self, rels: list[str]) -> None:
        if rels:
            self.api.batch_bucket_files(self.bucket, delete=[self._p(r) for r in rels])


def open_store(spec: str | Path | None = None):
    """`hf://buckets/<org>/<name>[/<prefix>]` is a bucket through the API; anything else a folder. The default is
    RLX_SNAPSHOT_STORE, else STORAGE_DIR (the bucket mount on a Space)."""
    spec = str(spec or os.environ.get("RLX_SNAPSHOT_STORE") or config.STORAGE_DIR)
    m = re.match(r"^hf://buckets/([^/]+/[^/]+)/?(.*)$", spec)
    if m:
        return BucketStore(m.group(1), m.group(2))
    return LocalStore(spec)


def read_pointer(store) -> dict[str, Any] | None:
    """The store's pointer for this schema, checked for shape; None when there is none yet."""
    raw = store.read_bytes(POINTER)
    if raw is None:
        return None
    try:
        ptr = json.loads(raw)
    except ValueError as e:
        raise SnapshotError(f"pointer is not JSON: {e}") from e
    if not isinstance(ptr, dict) or ptr.get("schema") != SCHEMA or not DB_NAME.match(str(ptr.get("db") or "")) \
            or not re.fullmatch(r"[0-9a-f]{64}", str(ptr.get("sha256") or "")):
        raise SnapshotError(f"pointer is malformed: {str(raw[:200])!r}")
    return ptr


# ── the database ─────────────────────────────────────────────────────────────
DDL = """
CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID;
CREATE TABLE envs(
  rowid INTEGER PRIMARY KEY,
  ord INTEGER NOT NULL,              -- its place in the listing: the tiebreak the page has always had
  id TEXT NOT NULL, key TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
  framework TEXT,                    -- datasets: by Hub tags (harbor when its index found tasks); Spaces: openenv/ors/space
  openenv INTEGER NOT NULL DEFAULT 0,
  heading TEXT, brief TEXT,
  downloads INTEGER NOT NULL DEFAULT 0, likes INTEGER NOT NULL DEFAULT 0, trending NUMERIC NOT NULL DEFAULT 0,
  created TEXT, updated TEXT, created_ms INTEGER NOT NULL DEFAULT 0, updated_ms INTEGER NOT NULL DEFAULT 0,
  stage TEXT, mcp INTEGER NOT NULL DEFAULT 0, openenv_version TEXT, manifest TEXT, hardware TEXT, sdk TEXT,
  badges TEXT NOT NULL DEFAULT '[]', tags TEXT NOT NULL DEFAULT '[]', tags_text TEXT NOT NULL DEFAULT '',
  tasks INTEGER,                     -- indexed task count; NULL when not indexed
  indexed TEXT,                      -- {"tasks", "graded", "image"} of its index, JSON
  sha TEXT,                          -- the revision its index was built from
  size_f TEXT, stage_f TEXT, mcp_f TEXT, oe_f TEXT,   -- facet values that don't depend on admin settings
  blob TEXT NOT NULL                 -- id, heading, brief and tags, lowercased: what every search word must be in
);
CREATE VIRTUAL TABLE envs_fts USING fts5(id, heading, brief, tags_text, content='envs', content_rowid='rowid',
                                         tokenize='trigram');
CREATE TABLE tasks(
  rowid INTEGER PRIMARY KEY,
  env TEXT NOT NULL, ref TEXT NOT NULL, title TEXT, brief TEXT, category TEXT, difficulty TEXT, grp TEXT,
  run TEXT, grading TEXT, judge INTEGER NOT NULL DEFAULT 0, tags TEXT NOT NULL DEFAULT '[]', tags_text TEXT NOT NULL DEFAULT '',
  extras TEXT
);
CREATE VIRTUAL TABLE tasks_fts USING fts5(title, brief, ref, category, tags_text, content='tasks', content_rowid='rowid',
                                          tokenize='unicode61 remove_diacritics 2');
"""

INDEXES = """
CREATE INDEX envs_kind ON envs(kind, framework);
CREATE INDEX envs_trending ON envs(trending DESC, likes DESC, downloads DESC);
CREATE INDEX envs_downloads ON envs(downloads DESC);
CREATE INDEX envs_likes ON envs(likes DESC);
CREATE INDEX envs_tasks ON envs(tasks DESC);
CREATE INDEX envs_updated ON envs(updated_ms DESC);
CREATE INDEX envs_created ON envs(created_ms DESC);
CREATE INDEX envs_id ON envs(id);
CREATE INDEX envs_size ON envs(size_f) WHERE size_f IS NOT NULL;
CREATE INDEX envs_stage ON envs(stage_f) WHERE stage_f IS NOT NULL;
CREATE INDEX envs_mcp ON envs(mcp_f) WHERE mcp_f IS NOT NULL;
CREATE INDEX envs_oe ON envs(oe_f) WHERE oe_f IS NOT NULL;
CREATE INDEX tasks_env ON tasks(env, category);
"""

ENV_COLS = ("ord", "id", "key", "kind", "framework", "openenv", "heading", "brief", "downloads", "likes", "trending",
            "created", "updated", "created_ms", "updated_ms", "stage", "mcp", "openenv_version", "manifest", "hardware",
            "sdk", "badges", "tags", "tags_text", "tasks", "indexed", "sha", "size_f", "stage_f", "mcp_f", "oe_f", "blob")
TASK_COLS = ("env", "ref", "title", "brief", "category", "difficulty", "grp", "run", "grading", "judge", "tags",
             "tags_text", "extras")


def iso_ms(iso: Any) -> int:
    """An ISO time as the page's Date.parse reads it: milliseconds since the epoch, 0 when there is none."""
    if not iso:
        return 0
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return calendar.timegm(dt.timetuple()) * 1000 + dt.microsecond // 1000


def size_bucket(n: int | None) -> str:
    """The "Tasks" facet's value, as the page names it."""
    return "Not indexed yet" if n is None else "Under 100" if n < 100 else "100 to 1,000" if n < 1000 else "1,000 or more"


def stage_value(stage: Any) -> str:
    return "Running" if stage == "RUNNING" else "Asleep or stopped" if stage else "Unknown"


def blob_of(r: dict[str, Any]) -> str:
    """What a search word must be part of, exactly as the page built it: `${id} ${heading || ""} ${brief} ${tags}`."""
    brief = r.get("brief")
    return f"{r['id']} {r.get('heading') or ''} {'null' if brief is None else brief} {' '.join(r.get('tags') or [])}".lower()


def env_record(r: dict[str, Any], ord_: int) -> tuple:
    """One environment row for the database, from a listing row (catalog.environments() or the indexer's own)."""
    kind = r.get("kind") or "dataset"
    space = kind == "space"
    ix = r.get("indexed") or None
    tasks = ix.get("tasks") if isinstance(ix, dict) else None
    fw = r.get("framework")
    tags = [str(t) for t in (r.get("tags") or [])]
    badges = [str(b) for b in (r.get("badges") or [])]
    indexed = json.dumps({k: ix.get(k) for k in ("tasks", "graded", "image") if ix.get(k) is not None}, separators=(",", ":")) if ix else None
    rec = {
        "ord": ord_, "id": r["id"], "key": r.get("key") or r["id"], "kind": kind, "framework": fw,
        "openenv": int(bool(r.get("openenv"))), "heading": r.get("heading"), "brief": r.get("brief"),
        "downloads": int(r.get("downloads") or 0), "likes": int(r.get("likes") or 0), "trending": r.get("trending") or 0,
        "created": r.get("created"), "updated": r.get("updated"), "created_ms": iso_ms(r.get("created")), "updated_ms": iso_ms(r.get("updated")),
        "stage": r.get("stage"), "mcp": int(bool(r.get("mcp"))), "openenv_version": r.get("openenv_version"),
        "manifest": r.get("manifest"), "hardware": r.get("hardware"), "sdk": r.get("sdk"),
        "badges": json.dumps(badges, ensure_ascii=False), "tags": json.dumps(tags, ensure_ascii=False), "tags_text": " ".join(tags),
        "tasks": tasks, "indexed": indexed, "sha": (ix or {}).get("sha") if ix else None,
        "size_f": size_bucket(tasks) if not space and (fw or "harbor") == "harbor" else None,
        "stage_f": stage_value(r.get("stage")) if space else None,
        "mcp_f": ("Has MCP tools" if r.get("mcp") else "No MCP tag") if space else None,
        "oe_f": r.get("openenv_version") if space and r.get("openenv_version") else None,
        "blob": blob_of({**r, "tags": tags}),
    }
    return tuple(rec[c] for c in ENV_COLS)


def _clip(text: Any, n: int) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def task_record(env: str, t: dict[str, Any]) -> tuple:
    """One task row, from a row of a dataset's index: its title, brief and facets; nothing that holds an answer."""
    ver, envd = t.get("verifier") or {}, t.get("env") or {}
    tags = [str(x)[:60] for x in (t.get("tags") or [])][:12]
    extras = {k: v for k, v in {"steps": t.get("steps"), "gpus": envd.get("gpus") or None, "network": envd.get("network"),
                                "image": bool(envd.get("image")) or None, "pending": t.get("pending") or None,
                                "invalid": t.get("invalid") or None}.items() if v}
    return (env, str(t.get("path") or ""), _clip(t.get("title") or t.get("name"), 200), _clip(t.get("brief"), 200),
            t.get("category"), t.get("difficulty"), t.get("group"), t.get("run"), ver.get("kind"), int(bool(ver.get("judge"))),
            json.dumps(tags, ensure_ascii=False), " ".join(tags), json.dumps(extras, separators=(",", ":")) if extras else None)


def iter_index_tasks(specs: Iterable[str], *, mimo: bool = True, seen: dict[str, Any] | None = None) -> Iterator[tuple]:
    """Every task of these datasets' indexes on disk, one dataset in memory at a time. An index of another format, of
    another dataset, or of a private or gated one (built for a signed-in visitor) is left out. `seen` collects what
    was read, by dataset: its task count and whether the index is a partial one."""
    import gzip

    from . import catalog

    for spec in specs:
        if spec == MIMO:
            continue
        p = catalog._index_path(spec)
        try:
            idx = json.loads(gzip.decompress(p.read_bytes()))
        except (OSError, ValueError, EOFError):
            continue
        if not isinstance(idx, dict) or idx.get("version") != catalog.INDEX_VERSION or idx.get("spec") != spec \
                or (idx.get("info") or {}).get("restricted"):
            continue
        if seen is not None:
            seen[spec] = {"tasks": len(idx.get("tasks") or []), "partial": bool(idx.get("partial"))}
        for t in idx.get("tasks") or []:
            if isinstance(t, dict) and t.get("path") is not None:
                yield task_record(spec, t)
        del idx
    if mimo:   # the MiMo release's own rows (app/mimo/data, shipped in the image)
        try:
            from .mimo import catalog as mcat

            for e in mcat.index()["envs"]:
                yield (MIMO, str(e["id"]), _clip(e.get("t"), 200), _clip(e.get("s"), 200), e.get("d"), None, None,
                       None, None, 0, "[]", "", None)
        except Exception:  # noqa: BLE001 - its data missing leaves the rest
            log.warning("snapshot.mimo_tasks_skipped")


def build_db(path: Path, envs: Iterable[dict[str, Any]], tasks: Iterable[tuple] = (), meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a snapshot database at `path` (a new file): envs from listing rows, tasks streamed in (never all in memory
    at once), FTS, indexes, statistics, then VACUUM and a quick check. Returns its counts. Raises on anything wrong;
    `path` is then left for the caller to delete."""
    path = Path(path)
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA page_size=8192")
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")
        conn.executescript(DDL)
        seen: set[str] = set()

        def env_rows() -> Iterator[tuple]:
            for i, r in enumerate(envs):
                if r.get("private") or r.get("gated"):   # only what everyone may see
                    continue
                key = r.get("key") or r["id"]
                if key in seen:
                    continue
                seen.add(key)
                yield env_record(r, i)

        conn.executemany(f"INSERT INTO envs({','.join(ENV_COLS)}) VALUES ({','.join('?' * len(ENV_COLS))})", env_rows())
        public = {r[0] for r in conn.execute("SELECT id FROM envs WHERE kind = 'dataset'")}
        conn.executemany(f"INSERT INTO tasks({','.join(TASK_COLS)}) VALUES ({','.join('?' * len(TASK_COLS))})",
                         (t for t in tasks if t[0] in public))
        conn.execute("INSERT INTO envs_fts(envs_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO tasks_fts(tasks_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO tasks_fts(tasks_fts) VALUES('optimize')")
        conn.executescript(INDEXES)
        counts = counts_of(conn)
        built_at = datetime.now(UTC).isoformat(timespec="seconds")
        rows = {"schema": SCHEMA, "built_at": built_at, "counts": counts, **(meta or {})}
        conn.executemany("INSERT INTO meta(k, v) VALUES (?, ?)", [(k, json.dumps(v)) for k, v in rows.items()])
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
        conn.execute("VACUUM")
        ok = conn.execute("PRAGMA quick_check").fetchone()[0]
        if ok != "ok":
            raise SnapshotError(f"quick_check: {ok}")
        return {**counts, "built_at": built_at}
    finally:
        conn.close()


def counts_of(conn: sqlite3.Connection) -> dict[str, Any]:
    one = lambda sql: conn.execute(sql).fetchone()[0]
    return {"envs": one("SELECT COUNT(*) FROM envs"), "datasets": one("SELECT COUNT(*) FROM envs WHERE kind = 'dataset'"),
            "spaces": one("SELECT COUNT(*) FROM envs WHERE kind = 'space'"), "tasks": one("SELECT COUNT(*) FROM tasks"),
            "indexed": one("SELECT COUNT(*) FROM envs WHERE tasks > 0"),
            "task_envs": one("SELECT COUNT(DISTINCT env) FROM tasks")}


# ── a snapshot, open ─────────────────────────────────────────────────────────
class Snapshot:
    """One read-only database file and a small pool of connections to it. Retired when another replaces it: its
    connections close when the last query using it is done (and a file built here is deleted then)."""

    def __init__(self, path: Path, *, name: str, source: str, built_at: str | None, counts: dict[str, Any] | None = None,
                 owned: bool = False):
        self.path, self.name, self.source, self.built_at = Path(path), name, source, built_at
        self.counts = counts or {}
        self.owned = owned
        self.made = time.time()
        self._pool: list[sqlite3.Connection] = []
        self._users = 0
        self._retired = self._closed = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<Snapshot {self.name} {self.source}>"

    def _open(self) -> sqlite3.Connection:
        c = sqlite3.connect(f"file:{quote(str(self.path))}?mode=ro&immutable=1", uri=True, check_same_thread=False,
                            isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA query_only=1")
        c.execute("PRAGMA mmap_size=268435456")
        c.execute("PRAGMA cache_size=-16000")
        return c

    def _enter(self) -> None:
        with self._lock:
            if self._closed:
                raise SnapshotError("snapshot closed")
            self._users += 1

    def _leave(self, conn: sqlite3.Connection | None) -> None:
        with self._lock:
            self._users -= 1
            if conn is not None and not self._retired and len(self._pool) < POOL:
                self._pool.append(conn)
                conn = None
            done = self._retired and self._users == 0 and not self._closed
            if done:
                self._closed = True
        if conn is not None:
            conn.close()
        if done:
            self._close()

    def _checkout(self) -> sqlite3.Connection:
        """A connection from the pool, or a new one. Raises SnapshotError when the file can't be opened (gone)."""
        with self._lock:
            conn = self._pool.pop() if self._pool else None
        if conn is None:
            try:
                conn = self._open()
            except sqlite3.Error as e:
                raise SnapshotError(f"cannot open {self.path.name}: {e}") from e
        return conn

    def _checkin(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            if not self._retired and not self._closed and len(self._pool) < POOL:
                self._pool.append(conn)
                return
        conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A connection for a few queries (callers holding the snapshot via `use()` are already counted)."""
        conn = self._checkout()
        try:
            yield conn
        finally:
            self._checkin(conn)

    def retire(self) -> None:
        with self._lock:
            self._retired = True
            done = self._users == 0 and not self._closed
            if done:
                self._closed = True
        if done:
            self._close()

    def _close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, []
        for c in pool:
            try:
                c.close()
            except sqlite3.Error:
                pass
        if self.owned:
            self.path.unlink(missing_ok=True)

    def check(self, expect: dict[str, Any] | None = None, *, full: bool = True) -> None:
        """Open it and make sure it is what it says: this schema, the counts its pointer gives, and (full) SQLite's own
        quick_check. Raises SnapshotError."""
        try:
            conn = self._open()
        except sqlite3.Error as e:
            raise SnapshotError(f"cannot open: {e}") from e
        try:
            meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
            if meta.get("schema") != SCHEMA:
                raise SnapshotError(f"schema {meta.get('schema')!r}, this app reads {SCHEMA}")
            counts = counts_of(conn)
            for k, v in (expect or {}).items():
                if k in counts and counts[k] != v:
                    raise SnapshotError(f"{k}: {counts[k]} rows, the pointer says {v}")
            if full:
                ok = conn.execute("PRAGMA quick_check").fetchone()[0]
                if ok != "ok":
                    raise SnapshotError(f"quick_check: {ok}")
            self.counts = counts
            self.built_at = self.built_at or meta.get("built_at")
        except sqlite3.Error as e:
            raise SnapshotError(f"unreadable: {e}") from e
        finally:
            conn.close()


# ── which snapshot is in use ─────────────────────────────────────────────────
_lock = threading.Lock()
_cur: Snapshot | None = None        # from the store
_fb: Snapshot | None = None         # built here, when there is none
_fb_state: dict[str, Any] = {"sig": None, "checked": 0.0, "building": False}
_fb_lock = threading.Lock()         # _fb_state
_fb_first = threading.Lock()        # the first fallback build
_fb_seq = itertools.count(1)
_status: dict[str, Any] = {"checked_at": 0.0, "error": None, "rejected": {}}
_first = threading.Event()
_watcher: threading.Thread | None = None


def watching() -> bool:
    """A background thread looks at the pointer, except under tests (RLX_TEST_TMP) or with RLX_SNAPSHOT_WATCH=0."""
    return os.environ.get("RLX_SNAPSHOT_WATCH", "0" if os.environ.get("RLX_TEST_TMP") else "1") == "1"


def current() -> Snapshot | None:
    return _cur


def check(store=None) -> bool:
    """Look at the pointer once: a snapshot it names that isn't the one in use is fetched to local disk, verified and
    swapped in. Returns whether it swapped. Never raises: a bad snapshot is logged and the current one kept."""
    global _cur
    _status["checked_at"] = time.time()
    try:
        store = store or open_store()
        ptr = read_pointer(store)
    except Exception as e:  # noqa: BLE001 - the store unreachable or the pointer bad: keep what we have
        _status["error"] = f"pointer: {type(e).__name__}: {str(e)[:200]}"
        log.warning("snapshot.pointer_unreadable error=%s", _status["error"])
        return False
    if ptr is None:
        _status["error"] = None
        return False
    cur = _cur
    if cur is not None and cur.name == ptr["db"]:
        return False
    bad = _status["rejected"].get(ptr["db"])
    if bad and bad["sha256"] == ptr["sha256"] and time.time() - bad["at"] < 600:   # not again every minute
        return False
    t0 = time.time()
    try:
        snap = _fetch(store, ptr)
    except Exception as e:  # noqa: BLE001 - corrupt, mismatched, unreachable: rejected, the old one stays
        _status["rejected"][ptr["db"]] = {"sha256": ptr["sha256"], "at": time.time(), "error": f"{type(e).__name__}: {str(e)[:300]}"}
        _status["error"] = f"rejected {ptr['db']}: {type(e).__name__}: {str(e)[:200]}"
        log.error("snapshot.rejected db=%s error=%s", ptr["db"], _status["error"])
        return False
    with _lock:
        old, _cur = _cur, snap
    _status["error"] = None
    log.info("snapshot.swapped db=%s built_at=%s envs=%s tasks=%s seconds=%.2f", snap.name, snap.built_at,
             snap.counts.get("envs"), snap.counts.get("tasks"), time.time() - t0)
    if old is not None:
        old.retire()
    if _fb is not None:   # built while there was none: not needed any more
        _swap_fallback(None)
    _prune_local(keep={snap.path.name})
    _prune_fallbacks()
    return True


def _fetch(store, ptr: dict[str, Any]) -> Snapshot:
    """The pointer's database on local disk, sha256 checked (a copy already there is reused if it matches)."""
    local = local_dir() / Path(ptr["db"]).name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not (local.exists() and sha256_file(local) == ptr["sha256"]):
        tmp = _tmp_name(local)
        try:
            digest = store.fetch(ptr["db"], tmp)
            if digest != ptr["sha256"]:
                raise SnapshotError(f"sha256 {digest[:12]}…, the pointer says {ptr['sha256'][:12]}…")
            tmp.replace(local)
        finally:
            tmp.unlink(missing_ok=True)
    snap = Snapshot(local, name=ptr["db"], source="snapshot", built_at=ptr.get("built_at"))
    try:
        snap.check({k: v for k, v in (ptr.get("counts") or {}).items() if k in ("envs", "tasks")})
    except SnapshotError:
        local.unlink(missing_ok=True)   # don't reuse it next time
        raise
    return snap


def _prune_local(keep: set[str]) -> None:
    """Local copies: the one in use and the two before it stay (another worker may still read them); older go."""
    try:
        files = sorted(local_dir().glob("catalog-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for p in files[3:]:
        if p.name not in keep and time.time() - p.stat().st_mtime > 600:
            p.unlink(missing_ok=True)


def _prune_fallbacks() -> None:
    """Fallback files of processes that are gone (a restart, a reload): deleted. Each process deletes its own when
    it retires one; a process that was killed can't."""
    for p in local_dir().glob("fallback-*-*.db"):
        try:
            pid = int(p.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            p.unlink(missing_ok=True)
        except (PermissionError, OSError):
            continue


def _watch() -> None:
    while True:
        try:
            check()
        except Exception:  # never let the watcher die
            log.exception("snapshot.watch_failed")
        _first.set()
        _touch()
        time.sleep(CHECK_EVERY * (0.9 + 0.2 * (os.getpid() % 10) / 10))


def start() -> None:
    """Start looking at the pointer (once per process): from the router's startup, or a first request."""
    global _watcher
    with _lock:
        if _watcher is not None:
            return
        if not watching():
            _watcher = threading.current_thread()   # a marker: checks happen on requests instead
            return
        _watcher = threading.Thread(target=_watch, daemon=True, name="snapshot-watch")
        _watcher.start()


def get() -> Snapshot:
    """The snapshot to read: the store's, else one built here from the live catalog."""
    start()
    if not watching():
        if time.time() - _status["checked_at"] > CHECK_EVERY:
            check()
        _first.set()
    _first.wait(FIRST_WAIT)
    return _cur or fallback()


@contextmanager
def use() -> Iterator[tuple[Snapshot, sqlite3.Connection]]:
    """The snapshot and a connection to it, for one request: a swap meanwhile doesn't close it under the request. A
    snapshot whose file has gone (a disk cleaner, a hand) is dropped, and the next one is fetched again."""
    for _ in range(3):
        snap = get()
        try:
            snap._enter()
        except SnapshotError:   # retired and closed between get() and here: take the new one
            continue
        try:
            conn = snap._checkout()
        except SnapshotError as e:
            snap._leave(None)
            _drop(snap, e)
            continue
        try:
            yield snap, conn
        finally:
            snap._checkin(conn)
            snap._leave(None)
        return
    raise SnapshotError("no snapshot could be opened")


def _drop(snap: Snapshot, err: Exception) -> None:
    global _cur, _fb
    with _lock:
        if _cur is snap:
            _cur = None
        elif _fb is snap:
            _fb = None
    log.error("snapshot.lost db=%s error=%s", snap.name, err)
    snap.retire()
    _status["checked_at"] = 0.0   # look at the pointer again now
    if watching():
        threading.Thread(target=check, daemon=True, name="snapshot-recheck").start()


def _touch() -> None:
    """Mark the files in use as used, so the disk janitor (runtime.sweep: least recently used first, never anything
    touched in the last hour) leaves them alone."""
    for p in [s.path for s in (_cur, _fb) if s is not None] + [local_dir()]:
        try:
            os.utime(p)
        except OSError:
            pass


def status() -> dict[str, Any]:
    s = _cur or _fb
    return {"source": s.source if s else None, "db": s.name if s else None, "built_at": s.built_at if s else None,
            "counts": s.counts if s else None, "checked_at": _status["checked_at"] or None, "error": _status["error"]}


# ── no snapshot: the same database, built here from the live catalog ──────────
FALLBACK_EVERY = float(os.environ.get("RLX_FALLBACK_EVERY", "30"))   # seconds between looks at whether it's stale


def _fallback_sig() -> tuple:
    from . import catalog

    try:
        idx_m = config.INDEX_DIR.stat().st_mtime
    except OSError:
        idx_m = 0.0
    return (catalog._env_list.get("at"), id(catalog._env_list.get("rows")), idx_m)


def build_fallback(*, tasks: bool = True) -> Snapshot:
    """A snapshot built from catalog.environments() (hidden ones included: they're left out when queried) and, with
    `tasks`, the indexes on disk: a local file, opened like any other."""
    from . import catalog

    rows = catalog.environments(include_hidden=True)
    d = local_dir()
    d.mkdir(parents=True, exist_ok=True)
    _prune_fallbacks()
    path = d / f"fallback-{os.getpid()}-{next(_fb_seq)}.db"
    public = [r["id"] for r in rows if r.get("kind") == "dataset" and not r.get("private") and not r.get("gated")]
    t0 = time.time()
    try:
        counts = build_db(path, rows, iter_index_tasks(public, mimo=MIMO in public) if tasks else (), {"source": "live"})
    except Exception:
        path.unlink(missing_ok=True)
        raise
    log.info("snapshot.fallback_built envs=%s tasks=%s seconds=%.2f", counts["envs"], counts["tasks"], time.time() - t0)
    return Snapshot(path, name=path.name, source="live", built_at=counts["built_at"], counts=counts, owned=True)


def _swap_fallback(snap: Snapshot | None) -> None:
    global _fb
    with _lock:
        old, _fb = _fb, snap
    if old is not None:
        old.retire()


def fallback() -> Snapshot:
    """The live catalog as a snapshot. The first is built at once (tasks too, unless a watcher thread runs: then its
    tasks follow in the background); after that it is rebuilt in the background when the listing or the indexes on
    disk change, and requests keep reading the one they have."""
    fb = _fb
    if fb is None:
        kick = False
        with _fb_first:   # one first build at a time; the others wait for it
            fb = _fb
            if fb is None:
                sync_tasks = not watching()
                _fb_state.update(sig=_fallback_sig(), checked=time.time())
                fb = build_fallback(tasks=sync_tasks)
                _swap_fallback(fb)
                kick = not sync_tasks
        if kick:
            _rebuild_fallback(force=True)
        return fb
    with _fb_lock:
        now = time.time()
        due = now - _fb_state["checked"] > FALLBACK_EVERY and not _fb_state["building"]
        if due:
            _fb_state["checked"] = now
    if due:
        _rebuild_fallback()
    return fb


def _rebuild_fallback(force: bool = False) -> None:
    sig = _fallback_sig()
    with _fb_lock:
        if _fb_state["building"] or (not force and sig == _fb_state["sig"]):
            return
        _fb_state["building"] = True

    def run() -> None:
        try:
            snap = build_fallback(tasks=True)
            _fb_state["sig"] = sig
            if _cur is None:
                _swap_fallback(snap)
            else:   # a snapshot arrived meanwhile
                snap.retire()
        except Exception:  # the last one stays
            log.exception("snapshot.fallback_failed")
        finally:
            with _fb_lock:
                _fb_state["building"] = False

    if watching():
        threading.Thread(target=run, daemon=True, name="snapshot-fallback").start()
    else:
        run()


def reset() -> None:
    """Forget every snapshot in use (tests)."""
    global _cur, _fb, _watcher
    with _lock:
        old = [s for s in (_cur, _fb) if s is not None]
        _cur = _fb = None
        _watcher = None
    for s in old:
        s.retire()
    _fb_state.update(sig=None, checked=0.0, building=False)
    _status.update(checked_at=0.0, error=None, rejected={})
    _first.clear()
