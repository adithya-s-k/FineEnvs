"""A serving index for one release, built once so that a server starts in seconds.

The release (``LiteFold/RetroEnv``) stays the source of truth. The index holds what every
server process would otherwise derive from it at startup: the task rows (about 160 MB of
JSON), the stock's canonical SMILES and fingerprints, and the train-visible precedents with
their leakage keys and fingerprints. Deriving those took about 65 seconds and a gigabyte of
parsed molecules per process; reading them back takes a few seconds.

Fingerprints, canonical SMILES and scaffold keys depend on the RDKit version, so the index
records the version that built it. A server running another version rebuilds those parts
from the release instead of trusting them. Task rows do not depend on RDKit: they are the
release's own JSON lines, compressed and stored verbatim.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import zlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import rdkit
from rdkit import DataStructs

from .leakage import LeakageRules
from .models import RetroTask
from .reactions import iter_reactions
from .retrieval import Precedent, PrecedentIndex, StockIndex, register_stock_index
from .store import load_stock

INDEX_VERSION = 1
INDEX_FILE = "serving.sqlite"
TASK_CACHE = 4096

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE tasks (
    split TEXT NOT NULL,
    position INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    tier TEXT,
    min_depth INTEGER,
    record BLOB NOT NULL,
    PRIMARY KEY (split, position)
);
CREATE INDEX tasks_by_id ON tasks (task_id);
CREATE TABLE stock (
    stock_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    smiles TEXT NOT NULL,
    fingerprint BLOB NOT NULL,
    PRIMARY KEY (stock_id, position)
);
CREATE TABLE precedents (
    position INTEGER PRIMARY KEY,
    product TEXT NOT NULL,
    reactants TEXT NOT NULL,
    reaction_class TEXT NOT NULL,
    reagents TEXT NOT NULL,
    patents TEXT NOT NULL,
    keys TEXT NOT NULL,
    fingerprint BLOB NOT NULL
);
"""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(release: str | Path, output: str | Path) -> dict[str, Any]:
    """Write ``output/serving.sqlite`` from a release directory and describe what it holds.

    Stock and precedents go through ``StockIndex`` and ``PrecedentIndex`` themselves, so the
    stored values are exactly what a server computing them at startup would hold.
    """
    release, output = Path(release), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / INDEX_FILE
    partial = target.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    manifest = json.loads((release / "manifest.json").read_text())
    rules = LeakageRules.from_manifest(manifest)

    db = sqlite3.connect(partial)
    try:
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + _SCHEMA)
        splits = _write_tasks(db, release / "tasks-private")
        stocks = _write_stocks(db, release / "stocks")
        precedents = _write_precedents(db, release / "library" / "reactions.jsonl.gz", rules)
        meta = {
            "index_version": str(INDEX_VERSION),
            "rdkit_version": rdkit.__version__,
            "release_manifest_sha256": sha256_file(release / "manifest.json"),
            "release_checksums_sha256": sha256_file(release / "checksums.json"),
        }
        db.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    partial.replace(target)
    described = describe_index(target)
    if (described["splits"], described["stocks"], described["precedents"]) != (splits, stocks, precedents):
        raise RuntimeError(f"{target} does not read back what was written")
    return described


def describe_index(path: str | Path) -> dict[str, Any]:
    """Size, digest, versions and contents of an existing serving index."""
    path = Path(path)
    index = ServingIndex(path)
    db = index._db()
    splits: dict[str, dict[str, Any]] = {}
    for split, tier, count in db.execute("SELECT split, tier, COUNT(*) FROM tasks GROUP BY split, tier"):
        entry = splits.setdefault(split, {"tasks": 0, "tiers": {}})
        entry["tasks"] += count
        entry["tiers"][str(tier)] = count
    for entry in splits.values():
        entry["tiers"] = dict(sorted(entry["tiers"].items()))
    return {
        "path": INDEX_FILE,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "index_version": int(index.meta["index_version"]),
        "rdkit_version": index.meta["rdkit_version"],
        "release_checksums_sha256": index.meta["release_checksums_sha256"],
        "splits": dict(sorted(splits.items())),
        "stocks": dict(db.execute("SELECT stock_id, COUNT(*) FROM stock GROUP BY stock_id ORDER BY stock_id")),
        "precedents": db.execute("SELECT COUNT(*) FROM precedents").fetchone()[0],
    }


def _write_tasks(db: sqlite3.Connection, tasks_dir: Path) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    for path in sorted(tasks_dir.glob("*.jsonl")):
        split, seen, tiers, rows = path.stem, set(), {}, []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                task = RetroTask.from_dict(json.loads(line))  # the same validation TaskStore applies
                if task.split != split:
                    raise ValueError(f"{path}:{line_number}: task split {task.split!r} != {split!r}")
                if task.task_id in seen:
                    raise ValueError(f"duplicate task_id in {path}: {task.task_id}")
                seen.add(task.task_id)
                tier = task.difficulty.get("tier")
                tiers[tier] = tiers.get(tier, 0) + 1
                rows.append(
                    (
                        split,
                        len(rows),
                        task.task_id,
                        task.parent_id,
                        task.stock_id,
                        task.variant,
                        tier,
                        task.difficulty.get("min_depth"),
                        zlib.compress(line.strip().encode("utf-8"), 6),
                    )
                )
        db.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        splits[split] = {"tasks": len(rows), "tiers": dict(sorted((str(k), v) for k, v in tiers.items()))}
    if not splits:
        raise ValueError(f"no task splits in {tasks_dir}")
    return splits


def _write_stocks(db: sqlite3.Connection, stocks_dir: Path) -> dict[str, int]:
    stocks: dict[str, int] = {}
    for path in sorted(stocks_dir.glob("*.smi")):
        index = StockIndex(load_stock(path))
        db.executemany(
            "INSERT INTO stock VALUES (?, ?, ?, ?)",
            (
                (path.stem, position, smiles, fingerprint.ToBinary())
                for position, (smiles, fingerprint) in enumerate(zip(index.smiles, index.fingerprints()))
            ),
        )
        stocks[path.stem] = len(index.smiles)
    if not stocks:
        raise ValueError(f"no stocks in {stocks_dir}")
    return stocks


def _write_precedents(db: sqlite3.Connection, reactions: Path, rules: LeakageRules) -> int:
    index = PrecedentIndex(iter_reactions(reactions), rules)
    db.executemany(
        "INSERT INTO precedents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            (
                position,
                record.product,
                json.dumps(list(record.reactants)),
                record.reaction_class,
                json.dumps(list(record.reagents)),
                json.dumps(list(record.patents)),
                json.dumps(sorted([list(key) for key in record.keys])),
                fingerprint.ToBinary(),
            )
            for position, (record, fingerprint) in enumerate(zip(index.records, index.fingerprints()))
        ),
    )
    return len(index.records)


class ServingIndex:
    """Read-only access to ``serving.sqlite``; one connection per thread."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"serving index not found: {self.path}")
        self._local = threading.local()
        self.meta = dict(self._db().execute("SELECT key, value FROM meta"))
        if int(self.meta.get("index_version", 0)) != INDEX_VERSION:
            raise ValueError(f"{self.path} is index version {self.meta.get('index_version')}, expected {INDEX_VERSION}")

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            # immutable=1: the file never changes under a running server, so SQLite skips locking.
            db = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
            self._local.db = db
        return db

    @property
    def rdkit_matches(self) -> bool:
        """Whether this process can trust the stored fingerprints, canonical SMILES and keys."""
        return self.meta.get("rdkit_version") == rdkit.__version__

    def stock_smiles(self, stock_id: str) -> list[str]:
        rows = self._db().execute("SELECT smiles FROM stock WHERE stock_id = ? ORDER BY position", (stock_id,))
        return [smiles for (smiles,) in rows]

    def stock_index(self, stock_id: str) -> StockIndex:
        """The stock's index: stored fingerprints when RDKit matches, otherwise recomputed."""
        if not self.rdkit_matches:
            return StockIndex(self.stock_smiles(stock_id))
        rows = (
            self._db()
            .execute("SELECT smiles, fingerprint FROM stock WHERE stock_id = ? ORDER BY position", (stock_id,))
            .fetchall()
        )
        if not rows:
            raise FileNotFoundError(f"stock {stock_id!r} is not in {self.path}")
        return StockIndex.from_canonical(
            (smiles for smiles, _ in rows), (DataStructs.ExplicitBitVect(blob) for _, blob in rows)
        )

    def precedents(self, rules: LeakageRules, reactions: Path | None = None) -> PrecedentIndex:
        """Stored precedents, or a rebuild from the release's reactions when RDKit differs."""
        if not self.rdkit_matches:
            if reactions is None:
                raise ValueError("RDKit differs from the index; pass the release's reactions to rebuild precedents")
            return PrecedentIndex(iter_reactions(reactions), rules)
        records, fingerprints = [], []
        for product, reactants, reaction_class, reagents, patents, keys, blob in self._db().execute(
            "SELECT product, reactants, reaction_class, reagents, patents, keys, fingerprint "
            "FROM precedents ORDER BY position"
        ):
            records.append(
                Precedent(
                    product=product,
                    reactants=tuple(json.loads(reactants)),
                    reaction_class=reaction_class,
                    reagents=tuple(json.loads(reagents)),
                    patents=tuple(json.loads(patents)),
                    keys=frozenset((kind, value) for kind, value in json.loads(keys)),
                )
            )
            fingerprints.append(DataStructs.ExplicitBitVect(blob))
        return PrecedentIndex.from_records(records, fingerprints, rules)

    def task_store(self) -> "SqliteTaskStore":
        return SqliteTaskStore(self)


class TaskList(Sequence):
    """One split's tasks, read from the index on demand; behaves like the tuple ``TaskStore`` returns."""

    def __init__(self, store: "SqliteTaskStore", split: str, count: int):
        self._store, self._split, self._count = store, split, count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        index = int(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(f"task index {index} is out of range for split {self._split!r}")
        return self._store._task(self._split, index)

    def __iter__(self) -> Iterator[RetroTask]:
        rows = self._store.index._db().execute(
            "SELECT record FROM tasks WHERE split = ? ORDER BY position", (self._split,)
        )
        for (record,) in rows:
            yield _decode(record)


def _decode(record: bytes) -> RetroTask:
    return RetroTask.from_dict(json.loads(zlib.decompress(record)))


class SqliteTaskStore:
    """``TaskStore``'s interface over a serving index, without loading every task up front."""

    def __init__(self, index: ServingIndex):
        self.index = index
        db = index._db()
        self._counts = dict(db.execute("SELECT split, COUNT(*) FROM tasks GROUP BY split ORDER BY split"))
        self._stock_ids = [stock_id for (stock_id,) in db.execute("SELECT DISTINCT stock_id FROM stock")]
        self._stocks: dict[str, frozenset[str]] = {}
        self._stock_lock = threading.Lock()
        self._task = lru_cache(maxsize=TASK_CACHE)(self._load_task)

    def splits(self) -> list[str]:
        return sorted(self._counts)

    def tasks(self, split: str) -> TaskList:
        if split not in self._counts:
            raise KeyError(f"unknown split {split!r}; available: {self.splits()}")
        return TaskList(self, split, self._counts[split])

    def task(self, split: str, index: int) -> RetroTask:
        return self.tasks(split)[index]

    def _load_task(self, split: str, position: int) -> RetroTask:
        row = (
            self.index._db()
            .execute("SELECT record FROM tasks WHERE split = ? AND position = ?", (split, position))
            .fetchone()
        )
        if row is None:
            raise IndexError(f"task index {position} is out of range for split {split!r}")
        return _decode(row[0])

    def position(self, split: str, task_id: str) -> int | None:
        row = (
            self.index._db()
            .execute("SELECT position FROM tasks WHERE split = ? AND task_id = ?", (split, task_id))
            .fetchone()
        )
        return None if row is None else int(row[0])

    def find(self, task_id: str) -> RetroTask | None:
        row = (
            self.index._db()
            .execute("SELECT split, position FROM tasks WHERE task_id = ? ORDER BY split LIMIT 1", (task_id,))
            .fetchone()
        )
        return None if row is None else self._task(row[0], int(row[1]))

    def stock_ids(self) -> list[str]:
        return sorted(self._stock_ids)

    def stock(self, stock_id: str) -> frozenset[str]:
        """The stock as one shared set, its index registered so sessions never rebuild it."""
        with self._stock_lock:
            if stock_id not in self._stocks:
                self._stocks[stock_id] = register_stock_index(self.index.stock_index(stock_id))
            return self._stocks[stock_id]

    def public_task(self, split: str, index: int) -> dict:
        return self.task(split, index).to_dict(include_hidden=False)

    def iter_all(self) -> Iterable[RetroTask]:
        for split in self.splits():
            yield from self.tasks(split)
