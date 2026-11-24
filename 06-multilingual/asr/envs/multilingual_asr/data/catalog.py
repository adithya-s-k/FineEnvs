"""Serve a prepared snapshot. Discovery never returns a reference or decodes audio."""

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from .schema import SCHEMA_VERSION, SPLITS

PUBLIC_FIELDS = (
    "task_id",
    "split",
    "language",
    "language_name",
    "family",
    "sample_id",
    "recording",
    "prompt",
    "asset_sha256",
    "mime",
    "sampling_rate",
    "num_samples",
    "duration_seconds",
    "error_unit",
)


class Catalog:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("status") != "ready":
            raise ValueError("Snapshot must be finalized before serving")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Prepare a schema-v{SCHEMA_VERSION} snapshot; this one uses an older contract"
            )
        self.snapshot_id = self.manifest["snapshot_id"]
        self.languages = self.manifest["config"]["languages"]

    @contextmanager
    def _connect(self):
        with closing(
            sqlite3.connect(
                (self.directory / "catalog.sqlite").as_uri() + "?mode=ro", uri=True
            )
        ) as db:
            db.row_factory = sqlite3.Row
            yield db

    @staticmethod
    def _split(split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")

    def count(self, split):
        self._split(split)
        with self._connect() as db:
            return db.execute(
                "SELECT COUNT(*) FROM tasks WHERE split=?", (split,)
            ).fetchone()[0]

    def group_count(self, split, language, family):
        with self._connect() as db:
            return db.execute(
                """SELECT COUNT(*) FROM tasks WHERE split=?
                   AND json_extract(payload,'$.language')=?
                   AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchone()[0]

    def _group_ids(self, split, language, family):
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, json_extract(payload,'$.sample_id') FROM tasks
                   WHERE split=? AND json_extract(payload,'$.language')=?
                   AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchall()
        # Ordered by the source utterance id so navigation is stable across snapshots.
        return [row[0] for row in sorted(rows, key=lambda row: (row[1], row[0]))]

    def group_at(self, split, language, family, index):
        ids = self._group_ids(split, language, family)
        if not 0 <= index < len(ids):
            raise IndexError("Task index outside group")
        return self.get(ids[index])

    def group_position(self, task_id, split, language, family):
        ids = self._group_ids(split, language, family)
        return ids.index(task_id) if task_id in ids else None

    def get(self, task_id):
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Task {task_id!r} is absent from this snapshot")
        return json.loads(row[0])

    def at(self, split, index):
        self._split(split)
        if index < 0:
            raise IndexError("Task index must be nonnegative")
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM tasks WHERE split=? AND position=?", (split, index)
            ).fetchone()
        if row is None:
            raise IndexError(f"No task {index} in {split}")
        return json.loads(row[0])

    def public(self, task):
        return {
            **{key: task.get(key) for key in PUBLIC_FIELDS},
            "asset_path": f"/assets/{task['asset_sha256']}",
            "snapshot_id": self.snapshot_id,
        }

    def task_range(self, split, start=0, stop=None):
        count = self.count(split)
        start = 0 if start is None else start
        stop = count if stop is None else stop
        if not 0 <= start <= stop <= count:
            raise IndexError(f"Range must satisfy 0 <= start <= stop <= {count}")
        if stop - start > 1000:
            raise ValueError("Task ranges are limited to 1000; paginate discovery")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM tasks WHERE split=? AND position>=? AND position<? "
                "ORDER BY position",
                (split, start, stop),
            ).fetchall()
        return [self.public(json.loads(row[0])) for row in rows]

    def group_range(self, split, language, family, positions):
        if len(positions) > 1000:
            raise IndexError("Ask for at most 1000 positions")
        return [
            self.public(self.group_at(split, language, family, position))
            for position in positions
        ]

    def asset(self, sha):
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise KeyError("Invalid asset hash")
        with self._connect() as db:
            row = db.execute("SELECT mime FROM assets WHERE sha=?", (sha,)).fetchone()
        if row is None:
            raise KeyError("Unknown asset")
        return self.directory / "assets" / sha, row[0]

    def materialize(self, task):
        return task

    def audio_file(self, task):
        """A local path for this task's audio; already on disk in a snapshot."""
        return str(self.asset(task["asset_sha256"])[0])
