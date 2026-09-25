"""SQLite metadata plus content-addressed images. Discovery never decodes media."""

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from .schema import SCHEMA_VERSION

SPLITS = ("train", "validation", "test")
PUBLIC_FIELDS = (
    "task_id",
    "split",
    "language",
    "page_id",
    "document_id",
    "family",
    "unit",
    "prompt",
    "asset_sha256",
    "mime",
    "width",
    "height",
    "bbox",
    "page_width",
    "page_height",
)


class Catalog:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("status") != "ready":
            raise ValueError("Snapshot must be finalized before serving")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Prepare a new schema-v{SCHEMA_VERSION} window; this snapshot uses an older task contract"
            )

    @contextmanager
    def _connect(self):
        with closing(
            sqlite3.connect(
                (self.directory / "catalog.sqlite").as_uri() + "?mode=ro", uri=True
            )
        ) as db:
            yield db

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
                AND json_extract(payload,'$.language')=? AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchone()[0]

    def _group_ids(self, split, language, family):
        with self._connect() as db:
            rows = db.execute(
                """SELECT id,json_extract(payload,'$.page_id'),json_extract(payload,'$.unit')
                FROM tasks WHERE split=? AND json_extract(payload,'$.language')=?
                AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchall()
        return [
            r[0]
            for r in sorted(
                rows,
                key=lambda r: (
                    tuple(int(n) for n in re.findall(r"\d+", r[1])),
                    r[2],
                    r[0],
                ),
            )
        ]

    def group_at(self, split, language, family, index):
        ids = self._group_ids(split, language, family)
        if not 0 <= index < len(ids):
            raise IndexError("Task index outside group")
        return self.get(ids[index])

    def group_position(self, task_id, split, language, family):
        ids = self._group_ids(split, language, family)
        return ids.index(task_id) if task_id in ids else None

    def materialize(self, task):
        return task

    def image(self, task):
        from PIL import Image

        with Image.open(self.asset(task["asset_sha256"])[0]) as image:
            return image.copy()

    @staticmethod
    def _split(split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")

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
        result = {
            **{key: task[key] for key in PUBLIC_FIELDS},
            "asset_path": f"/assets/{task['asset_sha256']}",
            "snapshot_id": self.manifest["snapshot_id"],
        }
        for key in ("reading_order_policy", "reading_order", "annotation_masked"):
            if key in task:
                result[key] = task[key]
        return result

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
                "SELECT payload FROM tasks WHERE split=? AND position>=? AND position<? ORDER BY position",
                (split, start, stop),
            ).fetchall()
        return [self.public(json.loads(row[0])) for row in rows]

    def asset(self, sha):
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise KeyError("Invalid asset hash")
        with self._connect() as db:
            row = db.execute("SELECT mime FROM assets WHERE sha=?", (sha,)).fetchone()
        if row is None:
            raise KeyError("Unknown asset")
        return self.directory / "assets" / sha, row[0]
