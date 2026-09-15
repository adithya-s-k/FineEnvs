"""Indexed access to the entire corpus; metadata lookup never fetches images."""

import hashlib
import io
import json
import random
import re
import shutil
import sqlite3
import threading
import zlib
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import get_token
from PIL import Image

from .cache import DiskCache
from .catalog import PUBLIC_FIELDS, SPLITS
from .index import INDEX_VERSION, sha256_file
from .schema import FAMILIES, canonical_json
from .storage import CorpusStorage, load_manifest
from .tasks import derive_tasks


class CorpusCatalog:
    def __init__(
        self,
        manifest_path,
        cache_dir,
        *,
        source_root=None,
        local_source=False,
        index_cache_bytes=4_000_000_000,
        asset_cache_bytes=512_000_000,
        **storage_options,
    ):
        self.manifest, self.index_root = load_manifest(manifest_path)
        if (
            self.manifest.get("status") != "ready"
            or self.manifest.get("storage") != "bucket-parquet"
            or self.manifest.get("index_version") != INDEX_VERSION
        ):
            raise ValueError("Use a ready, supported full-corpus index")
        inventory = {
            "source": self.manifest["config"]["source"],
            "revision": self.manifest["config"]["revision"],
            "bucket_id": self.manifest["bucket_id"],
            "source_license": self.manifest["source_license"],
            "files": self.manifest["source_files"],
            "source_bytes": self.manifest["media_bytes"],
            "parquet_files": sum(
                f["path"].endswith(".parquet") for f in self.manifest["source_files"]
            ),
        }
        if (
            hashlib.sha256(canonical_json(inventory).encode()).hexdigest()
            != self.manifest["inventory_id"]
        ):
            raise ValueError("Corpus source inventory identity mismatch")
        identity = {
            "index_version": INDEX_VERSION,
            "inventory_id": self.manifest["inventory_id"],
            "split_seed": self.manifest["config"]["split_seed"],
            "indexes": {
                lang: info["sha256"] for lang, info in self.manifest["indexes"].items()
            },
        }
        if (
            hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            != self.manifest["snapshot_id"]
        ):
            raise ValueError("Corpus manifest identity mismatch")
        if any(
            info["path"] != f"{lang}.sqlite"
            for lang, info in self.manifest["indexes"].items()
        ):
            raise ValueError("Invalid index path")
        self.snapshot_id = self.manifest["snapshot_id"]
        self.languages = self.manifest["config"]["languages"]
        self.index_cache = DiskCache(Path(cache_dir) / "indexes", index_cache_bytes)
        self.asset_cache = DiskCache(
            Path(cache_dir) / "assets", asset_cache_bytes, 64_000_000
        )
        self.storage = CorpusStorage(
            self.manifest,
            cache_dir,
            source_root=source_root,
            local_source=local_source,
            **storage_options,
        )
        self.files = {f["path"]: f for f in self.manifest["source_files"]}
        self._page = lru_cache(maxsize=128)(self._load_page)
        self._render_slots = threading.BoundedSemaphore(2)
        self.counts = {
            (c["split"], c["language"], c["family"]): c["tasks"]
            for c in self.manifest["counts"]
        }

    @contextmanager
    def _db(self, language):
        if language not in self.languages:
            raise ValueError(f"Language {language!r} is absent from this corpus")
        info = self.manifest["indexes"][language]
        if not self.index_root.startswith("hf://"):
            path = Path(self.index_root) / info["path"]
            self._check_local_index(str(path), info["sha256"], path.stat().st_mtime_ns)
            with self._connection(path) as db:
                yield db
            return

        def download(path):
            mounted = (
                self.storage.source_root
                / "openenv"
                / "indexes"
                / self.snapshot_id
                / info["path"]
                if self.storage.source_root
                else None
            )
            if mounted is not None and mounted.is_file():
                if (
                    mounted.stat().st_size != info["size"]
                    or info["size"] > self.index_cache.max_entry_bytes
                ):
                    raise ValueError(
                        "Mounted index exceeds expected size or cache budget"
                    )
                shutil.copyfile(mounted, path)
                if sha256_file(path) != info["sha256"]:
                    raise ValueError("Mounted index checksum mismatch")
                return
            uri = self.index_root + "/" + info["path"]
            namespace, bucket, key = uri.removeprefix("hf://buckets/").split("/", 2)
            token = get_token()
            with requests.get(
                f"https://huggingface.co/buckets/{namespace}/{bucket}/resolve/{quote(key)}",
                headers={"Authorization": f"Bearer {token}"} if token else {},
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                used = 0
                with path.open("wb") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        used += len(chunk)
                        if used > min(info["size"], self.index_cache.max_entry_bytes):
                            raise ValueError(
                                "Index download exceeds declared size or cache limit"
                            )
                        target.write(chunk)
                if used != info["size"] or sha256_file(path) != info["sha256"]:
                    raise ValueError("Index checksum mismatch")

        with self.index_cache.lease(info["sha256"], download) as path:
            with self._connection(path) as db:
                yield db

    @staticmethod
    @lru_cache(maxsize=64)
    def _check_local_index(path, expected, mtime):
        if sha256_file(path) != expected:
            raise ValueError("Local index changed from corpus manifest")

    @staticmethod
    @contextmanager
    def _connection(path):
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as db:
            db.row_factory = sqlite3.Row
            yield db

    def count(self, split):
        self._split(split)
        return sum(n for (s, _, _), n in self.counts.items() if s == split)

    @staticmethod
    def _split(split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")

    def group_count(self, split, language, family):
        self._split(split)
        return self.counts.get((split, language, family), 0)

    def _id(self, language, split, position):
        return f"nayana-c1.{self.snapshot_id}.{language}.{split}.{position}"

    def _parse(self, task_id):
        match = re.fullmatch(
            r"nayana-c1\.([0-9a-f]{64})\.([a-z]{2})\.(train|validation|test)\.([0-9]+)",
            task_id,
        )
        if not match or match[1] != self.snapshot_id or match[2] not in self.languages:
            raise KeyError("Task ID belongs to a different or invalid corpus index")
        return match[2], match[3], int(match[4])

    def _load_page(self, language, page_id):
        with self._db(language) as db:
            record = db.execute(
                """
                SELECT p.*, b.row_group, f.path FROM pages p
                JOIN blocks b ON b.id=p.block_id JOIN files f ON f.id=b.file_id WHERE p.id=?
            """,
                (page_id,),
            ).fetchone()
        row = json.loads(zlib.decompress(record["payload"]))
        compiled, _ = derive_tasks(
            row,
            language,
            self.manifest["config"]["revision"],
            self.manifest["config"]["split_seed"],
            metadata_only=True,
        )
        return dict(record), row, {(t["family"], t["unit"]): t for t in compiled}

    def _task(self, language, record):
        page, _, compiled = self._page(language, record["page_id"])
        task = dict(compiled[record["family"], record["unit"]])
        task.pop("media", None)
        task.update(
            task_id=self._id(language, record["split"], record["position"]),
            asset_sha256="",
            snapshot_id=self.snapshot_id,
            _page_index=record["page_id"],
            _source_path=page["path"],
            _row_group=page["row_group"],
            _row_in_group=page["row_in_group"],
            _family_position=record["family_position"],
            block_id=f"{language}:{page['block_id']}",
        )
        return task

    def get(self, task_id):
        language, split, position = self._parse(task_id)
        with self._db(language) as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE split=? AND position=?", (split, position)
            ).fetchone()
        if row is None:
            raise KeyError("No such indexed task")
        return self._task(language, row)

    def at(self, split, index):
        if not 0 <= index < self.count(split):
            raise IndexError("Task index outside split")
        for language in self.languages:
            count = sum(
                self.group_count(split, language, family) for family in FAMILIES
            )
            if index < count:
                return self.get(self._id(language, split, index))
            index -= count
        raise IndexError(index)

    def group_at(self, split, language, family, index):
        if not 0 <= index < self.group_count(split, language, family):
            raise IndexError("Task index outside language/task group")
        with self._db(language) as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE split=? AND family=? AND family_position=?",
                (split, family, index),
            ).fetchone()
        return self._task(language, row)

    def group_position(self, task_id, split, language, family):
        try:
            task_language, task_split, _ = self._parse(task_id)
        except (KeyError, TypeError):
            return None
        if (task_language, task_split) != (language, split):
            return None
        task = self.get(task_id)
        return task["_family_position"] if task["family"] == family else None

    def public(self, task):
        result = {key: task.get(key) for key in PUBLIC_FIELDS}
        result.update(
            snapshot_id=self.snapshot_id,
            media_ready=bool(task.get("asset_sha256")),
            block_id=task["block_id"],
        )
        result["asset_path"] = (
            f"/assets/{task['asset_sha256']}?task_id={quote(task['task_id'])}"
            if task.get("asset_sha256")
            else None
        )
        for key in ("reading_order_policy", "reading_order", "annotation_masked"):
            if key in task:
                result[key] = task[key]
        return result

    def task_range(self, split, start=0, stop=None):
        count = self.count(split)
        start, stop = 0 if start is None else start, count if stop is None else stop
        if not 0 <= start <= stop <= count or stop - start > 1000:
            raise IndexError("Use an in-bounds range of at most 1000 tasks")
        return [self.public(self.at(split, i)) for i in range(start, stop)]

    def _render(self, task):
        _, row, _ = self._page(task["language"], task["_page_index"])
        raw = self.storage.image(
            self.files[task["_source_path"]],
            task["_row_group"],
            task["_row_in_group"],
            task["page_id"],
        )
        tasks, _ = derive_tasks(
            {**row, "jpg": {"bytes": raw}},
            task["language"],
            self.manifest["config"]["revision"],
            self.manifest["config"]["split_seed"],
            selection=(task["family"], task["unit"]),
        )
        selected = next(
            (
                t
                for t in tasks
                if (t["family"], t["unit"]) == (task["family"], task["unit"])
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "Indexed annotation fails validation against actual image dimensions"
            )
        return {**task, **selected, "task_id": task["task_id"]}

    @contextmanager
    def _asset(self, task):
        def load(path):
            with self._render_slots:
                rendered = self._render(task)
            raw = rendered.pop("media")
            rendered["asset_sha256"] = hashlib.sha256(raw).hexdigest()
            header = json.dumps(rendered, ensure_ascii=False).encode()
            if len(header) > 2_000_000:
                raise ValueError("Rendered task metadata exceeds 2 MB")
            with path.open("wb") as output:
                output.write(b"NAY1" + len(header).to_bytes(4, "little") + header + raw)

        # One render per task, shared by concurrent GRPO completions and asset fetches.
        with self.asset_cache.lease(task["task_id"], load) as path:
            with path.open("rb") as source:
                prefix = source.read(8)
                if prefix[:4] != b"NAY1":
                    raise ValueError("Invalid cached asset")
                length = int.from_bytes(prefix[4:], "little")
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid cached asset header")
                metadata = json.loads(source.read(length))
                if metadata["task_id"] != task["task_id"]:
                    raise ValueError("Cached task identity mismatch")
                yield metadata, source

    def materialize(self, task):
        with self._asset(task) as (metadata, _):
            return metadata

    def asset_bytes(self, sha, task_id):
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not task_id:
            raise KeyError("An indexed task ID and valid asset hash are required")
        task = self.get(task_id)
        with self._asset(task) as (metadata, source):
            if metadata["asset_sha256"] != sha:
                raise KeyError("Asset does not match this task")
            raw = source.read()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Cached image checksum mismatch")
            return raw, metadata["mime"]

    def image(self, task):
        task = self.materialize(task)
        raw, _ = self.asset_bytes(task["asset_sha256"], task["task_id"])
        with Image.open(io.BytesIO(raw)) as image:
            return image.copy()

    def blocks(self, split, languages=None, families=None):
        self._split(split)
        languages = languages or self.languages
        families = families or FAMILIES
        if (
            set(languages) - set(self.languages)
            or len(set(languages)) != len(languages)
            or len(set(families)) != len(families)
            or not families
            or set(families) - set(FAMILIES)
        ):
            raise ValueError("Invalid language/task selection")
        result = []
        placeholders = ",".join("?" for _ in families)
        for language in languages:
            with self._db(language) as db:
                rows = db.execute(
                    f"""
                    SELECT b.id,b.rows,b.image_bytes,COUNT(*) AS tasks FROM blocks b
                    JOIN tasks t ON t.block_id=b.id WHERE t.split=? AND t.family IN ({placeholders})
                    GROUP BY b.id ORDER BY b.id
                """,
                    (split, *families),
                ).fetchall()
            result.extend(
                {
                    "block_id": f"{language}:{r['id']}",
                    "language": language,
                    "pages": r["rows"],
                    "image_bytes": r["image_bytes"],
                    "tasks": r["tasks"],
                }
                for r in rows
            )
        return result

    def _block(self, block_id):
        match = re.fullmatch(r"([a-z]{2}):([0-9]+)", block_id)
        if not match or match[1] not in self.languages:
            raise KeyError("Invalid source block")
        language, index = match[1], int(match[2])
        with self._db(language) as db:
            row = db.execute(
                "SELECT b.*,f.path FROM blocks b JOIN files f ON f.id=b.file_id WHERE b.id=?",
                (index,),
            ).fetchone()
        if row is None:
            raise KeyError("Unknown source block")
        return language, index, dict(row)

    def block_tasks(self, block_id, split, families=None, start=0, limit=512):
        self._split(split)
        families = families or FAMILIES
        if (
            not 0 <= start
            or not 1 <= limit <= 1000
            or not families
            or set(families) - set(FAMILIES)
        ):
            raise ValueError("Invalid block range or task families")
        language, index, _ = self._block(block_id)
        placeholders = ",".join("?" for _ in families)
        with self._db(language) as db:
            rows = db.execute(
                f"""SELECT split,position,family FROM tasks
                WHERE block_id=? AND split=? AND family IN ({placeholders})
                ORDER BY position LIMIT ? OFFSET ?""",
                (index, split, *families, limit, start),
            ).fetchall()
        # OFFSET is bounded within one physical row group, never the million-page corpus.
        return [
            {
                "task_id": self._id(language, r["split"], r["position"]),
                "language": language,
                "family": r["family"],
                "block_id": block_id,
            }
            for r in rows
        ]

    def prefetch(self, *, task_ids=(), block_ids=()):
        if len(task_ids) + len(block_ids) > 64:
            raise ValueError("Prefetch accepts at most 64 tasks/blocks")
        items = {}
        for task_id in task_ids:
            task = self.get(task_id)
            spec, group = self.files[task["_source_path"]], task["_row_group"]
            items[spec["path"], group] = (spec, group)
        for block_id in block_ids:
            _, _, block = self._block(block_id)
            spec = self.files[block["path"]]
            items[spec["path"], block["row_group"]] = (spec, block["row_group"])
        return self.storage.prefetch(items.values())

    def sample(self, split, languages, families, per_group=4, seed=42):
        if (
            not 1 <= per_group <= 1000
            or len(languages) * len(families) * per_group > 8192
        ):
            raise ValueError("Sample exceeds request limits")
        if (
            not languages
            or not families
            or len(set(languages)) != len(languages)
            or len(set(families)) != len(families)
            or set(languages) - set(self.languages)
            or set(families) - set(FAMILIES)
        ):
            raise ValueError("Invalid sample groups")
        count = min(
            self.group_count(split, language, family)
            for language in languages
            for family in families
        )
        if not count:
            raise ValueError("Missing language/task groups in selected split")
        rng = random.Random(seed)
        groups = []
        for language in languages:
            for family in families:
                rows = [
                    self.group_at(split, language, family, i)
                    for i in rng.sample(
                        range(self.group_count(split, language, family)),
                        min(count, per_group),
                    )
                ]
                groups.append(
                    [
                        {k: r[k] for k in ("task_id", "language", "family", "block_id")}
                        for r in rows
                    ]
                )
        return [row for group in zip(*groups, strict=True) for row in group]

    def stats(self):
        return {
            "indexes": self.index_cache.stats(),
            "groups": self.storage.stats(),
            "assets": self.asset_cache.stats(),
            "snapshot_id": self.snapshot_id,
        }

    def close(self):
        self.storage.close()
        self._page.cache_clear()
