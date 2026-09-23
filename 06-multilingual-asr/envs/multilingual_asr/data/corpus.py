"""Serve the whole indexed corpus. Metadata lookups never touch audio."""

import hashlib
import sqlite3
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

from .evalset import eval_split_name
from .index import INDEX_VERSION
from .schema import FAMILIES, SPLITS, canonical_json, character_scored, parse_task_id
from .storage import CorpusStorage, load_manifest, runtime_bucket_id
from .tasks import PROMPTS, SAMPLING_RATE

PUBLIC_FIELDS = (
    "task_id",
    "split",
    "language",
    "language_name",
    "family",
    "sample_id",
    "prompt",
    "asset_sha256",
    "mime",
    "sampling_rate",
    "num_samples",
    "duration_seconds",
    "error_unit",
)


class CorpusCatalog:
    def __init__(
        self,
        manifest_path,
        cache_dir,
        *,
        source_root=None,
        local_source=False,
        index_cache_bytes=2_000_000_000,
        evalsets=(),
        **storage_options,
    ):
        self.manifest, self.index_root = load_manifest(manifest_path)
        if (
            self.manifest.get("status") != "ready"
            or self.manifest.get("storage") != "bucket-parquet"
            or self.manifest.get("index_version") != INDEX_VERSION
        ):
            raise ValueError("Use a ready, supported full-corpus index")
        identity = {
            "index_version": INDEX_VERSION,
            "inventory_id": self.manifest["inventory_id"],
            "indexes": {
                lang: info["sha256"] for lang, info in self.manifest["indexes"].items()
            },
        }
        if (
            hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            != self.manifest["snapshot_id"]
        ):
            raise ValueError("Corpus manifest identity mismatch")
        self.snapshot_id = self.manifest["snapshot_id"]
        self.languages = self.manifest["config"]["languages"]
        from .cache import DiskCache

        self.index_cache = DiskCache(Path(cache_dir) / "indexes", index_cache_bytes)
        self.storage = CorpusStorage(
            self.manifest,
            cache_dir,
            source_root=source_root,
            local_source=local_source,
            **storage_options,
        )
        self.files = {f["path"]: f for f in self.manifest["source_files"]}
        self.counts = {
            (c["language"], c["split"], c["family"]): c["tasks"]
            for c in self.manifest["counts"]
        }
        self._utterance = lru_cache(maxsize=512)(self._load_utterance)
        # A frozen evaluation set is addressable as its own split, so a caller can browse
        # or serve exactly the pinned tasks without first loading the JSON and resetting
        # by id. Its tasks live in the source splits; this is a view, not a copy.
        self.eval_splits = {}
        for record in evalsets:
            name = eval_split_name(record)
            if name in SPLITS:
                raise ValueError(f"{name!r} collides with a source split")
            if name in self.eval_splits:
                # Two sets covering the same languages and split would otherwise share a
                # name and one would silently replace the other.
                raise ValueError(
                    f"Two evaluation sets are both named {name!r}; they cover the same "
                    "languages and split, so one would be served in place of the other"
                )
            self.eval_splits[name] = [
                {k: entry[k] for k in ("task_id", "language", "family")}
                for entry in record["tasks"]
            ]

    @contextmanager
    def _db(self, language):
        if language not in self.languages:
            raise ValueError(f"Language {language!r} is absent from this corpus")
        info = self.manifest["indexes"][language]
        if not self.index_root.startswith("hf://"):
            path = Path(self.index_root) / info["path"]
        else:

            def download(target):
                import requests
                from huggingface_hub import get_token

                mounted = (
                    self.storage.source_root
                    / "openenv"
                    / "indexes"
                    / self.snapshot_id
                    / info["path"]
                    if self.storage.source_root
                    else None
                )
                if mounted and mounted.is_file():
                    target.write_bytes(mounted.read_bytes())
                    return
                token = get_token()
                url = (
                    f"https://huggingface.co/buckets/{runtime_bucket_id(self.manifest)}"
                    f"/resolve/openenv/indexes/{self.snapshot_id}/{info['path']}"
                )
                with requests.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"} if token else {},
                    timeout=600,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    with target.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)

            with self.index_cache.lease(f"index:{language}", download) as path:
                # Never query SQLite through a remote mount; it is a local copy by now.
                self._check(path, info["sha256"])
                with self._connect(path) as db:
                    yield db
                return
        self._check(path, info["sha256"])
        with self._connect(path) as db:
            yield db

    @staticmethod
    @lru_cache(maxsize=256)
    def _verified(path, expected, mtime):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Index {path} does not match the manifest checksum")
        return True

    def _check(self, path, expected):
        self._verified(str(path), expected, Path(path).stat().st_mtime_ns)

    @staticmethod
    @contextmanager
    def _connect(path):
        with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            yield db

    def splits(self):
        """Source splits plus every frozen evaluation set served as its own split."""
        return list(SPLITS) + sorted(self.eval_splits)

    def _split(self, split):
        if split not in SPLITS and split not in self.eval_splits:
            raise ValueError(f"Unknown split {split!r}")

    def count(self, split):
        self._split(split)
        if split in self.eval_splits:
            return len(self.eval_splits[split])
        return sum(n for (lang, s, fam), n in self.counts.items() if s == split)

    def group_count(self, split, language, family):
        if split in self.eval_splits:
            return sum(
                1
                for row in self.eval_splits[split]
                if row["language"] == language and row["family"] == family
            )
        return self.counts.get((language, split, family), 0)

    def _load_utterance(self, language, split, row_id):
        with self._db(language) as db:
            row = db.execute(
                """SELECT u.*, f.path AS file_path, f.size AS file_size
                   FROM utterances u JOIN files f ON f.id=u.file_id WHERE u.id=?""",
                (row_id,),
            ).fetchone()
        if row is None:
            raise KeyError("Unknown utterance")
        return dict(row)

    def _task(self, language, row):
        utterance = self._utterance(language, row["split"], row["utterance"])
        family = row["family"]
        reference = (
            language
            if family == "language_id"
            else utterance["transcription"]
            if family == "transcription"
            else utterance["raw_transcription"]
        )
        name = utterance["language_name"] or language
        duration = utterance["num_samples"] / SAMPLING_RATE
        return {
            "task_id": row["id"],
            "split": row["split"],
            "language": language,
            "language_name": name,
            "family": family,
            "sample_id": utterance["sample_id"],
            "recording": utterance["recording"],
            "prompt": PROMPTS[family].format(name=name),
            "reference": reference,
            "mime": "audio/wav",
            "sampling_rate": SAMPLING_RATE,
            "num_samples": utterance["num_samples"],
            "duration_seconds": round(duration, 3),
            "error_unit": "cer" if character_scored(language) else "wer",
            "asset_sha256": "",
            "snapshot_id": self.snapshot_id,
            "_source_path": utterance["file_path"],
            "_row_group": utterance["row_group"],
            "_row_in_group": utterance["row_in_group"],
        }

    def get(self, task_id):
        language, split = parse_task_id(task_id)
        with self._db(language) as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task {task_id!r} is absent from this snapshot")
        return self._task(language, dict(row))

    def at(self, split, index):
        self._split(split)
        if not 0 <= index < self.count(split):
            raise IndexError("Task index outside split")
        if split in self.eval_splits:
            return self.get(self.eval_splits[split][index]["task_id"])
        for language in self.languages:
            total = sum(
                self.group_count(split, language, family) for family in FAMILIES
            )
            if index < total:
                with self._db(language) as db:
                    row = db.execute(
                        "SELECT * FROM tasks WHERE split=? AND position=?",
                        (split, index),
                    ).fetchone()
                if row is None:
                    raise IndexError(
                        f"{language}/{split} has no task at position {index}; the "
                        "manifest count disagrees with the index"
                    )
                return self._task(language, dict(row))
            index -= total
        raise IndexError(index)

    def _eval_group(self, split, language, family):
        return [
            row
            for row in self.eval_splits[split]
            if row["language"] == language and row["family"] == family
        ]

    def group_at(self, split, language, family, index):
        if not 0 <= index < self.group_count(split, language, family):
            raise IndexError("Task index outside language/task group")
        if split in self.eval_splits:
            return self.get(self._eval_group(split, language, family)[index]["task_id"])
        with self._db(language) as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE split=? AND family=? AND family_position=?",
                (split, family, index),
            ).fetchone()
        if row is None:
            raise IndexError(
                f"{language}/{split}/{family} has no task at position {index}; the "
                "manifest count disagrees with the index"
            )
        return self._task(language, dict(row))

    def group_position(self, task_id, split, language, family):
        if split in self.eval_splits:
            rows = [r["task_id"] for r in self._eval_group(split, language, family)]
            return rows.index(task_id) if task_id in rows else None
        try:
            found_language, found_split = parse_task_id(task_id)
        except KeyError:
            return None
        if (found_language, found_split) != (language, split):
            return None
        with self._db(language) as db:
            row = db.execute(
                "SELECT family_position, family FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return row["family_position"] if row and row["family"] == family else None

    def materialize(self, task):
        """Fetch the audio once and pin its hash onto the task."""
        raw = self.audio_bytes(task)
        return {**task, "asset_sha256": hashlib.sha256(raw).hexdigest()}

    def audio_bytes(self, task):
        spec = self.files[task["_source_path"]]
        return self.storage.audio(
            spec, task["_row_group"], task["_row_in_group"], task["sample_id"]
        )

    def asset(self, sha, task_id):
        task = self.materialize(self.get(task_id))
        if task["asset_sha256"] != sha:
            raise KeyError("Asset does not match this task")
        return self.audio_bytes(task), task["mime"]

    def audio_file(self, task):
        """A local path for this task's audio, written from the cached row group."""
        import tempfile

        task = task if task.get("asset_sha256") else self.materialize(task)
        directory = Path(tempfile.gettempdir()) / "fleurs-audio"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{task['asset_sha256']}.wav"
        if not path.exists():
            path.write_bytes(self.audio_bytes(task))
        return str(path)

    def public(self, task):
        result = {key: task.get(key) for key in PUBLIC_FIELDS}
        result.update(
            snapshot_id=self.snapshot_id, media_ready=bool(task.get("asset_sha256"))
        )
        result["asset_path"] = (
            f"/assets/{task['asset_sha256']}?task_id={task['task_id']}"
            if task.get("asset_sha256")
            else None
        )
        return result

    def task_range(self, split, start=0, stop=None):
        count = self.count(split)
        start, stop = 0 if start is None else start, count if stop is None else stop
        if not 0 <= start <= stop <= count or stop - start > 1000:
            raise IndexError("Use an in-bounds range of at most 1000 tasks")
        return [self.public(self.at(split, i)) for i in range(start, stop)]

    def stats(self):
        return {
            "indexes": self.index_cache.stats(),
            "groups": self.storage.stats(),
            "snapshot_id": self.snapshot_id,
        }

    def close(self):
        self.storage.close()
        self._utterance.cache_clear()
