"""Prepare bounded windows using sequential Parquet streaming, with page-level resume."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import LANGUAGES, REPO_ID, REVISION, SCHEMA_VERSION, canonical_json
from .tasks import derive_tasks


@dataclass(frozen=True)
class PrepareConfig:
    languages: tuple[str, ...] = ("en", "kn", "hi", "ar")
    revision: str = REVISION
    pages_per_language: int = 32
    max_media_bytes: int = 1_000_000_000
    max_pixels: int = 50_000_000
    split_seed: int = 42
    num_shards: int = 1
    shard_index: int = 0
    source: str = REPO_ID

    def validate(self):
        if (
            not self.languages
            or len(set(self.languages)) != len(self.languages)
            or set(self.languages) - set(LANGUAGES)
        ):
            raise ValueError("Choose distinct, supported language configs")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Pin the dataset to a full commit SHA")
        if (
            min(
                self.pages_per_language,
                self.max_media_bytes,
                self.max_pixels,
                self.num_shards,
            )
            < 1
        ):
            raise ValueError("Preparation limits must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")


def open_stream(config, language):
    from datasets import Image, load_dataset

    stream = load_dataset(
        config.source,
        language,
        split="train",
        revision=config.revision,
        streaming=True,
        columns=["jpg", "image_id.txt", "regions.json", "vqa.json"],
        batch_size=1,
    ).cast_column("jpg", Image(decode=False))
    if config.num_shards > stream.num_shards:
        raise ValueError(
            f"Requested {config.num_shards} partitions of {stream.num_shards} physical shards"
        )
    return stream.shard(num_shards=config.num_shards, index=config.shard_index)


def _json_write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def prepare(directory, config=None, *, source_factory=open_stream):
    """One writer per output directory. No shuffle buffer or row skipping on resume.

    Page tasks and the stream checkpoint commit in one SQLite transaction. Image
    files are written first; uncommitted files are removed on the next startup.
    A ready window is immutable and never advances while rollouts are using it.
    """
    import datasets

    config = config or PrepareConfig()
    config.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assets = directory / "assets"
    assets.mkdir(exist_ok=True)
    # OS lock releases after a crash; unlike a lockfile, it does not block resume.
    import fcntl

    with (directory / ".prepare.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another writer is preparing this directory") from error
        with closing(sqlite3.connect(directory / "catalog.sqlite")) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pages(language TEXT, page_id TEXT, PRIMARY KEY(language,page_id));
                CREATE TABLE IF NOT EXISTS assets(sha TEXT PRIMARY KEY, mime TEXT NOT NULL, bytes INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, split TEXT NOT NULL, position INTEGER,
                    payload TEXT NOT NULL, UNIQUE(split,position));
                CREATE INDEX IF NOT EXISTS task_split ON tasks(split,position);
            """)
            settings = {
                "schema_version": SCHEMA_VERSION,
                "datasets_version": datasets.__version__,
                "config": asdict(config),
            }
            stored = db.execute(
                "SELECT value FROM meta WHERE key='settings'"
            ).fetchone()
            if stored and stored[0] != canonical_json(settings):
                raise ValueError(
                    "Resume settings differ. Use the original config or a new output directory"
                )
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO meta VALUES('settings', ?)",
                    (canonical_json(settings),),
                )
            if (directory / "manifest.json").exists():
                return json.loads((directory / "manifest.json").read_text())
            committed = {row[0] for row in db.execute("SELECT sha FROM assets")}
            for path in assets.iterdir():
                if path.is_file() and path.name not in committed:
                    path.unlink()
            started = time.monotonic()
            used = db.execute("SELECT COALESCE(SUM(bytes),0) FROM assets").fetchone()[0]
            for language in config.languages:
                key = "progress:" + language
                saved = db.execute(
                    "SELECT value FROM meta WHERE key=?", (key,)
                ).fetchone()
                progress = (
                    json.loads(saved[0])
                    if saved
                    else {"pages": 0, "state": None, "skipped": {}, "exhausted": False}
                )
                if (
                    progress["pages"] >= config.pages_per_language
                    or progress["exhausted"]
                ):
                    continue
                stream = source_factory(config, language)
                if progress["state"] is not None:
                    stream.load_state_dict(progress["state"])
                iterator = iter(stream)
                while progress["pages"] < config.pages_per_language:
                    try:
                        row = next(iterator)
                    except StopIteration:
                        progress["exhausted"] = True
                        with db:
                            db.execute(
                                "INSERT OR REPLACE INTO meta VALUES(?,?)",
                                (key, canonical_json(progress)),
                            )
                        break
                    tasks, skipped = derive_tasks(
                        row,
                        language,
                        config.revision,
                        config.split_seed,
                        config.max_pixels,
                    )
                    new_assets = {}
                    for task in tasks:
                        media = task.pop("media")
                        sha = hashlib.sha256(media).hexdigest()
                        task["asset_sha256"] = sha
                        if sha not in committed:
                            new_assets[sha] = (media, task["mime"])
                    needed = sum(len(media) for media, _ in new_assets.values())
                    if used + needed > config.max_media_bytes:
                        raise ValueError(
                            "Media byte limit reached; no partial page committed. "
                            "Prepare fewer pages in a new directory or use a larger budget there"
                        )
                    for sha, (media, _) in new_assets.items():
                        temp = assets / (sha + ".tmp")
                        temp.write_bytes(media)
                        os.replace(temp, assets / sha)
                    progress = {
                        "pages": progress["pages"] + 1,
                        "state": stream.state_dict(),
                        "skipped": dict(Counter(progress["skipped"]) + skipped),
                        "exhausted": False,
                    }
                    with db:
                        db.execute(
                            "INSERT INTO pages VALUES(?,?)",
                            (language, row["image_id.txt"]),
                        )
                        db.executemany(
                            "INSERT INTO assets VALUES(?,?,?)",
                            [
                                (sha, mime, len(media))
                                for sha, (media, mime) in new_assets.items()
                            ],
                        )
                        db.executemany(
                            "INSERT INTO tasks(id,split,payload) VALUES(?,?,?)",
                            [
                                (t["task_id"], t["split"], canonical_json(t))
                                for t in tasks
                            ],
                        )
                        db.execute(
                            "INSERT OR REPLACE INTO meta VALUES(?,?)",
                            (key, canonical_json(progress)),
                        )
                    committed.update(new_assets)
                    used += needed
                    print(
                        f"{language}: {progress['pages']}/{config.pages_per_language} pages; {used} media bytes",
                        flush=True,
                    )

            # Stable ordering is independent of language iteration and source row ordering.
            with db:
                for split in ("train", "validation", "test"):
                    ids = [
                        r[0]
                        for r in db.execute(
                            "SELECT id FROM tasks WHERE split=? ORDER BY id", (split,)
                        )
                    ]
                    db.executemany(
                        "UPDATE tasks SET position=? WHERE id=?", enumerate(ids)
                    )
            counts = [
                {"split": s, "language": lang, "family": family, "tasks": count}
                for s, lang, family, count in db.execute("""
                        SELECT split,json_extract(payload,'$.language'),json_extract(payload,'$.family'),COUNT(*)
                        FROM tasks GROUP BY 1,2,3 ORDER BY 1,2,3""")
            ]
            identity = hashlib.sha256(canonical_json(settings).encode())
            for (payload,) in db.execute("SELECT payload FROM tasks ORDER BY id"):
                identity.update(payload.encode())
            manifest = {
                **settings,
                "status": "ready",
                "snapshot_id": identity.hexdigest(),
                "source_license": "cc-by-nc-4.0"
                if config.source == REPO_ID
                else "synthetic-fixture",
                "media_bytes": used,
                "counts": counts,
                "pages": dict(
                    db.execute("SELECT language,COUNT(*) FROM pages GROUP BY language")
                ),
                "progress": {
                    k.removeprefix("progress:"): json.loads(v)
                    for k, v in db.execute(
                        "SELECT key,value FROM meta WHERE key LIKE 'progress:%'"
                    )
                },
                "last_prepare_seconds": round(time.monotonic() - started, 3),
            }
            _json_write(directory / "manifest.json", manifest)
            return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(PrepareConfig.languages))
    parser.add_argument("--pages-per-language", type=int, default=32)
    parser.add_argument("--max-media-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--max-pixels", type=int, default=50_000_000)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = vars(parser.parse_args())
    output = args.pop("output")
    args["languages"] = tuple(args["languages"])
    print(
        json.dumps(prepare(output, PrepareConfig(**args)), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
