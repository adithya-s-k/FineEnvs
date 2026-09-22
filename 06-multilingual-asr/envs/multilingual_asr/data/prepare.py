"""Build a bounded snapshot. Audio is stored once, content addressed, references sealed."""

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .schema import BUCKET_ID, FAMILIES, REPO_ID, SCHEMA_VERSION, SPLITS, canonical_json
from .tasks import derive_tasks

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets(sha TEXT PRIMARY KEY, mime TEXT NOT NULL, bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, split TEXT NOT NULL, position INTEGER,
    payload TEXT NOT NULL, UNIQUE(split,position));
CREATE INDEX IF NOT EXISTS task_split ON tasks(split,position);
"""


def write_snapshot(
    directory, sources, *, source, revision, languages, families=FAMILIES
):
    """Write a ready snapshot from (language, rows) pairs."""
    started = time.monotonic()
    directory = Path(directory)
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    settings = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "source": source,
            "revision": revision,
            "languages": list(languages),
            "families": list(families),
        },
    }
    media_bytes, skipped, counts = 0, {}, {}
    with closing(sqlite3.connect(directory / "catalog.sqlite")) as db:
        db.executescript(DDL)
        with db:
            db.execute("DELETE FROM tasks")
            db.execute("DELETE FROM assets")
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES('settings',?)",
                (canonical_json(settings),),
            )
            for language, rows in sources:
                for row in rows:
                    tasks, reasons = derive_tasks(
                        row, language, revision, families=families
                    )
                    for key, value in reasons.items():
                        skipped[key] = skipped.get(key, 0) + value
                    for task in tasks:
                        media = task.pop("media")
                        sha = hashlib.sha256(media).hexdigest()
                        path = directory / "assets" / sha
                        if not path.exists():
                            path.write_bytes(media)
                            media_bytes += len(media)
                            db.execute(
                                "INSERT OR REPLACE INTO assets VALUES(?,?,?)",
                                (sha, task["mime"], len(media)),
                            )
                        task["asset_sha256"] = sha
                        db.execute(
                            "INSERT OR REPLACE INTO tasks(id,split,payload) VALUES(?,?,?)",
                            (task["task_id"], task["split"], canonical_json(task)),
                        )
                        key = (task["split"], language, task["family"])
                        counts[key] = counts.get(key, 0) + 1
        # Position is assigned last so it is stable regardless of arrival order.
        with db:
            for split in SPLITS:
                ids = [
                    r[0]
                    for r in db.execute(
                        "SELECT id FROM tasks WHERE split=? ORDER BY id", (split,)
                    )
                ]
                db.executemany("UPDATE tasks SET position=? WHERE id=?", enumerate(ids))
        identity = hashlib.sha256(canonical_json(settings).encode())
        for (payload,) in db.execute("SELECT payload FROM tasks ORDER BY id"):
            identity.update(payload.encode())
    manifest = {
        **settings,
        "status": "ready",
        "snapshot_id": identity.hexdigest(),
        "source_license": "cc-by-4.0" if source == REPO_ID else "synthetic-fixture",
        "bucket_id": BUCKET_ID if source == REPO_ID else None,
        "media_bytes": media_bytes,
        "counts": [
            {"split": s, "language": lang, "family": f, "tasks": n}
            for (s, lang, f), n in sorted(counts.items())
        ],
        "skipped": skipped,
        "prepare_seconds": round(time.monotonic() - started, 3),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def bucket_shards(language, split, api=None):
    """List the published shards for one language/split.

    A bucket's resolve endpoint serves bytes, not directory listings, so shards are
    discovered through the Hub API rather than by guessing the `of-000NN` suffix, which
    differs per language.
    """
    from huggingface_hub import HfApi

    api = api or HfApi()
    prefix = f"parquet-data/{language}/"
    names = [
        entry.path
        for entry in api.list_bucket_tree(BUCKET_ID, prefix, recursive=True)
        if entry.path.endswith(".parquet")
        and entry.path.rsplit("/", 1)[-1].startswith(f"{split}-")
    ]
    if not names:
        raise ValueError(f"No {split} shard for {language} in {BUCKET_ID}")
    return sorted(names)


def bucket_rows(language, split, limit, token=None):
    """Read one FLEURS language/split from the pinned bucket copy.

    Each published file is a single row group, so a read is all-or-nothing: bounding the
    snapshot bounds what is stored, not what is transferred.
    """
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import get_token

    token = token or get_token()
    fs = fsspec.filesystem(
        "http",
        client_kwargs={"headers": {"Authorization": f"Bearer {token}"}}
        if token
        else {},
    )
    produced = 0
    for path in bucket_shards(language, split):
        if produced >= limit:
            return
        url = f"https://huggingface.co/buckets/{BUCKET_ID}/resolve/{path}"
        with fs.open(url) as handle:
            table = pq.ParquetFile(handle).read()
        for row in table.to_pylist():
            if produced >= limit:
                return
            yield {**row, "split": split}
            produced += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--per-split", type=int, default=16)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    def sources():
        for language in args.languages:

            def rows(language=language):
                for split in args.splits:
                    yield from bucket_rows(language, split, args.per_split)

            yield language, rows()

    manifest = write_snapshot(
        args.output,
        sources(),
        source=REPO_ID,
        revision=args.revision,
        languages=args.languages,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "counts"}, indent=2))
    print(f"tasks: {sum(c['tasks'] for c in manifest['counts'])}")


if __name__ == "__main__":
    main()
