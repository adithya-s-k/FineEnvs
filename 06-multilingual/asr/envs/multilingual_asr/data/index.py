"""Build per-language indexes by projecting metadata, never the audio column.

The whole corpus is addressable without copying it: each language gets a SQLite index of
every eligible utterance and where it physically lives, and audio is fetched on demand.
Indexing reads metadata columns only - 0.3 MB against 309.6 MB for one measured shard -
so all 102 languages can be indexed without moving the 878 GB the bucket holds.
"""

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .schema import (
    BUCKET_ID,
    DEFAULT_REVISION,
    FAMILIES,
    REPO_ID,
    SPLITS,
    canonical_json,
    recording_id,
    task_id,
)
from .tasks import MAX_SECONDS, MIN_SECONDS, SAMPLING_RATE

INDEX_VERSION = 1
META_COLUMNS = [
    "id",
    "path",
    "num_samples",
    "transcription",
    "raw_transcription",
    "language",
]

DDL = """
CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, size INTEGER);
CREATE TABLE IF NOT EXISTS utterances(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording TEXT NOT NULL, sample_id INTEGER NOT NULL, split TEXT NOT NULL,
    file_id INTEGER NOT NULL, row_group INTEGER NOT NULL, row_in_group INTEGER NOT NULL,
    num_samples INTEGER NOT NULL, language_name TEXT,
    transcription TEXT, raw_transcription TEXT,
    UNIQUE(split, recording));
CREATE TABLE IF NOT EXISTS tasks(
    id TEXT PRIMARY KEY, split TEXT NOT NULL, family TEXT NOT NULL,
    utterance INTEGER NOT NULL, position INTEGER, family_position INTEGER,
    UNIQUE(split, position));
CREATE INDEX IF NOT EXISTS task_split ON tasks(split, position);
CREATE INDEX IF NOT EXISTS task_family ON tasks(split, family, family_position);
"""


def shard_paths(api, language, source_root=None):
    """List a language's shards, from a local copy when one is attached.

    A synced or mounted corpus should index without reaching the Hub at all; going out to
    list files an operator already has on disk is both slower and a needless dependency.
    """
    prefix = f"parquet-data/{language}/"
    if source_root:
        directory = Path(source_root) / prefix
        return sorted(
            (f"{prefix}{path.name}", path.stat().st_size)
            for path in directory.glob("*.parquet")
        )
    return sorted(
        (entry.path, getattr(entry, "size", 0) or 0)
        for entry in api.list_bucket_tree(BUCKET_ID, prefix, recursive=True)
        if entry.path.endswith(".parquet")
    )


def read_shard_metadata(path, size, *, source_root=None, headers=None):
    """Read one shard's metadata columns and its row-group layout."""
    import pyarrow.parquet as pq

    from .cache import RangeReader

    def scan(handle):
        parquet = pq.ParquetFile(handle)
        rows = []
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=META_COLUMNS)
            for offset, row in enumerate(table.to_pylist()):
                rows.append((group, offset, row))
        return rows

    if source_root:
        with (Path(source_root) / path).open("rb") as handle:
            return scan(handle)
    url = f"https://huggingface.co/buckets/{BUCKET_ID}/resolve/{path}"
    with RangeReader(url, size, headers=headers or {}) as handle:
        return scan(handle)


def index_language(language, directory, *, source_root=None, headers=None, api=None):
    """Write one language's index; returns its summary."""
    from huggingface_hub import HfApi

    api = api if api is not None else (None if source_root else HfApi())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{language}.sqlite"
    if path.exists():
        path.unlink()
    counts, skipped, files = {}, {}, {}
    with closing(sqlite3.connect(path)) as db:
        db.executescript(DDL)
        with db:
            for shard, size in shard_paths(api, language, source_root):
                split = shard.rsplit("/", 1)[-1].split("-")[0]
                if split not in SPLITS:
                    skipped["unknown_split"] = skipped.get("unknown_split", 0) + 1
                    continue
                cursor = db.execute(
                    "INSERT OR IGNORE INTO files(path,size) VALUES(?,?)", (shard, size)
                )
                file_id = (
                    cursor.lastrowid
                    or db.execute(
                        "SELECT id FROM files WHERE path=?", (shard,)
                    ).fetchone()[0]
                )
                files[shard] = size
                for group, offset, row in read_shard_metadata(
                    shard, size, source_root=source_root, headers=headers
                ):
                    duration = (row.get("num_samples") or 0) / SAMPLING_RATE
                    try:
                        recording = recording_id(row.get("path"))
                    except ValueError:
                        skipped["missing_path"] = skipped.get("missing_path", 0) + 1
                        continue
                    if row.get("id") is None:
                        skipped["missing_id"] = skipped.get("missing_id", 0) + 1
                        continue
                    if not MIN_SECONDS <= duration <= MAX_SECONDS:
                        skipped["duration_outside_policy"] = (
                            skipped.get("duration_outside_policy", 0) + 1
                        )
                        continue
                    db.execute(
                        """INSERT OR IGNORE INTO utterances(recording,sample_id,split,
                           file_id,row_group,row_in_group,num_samples,language_name,
                           transcription,raw_transcription)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            recording,
                            row["id"],
                            split,
                            file_id,
                            group,
                            offset,
                            row.get("num_samples") or 0,
                            row.get("language"),
                            row.get("transcription"),
                            row.get("raw_transcription"),
                        ),
                    )
                    # lastrowid is meaningless after an ignored insert - it reports the
                    # previous one - so read the id back. Trusting it dropped utterances
                    # that a second shard repeated, and could have attached their tasks
                    # to whichever row happened to be inserted last.
                    utterance = db.execute(
                        "SELECT id FROM utterances WHERE split=? AND recording=?",
                        (split, recording),
                    ).fetchone()[0]
                    for family in FAMILIES:
                        reference = (
                            language
                            if family == "language_id"
                            else row.get(
                                "transcription"
                                if family == "transcription"
                                else "raw_transcription"
                            )
                        )
                        if not isinstance(reference, str) or not reference.strip():
                            key = f"empty_{family}_reference"
                            skipped[key] = skipped.get(key, 0) + 1
                            continue
                        db.execute(
                            "INSERT OR IGNORE INTO tasks(id,split,family,utterance) VALUES(?,?,?,?)",
                            (
                                task_id(
                                    DEFAULT_REVISION,
                                    language,
                                    split,
                                    recording,
                                    family,
                                ),
                                split,
                                family,
                                utterance,
                            ),
                        )
                        # Counted from the table below, never from attempts.
        # Positions are assigned last so they are stable regardless of arrival order.
        with db:
            for split in SPLITS:
                ids = [
                    r[0]
                    for r in db.execute(
                        "SELECT id FROM tasks WHERE split=? ORDER BY id", (split,)
                    )
                ]
                db.executemany("UPDATE tasks SET position=? WHERE id=?", enumerate(ids))
                for family in FAMILIES:
                    ids = [
                        r[0]
                        for r in db.execute(
                            "SELECT id FROM tasks WHERE split=? AND family=? ORDER BY id",
                            (split, family),
                        )
                    ]
                    db.executemany(
                        "UPDATE tasks SET family_position=? WHERE id=?", enumerate(ids)
                    )
        # Counts come from what the table actually holds: an INSERT OR IGNORE that
        # discarded a duplicate must not be promised to a caller as a task, or an index
        # near the reported end resolves to nothing.
        for split, family, total in db.execute(
            "SELECT split, family, COUNT(*) FROM tasks GROUP BY split, family"
        ):
            counts[split, family] = total
        db.execute("VACUUM")
    return {
        "language": language,
        "path": f"{language}.sqlite",
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "counts": [
            {"split": s, "family": f, "tasks": n}
            for (s, f), n in sorted(counts.items())
        ],
        "skipped": skipped,
        "files": files,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(directory, languages=None, *, source_root=None, workers=8):
    from huggingface_hub import HfApi, get_token

    started = time.monotonic()
    api = None if source_root else HfApi()
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not languages and source_root:
        languages = sorted(
            path.name
            for path in (Path(source_root) / "parquet-data").iterdir()
            if path.is_dir() and path.name != "all"
        )
    if not languages:
        languages = sorted(
            {
                entry.path.split("/")[1]
                for entry in api.list_bucket_tree(
                    BUCKET_ID, "parquet-data/", recursive=False
                )
                if entry.path.count("/") >= 1
                and entry.path.split("/")[1] not in ("", "all")
            }
        )
    directory = Path(directory)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                index_language,
                language,
                directory,
                source_root=source_root,
                headers=headers,
                api=api,
            ): language
            for language in languages
        }
        for future in concurrent.futures.as_completed(futures):
            language = futures[future]
            results[language] = future.result()
            tasks = sum(c["tasks"] for c in results[language]["counts"])
            print(f"  {language}: {tasks} tasks", flush=True)

    source_files = sorted(
        {
            path: size for r in results.values() for path, size in r["files"].items()
        }.items()
    )
    inventory = {
        "source": REPO_ID,
        "revision": DEFAULT_REVISION,
        "bucket_id": BUCKET_ID,
        "files": [{"path": p, "size": s} for p, s in source_files],
    }
    inventory_id = hashlib.sha256(canonical_json(inventory).encode()).hexdigest()
    indexes = {
        language: {k: r[k] for k in ("path", "size", "sha256")}
        for language, r in sorted(results.items())
    }
    identity = {
        "index_version": INDEX_VERSION,
        "inventory_id": inventory_id,
        "indexes": {lang: info["sha256"] for lang, info in indexes.items()},
    }
    manifest = {
        "status": "ready",
        "storage": "bucket-parquet",
        "index_version": INDEX_VERSION,
        "snapshot_id": hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
        "inventory_id": inventory_id,
        "source_license": "cc-by-4.0",
        "config": {
            "source": REPO_ID,
            "revision": DEFAULT_REVISION,
            "languages": sorted(results),
            "families": list(FAMILIES),
        },
        "bucket_id": BUCKET_ID,
        "counts": [
            {"language": lang, **c}
            for lang, r in sorted(results.items())
            for c in r["counts"]
        ],
        "skipped": {
            lang: r["skipped"] for lang, r in sorted(results.items()) if r["skipped"]
        },
        "indexes": indexes,
        "source_files": [{"path": p, "size": s} for p, s in source_files],
        "index_seconds": round(time.monotonic() - started, 3),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="*")
    parser.add_argument("--source-root")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = build_index(
        args.output, args.languages, source_root=args.source_root, workers=args.workers
    )
    total = sum(c["tasks"] for c in manifest["counts"])
    print(
        json.dumps(
            {
                "snapshot_id": manifest["snapshot_id"],
                "languages": len(manifest["config"]["languages"]),
                "tasks": total,
                "index_bytes": sum(i["size"] for i in manifest["indexes"].values()),
                "seconds": manifest["index_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
