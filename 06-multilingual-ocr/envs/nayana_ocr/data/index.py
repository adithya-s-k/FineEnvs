"""Build complete language indexes by projecting annotations, never JPEG columns."""

import argparse
import hashlib
import json
import os
import sqlite3
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from .schema import LANGUAGES, SCHEMA_VERSION, canonical_json
from .tasks import derive_tasks

INDEX_VERSION = 2
META_COLUMNS = ["image_id.txt", "regions.json", "vqa.json"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _read_metadata(spec, bucket_id, source_root=None):
    """One shard at a time; Arrow coalesces adjacent selected column ranges."""
    fs = HfFileSystem()
    handle = (
        (Path(source_root) / spec["path"]).open("rb")
        if source_root
        else fs.open(f"buckets/{bucket_id}/{spec['path']}", "rb", cache_type="none")
    )
    with handle:
        parquet = pq.ParquetFile(handle, pre_buffer=True)
        metadata = parquet.metadata
        rows = parquet.read(columns=META_COLUMNS, use_threads=False).to_pylist()
        groups, offset = [], 0
        for index in range(metadata.num_row_groups):
            group = metadata.row_group(index)
            image_bytes = sum(
                group.column(c).total_compressed_size
                for c in range(group.num_columns)
                if group.column(c).path_in_schema.startswith("jpg.")
            )
            groups.append(
                {
                    "row_group": index,
                    "rows": group.num_rows,
                    "row_start": offset,
                    "image_bytes": image_bytes,
                }
            )
            offset += group.num_rows
        if len(rows) != metadata.num_rows or offset != len(rows):
            raise ValueError(f"Incomplete metadata projection: {spec['path']}")
        return rows, groups


def _reuse_metadata(directory, language, spec):
    """Read verified original annotations from an older index, never its derived tasks."""
    with closing(
        sqlite3.connect(
            (Path(directory) / f"{language}.sqlite").resolve().as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        saved = db.execute(
            "SELECT id,xet_hash,size,rows FROM files WHERE path=?", (spec["path"],)
        ).fetchone()
        if saved is None or saved[1:3] != (spec["xet_hash"], spec["size"]):
            raise ValueError(f"Reusable index source differs: {spec['path']}")
        rows, groups = [], []
        for block, group, count, size in db.execute(
            "SELECT id,row_group,rows,image_bytes FROM blocks WHERE file_id=? ORDER BY row_group",
            (saved[0],),
        ):
            groups.append(
                dict(row_group=group, rows=count, row_start=len(rows), image_bytes=size)
            )
            payloads = db.execute(
                "SELECT payload FROM pages WHERE block_id=? ORDER BY row_in_group",
                (block,),
            ).fetchall()
            if len(payloads) != count:
                raise ValueError("Incomplete reusable annotation block")
            rows.extend(json.loads(zlib.decompress(p)) for (p,) in payloads)
        if len(rows) != saved[3]:
            raise ValueError("Incomplete reusable annotation shard")
        return rows, groups


def build_language(
    inventory, language, output, source_root=None, split_seed=42, reuse_index=None
):
    path = output / f"{language}.sqlite"
    marker = output / f"{language}.json"
    settings = {
        "index_version": INDEX_VERSION,
        "inventory_id": inventory["inventory_id"],
        "language": language,
        "split_seed": split_seed,
        "pyarrow": pa.__version__,
    }
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["settings"] != settings or sha256_file(path) != result["sha256"]:
            raise ValueError(f"Completed index settings/content changed: {language}")
        return result
    files = [
        f
        for f in inventory["files"]
        if f["path"].startswith(language + "/") and f["path"].endswith(".parquet")
    ]
    if not files:
        raise ValueError(f"No source shards for {language}")
    if reuse_index:
        previous = json.loads((Path(reuse_index) / "manifest.json").read_text())
        info = previous["indexes"][language]
        if (
            previous["inventory_id"] != inventory["inventory_id"]
            or info["path"] != f"{language}.sqlite"
            or sha256_file(Path(reuse_index) / info["path"]) != info["sha256"]
        ):
            raise ValueError("Reusable index identity/content differs")
    started = time.monotonic()
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT UNIQUE,
                xet_hash TEXT,size INTEGER,rows INTEGER);
            CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,file_id INTEGER,
                row_group INTEGER,rows INTEGER,image_bytes INTEGER, UNIQUE(file_id,row_group));
            CREATE TABLE IF NOT EXISTS pages(id INTEGER PRIMARY KEY,block_id INTEGER,row_in_group INTEGER,
                page_id TEXT UNIQUE,document_id TEXT,payload BLOB);
            CREATE TABLE IF NOT EXISTS tasks(split TEXT,position INTEGER,family TEXT,
                family_position INTEGER,page_id INTEGER,unit TEXT,block_id INTEGER,
                PRIMARY KEY(split,position)) WITHOUT ROWID;
            CREATE UNIQUE INDEX IF NOT EXISTS task_family ON tasks(split,family,family_position);
            CREATE INDEX IF NOT EXISTS task_block ON tasks(block_id,split,family,position);
        """)
        saved = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        if saved and saved[0] != canonical_json(settings):
            raise ValueError("Resume settings differ; use a new index directory")
        with db:
            db.execute(
                "INSERT OR IGNORE INTO meta VALUES('settings',?)",
                (canonical_json(settings),),
            )
        done = {row[0] for row in db.execute("SELECT path FROM files")}
        positions = Counter(
            dict(db.execute("SELECT split,COUNT(*) FROM tasks GROUP BY split"))
        )
        family_positions = Counter(
            {
                (s, f): n
                for s, f, n in db.execute(
                    "SELECT split,family,COUNT(*) FROM tasks GROUP BY split,family"
                )
            }
        )
        saved = db.execute("SELECT value FROM meta WHERE key='skipped'").fetchone()
        skipped = Counter(json.loads(saved[0])) if saved else Counter()
        for file_id, spec in enumerate(files):
            if spec["path"] in done:
                continue
            rows, groups = (
                _reuse_metadata(reuse_index, language, spec)
                if reuse_index
                else _read_metadata(spec, inventory["bucket_id"], source_root)
            )
            with db:  # A complete source shard and its resume cursor commit together.
                db.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?)",
                    (file_id, spec["path"], spec["xet_hash"], spec["size"], len(rows)),
                )
                for group in groups:
                    block_id = db.execute(
                        "INSERT INTO blocks(file_id,row_group,rows,image_bytes) VALUES(?,?,?,?) RETURNING id",
                        (
                            file_id,
                            group["row_group"],
                            group["rows"],
                            group["image_bytes"],
                        ),
                    ).fetchone()[0]
                    for offset in range(group["rows"]):
                        row = rows[group["row_start"] + offset]
                        tasks, excluded = derive_tasks(
                            row,
                            language,
                            inventory["revision"],
                            split_seed,
                            metadata_only=True,
                        )
                        skipped.update(excluded)
                        # Keep all original annotations for future tasks. JPEGs remain in the bucket.
                        page_id = db.execute(
                            "INSERT INTO pages(block_id,row_in_group,page_id,document_id,payload) VALUES(?,?,?,?,?) RETURNING id",
                            (
                                block_id,
                                offset,
                                row["image_id.txt"],
                                row["image_id.txt"].rsplit("_page_", 1)[0],
                                zlib.compress(canonical_json(row).encode(), level=3),
                            ),
                        ).fetchone()[0]
                        values = []
                        for task in tasks:
                            split, family = task["split"], task["family"]
                            values.append(
                                (
                                    split,
                                    positions[split],
                                    family,
                                    family_positions[split, family],
                                    page_id,
                                    task["unit"],
                                    block_id,
                                )
                            )
                            positions[split] += 1
                            family_positions[split, family] += 1
                        db.executemany(
                            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?)", values
                        )
                db.execute(
                    "INSERT OR REPLACE INTO meta VALUES('skipped',?)",
                    (canonical_json(skipped),),
                )
            print(
                f"{language}: shard {file_id + 1}/{len(files)}, {sum(positions.values()):,} indexed tasks",
                flush=True,
            )
        counts = [
            {"split": split, "language": language, "family": family, "tasks": count}
            for (split, family), count in sorted(family_positions.items())
        ]
        pages = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        expected = db.execute("SELECT SUM(rows) FROM files").fetchone()[0]
        if pages != expected:
            raise ValueError(f"Lost rows while indexing {language}: {pages}/{expected}")
        blocks = db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("VACUUM")
    result = {
        "settings": settings,
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "pages": pages,
        "blocks": blocks,
        "counts": counts,
        "skipped": dict(skipped),
        "build_seconds": round(time.monotonic() - started, 3),
    }
    write_json(marker, result)
    return result


def build_index(
    inventory,
    output,
    *,
    languages=LANGUAGES,
    workers=4,
    source_root=None,
    split_seed=42,
    reuse_index=None,
):
    import fcntl

    if workers < 1 or not languages or len(set(languages)) != len(languages):
        raise ValueError("Use positive workers and distinct languages")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".index.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                lang: executor.submit(
                    build_language,
                    inventory,
                    lang,
                    output,
                    source_root,
                    split_seed,
                    reuse_index,
                )
                for lang in sorted(languages)
            }
            indexes = {lang: future.result() for lang, future in futures.items()}
        identity = {
            "index_version": INDEX_VERSION,
            "inventory_id": inventory["inventory_id"],
            "split_seed": split_seed,
            "indexes": {lang: value["sha256"] for lang, value in indexes.items()},
        }
        snapshot_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        manifest = {
            "status": "ready",
            "storage": "bucket-parquet",
            "index_version": INDEX_VERSION,
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "pyarrow_version": pa.__version__,
            "datasets_version": "not-used-for-indexing",
            "config": {
                "source": inventory["source"],
                "revision": inventory["revision"],
                "languages": sorted(languages),
                "split_seed": split_seed,
            },
            "bucket_id": inventory["bucket_id"],
            "inventory_id": inventory["inventory_id"],
            "source_license": inventory["source_license"],
            "media_bytes": inventory["source_bytes"],
            "pages": {lang: value["pages"] for lang, value in indexes.items()},
            "counts": [row for value in indexes.values() for row in value["counts"]],
            "indexes": indexes,
            "annotation_validation": "metadata indexed; image bounds validated on reset",
            # Keep the complete inventory so readers can verify inventory_id,
            # including offline SHA-256 values and source/bucket provenance.
            "source_files": inventory["files"],
        }
        write_json(output / "manifest.json", manifest)
        return manifest


def publish_index(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    prefix = f"openenv/indexes/{manifest['snapshot_id']}"
    api = HfApi()

    # Content-addressed immutable publication: all databases first, ready manifest last.
    def upload(item):
        lang, info = item
        path = directory / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Index changed before publication: {lang}")
        api.batch_bucket_files(
            manifest["bucket_id"], add=[(path, f"{prefix}/{path.name}")]
        )
        print(f"Published index {lang}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(upload, manifest["indexes"].items()))
    api.batch_bucket_files(
        manifest["bucket_id"],
        add=[(directory / "manifest.json", f"{prefix}/manifest.json")],
    )
    uri = f"hf://buckets/{manifest['bucket_id']}/{prefix}/manifest.json"
    print(uri, flush=True)
    return uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES)
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--reuse-index",
        type=Path,
        help="Reuse hash-verified original annotations from a previous local index",
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    manifest = build_index(
        json.loads(args.inventory.read_text()),
        args.output,
        languages=args.languages,
        workers=args.workers,
        source_root=args.source_root,
        reuse_index=args.reuse_index,
    )
    print(
        json.dumps(
            {
                "snapshot_id": manifest["snapshot_id"],
                "pages": manifest["pages"],
                "tasks": sum(r["tasks"] for r in manifest["counts"]),
            },
            indent=2,
        )
    )
    if args.publish:
        publish_index(args.output)


if __name__ == "__main__":
    main()
