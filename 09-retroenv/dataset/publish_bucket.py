#!/usr/bin/env python3
"""Publish the serving bucket: a pinned copy of the LiteFold/RetroEnv release plus its index.

    uv run python -m dataset.publish_bucket --dry-run      # check the release and plan the upload
    uv run python -m dataset.publish_bucket                # FineEnvs/retroenv-bucket

``LiteFold/RetroEnv`` stays the canonical dataset. The bucket holds what a server needs to
serve it, under these paths:

    <release paths>                               LiteFold/RetroEnv at a pinned revision; runs/ left out
    openenv/indexes/<snapshot_id>/serving.sqlite.gz  tasks, stock and precedents, precomputed
    openenv/evalsets/<name>.json                  frozen evaluation sets (final_eval)
    openenv/indexes/<snapshot_id>/manifest.json   what a server fetches and checks; published last

Release files are copied server-side by their Xet hash, so nothing large is re-uploaded; the
few small text files without one come from the local copy, which must match the pinned
revision. The same manifest is written to ``envs/retro_route/openenv/corpus-manifest.json``:
committing it pins what every server, local or on a Space, serves.

The index is stored gzipped (196 MB becomes 82 MB); prepare.py unpacks it and checks both
digests. A rerun skips whatever the bucket already holds at the right size, so an upload cut
short resumes. On a slow or lossy uplink, fewer parallel connections and more retries help:

    HF_XET_FIXED_UPLOAD_CONCURRENCY=4 HF_XET_CLIENT_RETRY_MAX_ATTEMPTS=20 \\
        uv run python -m dataset.publish_bucket
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from retroenv.evalsets import load_evalset
from retroenv.serving import INDEX_FILE, build_index, describe_index, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BUCKET_ID = "FineEnvs/retroenv-bucket"
SOURCE = "LiteFold/RetroEnv"
LICENSE = "CC-BY-4.0"
MANIFEST_OUT = ROOT / "envs" / "retro_route" / "openenv" / "corpus-manifest.json"
# What a server reads from the release; tasks-private/ is served from the index instead.
SERVED = ("manifest.json", "checksums.json", "library/", "stocks/")


def snapshot_id(index_version: int, source: dict[str, Any], rdkit_version: str, index_sha256: str) -> str:
    body = {"index_version": index_version, "source": source, "rdkit_version": rdkit_version, "index": index_sha256}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_files(api: Any, revision: str) -> list[Any]:
    return sorted(
        (
            f
            for f in api.list_repo_tree(SOURCE, repo_type="dataset", revision=revision, recursive=True)
            if hasattr(f, "size") and not f.path.startswith("runs/") and f.path != ".gitattributes"
        ),
        key=lambda f: f.path,
    )


def check_local_release(files: list[Any], release: Path) -> None:
    """The local copy supplies digests and the small files; it must be the pinned revision."""
    problems = []
    for f in files:
        local = release / f.path
        if not local.exists() or local.stat().st_size != f.size:
            problems.append(f.path)
        elif f.lfs and sha256_file(local) != f.lfs.sha256:
            problems.append(f.path)
    if problems:
        raise SystemExit(f"{release} differs from {SOURCE} at the pinned revision: {problems[:10]}")


def mirror(api: Any, bucket: str, files: list[Any], release: Path) -> None:
    """Copy the release into the bucket, refusing to overwrite anything that differs."""
    target = {f.path: f for f in api.list_bucket_tree(bucket, recursive=True) if getattr(f, "type", None) == "file"}
    clashes = [f.path for f in files if f.path in target and target[f.path].size != f.size]
    if clashes:
        raise SystemExit(f"{bucket} already holds different files at {clashes[:10]}; refusing to overwrite")
    missing = [f for f in files if f.path not in target]
    copies = [("dataset", SOURCE, f.xet_hash, f.path) for f in missing if f.xet_hash]
    adds = [(str(release / f.path), f.path) for f in missing if not f.xet_hash]
    if copies or adds:
        api.batch_bucket_files(bucket, copy=copies or None, add=adds or None)
    print(f"release: {len(copies)} copied server-side, {len(adds)} uploaded, {len(files) - len(missing)} already there")


def entry(path: str, local: Path, destination: str) -> dict[str, Any]:
    return {"path": path, "local": destination, "size": local.stat().st_size, "sha256": sha256_file(local)}


def pack(path: Path) -> Path:
    """Gzip ``path`` reproducibly: no name or timestamp in the header, so the digest is stable."""
    packed = path.with_name(path.name + ".gz")
    with path.open("rb") as source, packed.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as sink:
            shutil.copyfileobj(source, sink, 1 << 20)
    return packed


def upload_missing(api: Any, bucket: str, files: list[tuple[Path, str, int]]) -> None:
    """Upload each (local, bucket path, size) unless the bucket already holds it at that size."""
    present = {f.path: f.size for f in api.get_bucket_paths_info(bucket, [path for _, path, _ in files])}
    todo = [(str(local), path) for local, path, size in files if present.get(path) != size]
    if todo:
        api.batch_bucket_files(bucket, add=todo)
    print(f"snapshot files: {len(todo)} uploaded, {len(files) - len(todo)} already there")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=BUCKET_ID)
    parser.add_argument("--revision", help="LiteFold/RetroEnv revision to pin (default: current main)")
    parser.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "build" / "serving")
    parser.add_argument("--evalset", type=Path, action="append", default=[])
    parser.add_argument("--manifest-out", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--dry-run", action="store_true", help="check and describe; upload nothing")
    args = parser.parse_args(argv)
    evalset_paths = args.evalset or sorted((ROOT / "data").glob("eval-*.json"))

    from huggingface_hub import HfApi

    api = HfApi()
    revision = args.revision or api.dataset_info(SOURCE).sha
    files = source_files(api, revision)
    check_local_release(files, args.release)
    print(f"{SOURCE}@{revision[:12]}: {len(files)} files match {args.release}")

    index_path = args.index_dir / INDEX_FILE
    index = describe_index(index_path) if index_path.exists() else build_index(args.release, args.index_dir)
    if index["release_checksums_sha256"] != sha256_file(args.release / "checksums.json"):
        raise SystemExit(f"{index_path} was built from another release; rebuild it")
    source = {
        "repo": SOURCE,
        "revision": revision,
        "license": LICENSE,
        "checksums_sha256": index["release_checksums_sha256"],
    }
    snapshot = snapshot_id(index["index_version"], source, index["rdkit_version"], index["sha256"])
    prefix = f"openenv/indexes/{snapshot}"

    served = [f for f in files if f.path.startswith(SERVED)]
    manifest_files = [entry(f.path, args.release / f.path, f.path) for f in served]
    packed = pack(index_path)
    manifest_files.append(
        {
            **entry(f"{prefix}/{INDEX_FILE}.gz", packed, f"serving/{INDEX_FILE}"),
            "unpacked": {"size": index["size"], "sha256": index["sha256"]},
        }
    )
    evalsets = {}
    for path in evalset_paths:
        record = load_evalset(path)
        evalsets[record["name"]] = {
            **entry(f"openenv/evalsets/{record['name']}.json", path, f"evalsets/{record['name']}.json"),
            "evalset_id": record["evalset_id"],
            "tasks": record["size"],
        }
    manifest = {
        "status": "ready",
        "storage": "bucket-sqlite",
        "index_version": index["index_version"],
        "snapshot_id": snapshot,
        "bucket_id": args.bucket,
        "source": source,
        "rdkit_version": index["rdkit_version"],
        "files": manifest_files,
        "evalsets": evalsets,
        "splits": index["splits"],
        "stocks": index["stocks"],
        "precedents": index["precedents"],
    }
    total = sum(f["size"] for f in manifest_files) + sum(e["size"] for e in evalsets.values())
    print(f"snapshot {snapshot[:12]}: {len(manifest_files)} files and {len(evalsets)} evalset(s), {total / 1e6:.0f} MB")
    if args.dry_run:
        print(json.dumps({k: manifest[k] for k in ("splits", "evalsets", "stocks", "precedents")}, indent=1))
        return 0

    api.create_bucket(args.bucket, private=False, exist_ok=True)
    mirror(api, args.bucket, files, args.release)
    uploads = [(packed, f"{prefix}/{INDEX_FILE}.gz", packed.stat().st_size)]
    uploads += [
        (path, f"openenv/evalsets/{load_evalset(path)['name']}.json", path.stat().st_size) for path in evalset_paths
    ]
    upload_missing(api, args.bucket, uploads)
    # Verify everything the manifest names before publishing it: a server trusts the manifest.
    expected = {f["path"]: f["size"] for f in manifest_files} | {e["path"]: e["size"] for e in evalsets.values()}
    found = {f.path: f.size for f in api.get_bucket_paths_info(args.bucket, list(expected))}
    wrong = sorted(path for path, size in expected.items() if found.get(path) != size)
    if wrong:
        raise SystemExit(f"{args.bucket} is missing or has wrong sizes for {wrong[:10]}")
    text = json.dumps(manifest, indent=2) + "\n"
    api.batch_bucket_files(args.bucket, add=[(text.encode(), f"{prefix}/manifest.json")])
    args.manifest_out.write_text(text)
    print(f"published https://huggingface.co/buckets/{args.bucket}  ({prefix}/manifest.json)")
    print(f"pinned in {args.manifest_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
