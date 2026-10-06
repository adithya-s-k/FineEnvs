"""Audit/copy a pinned corpus into a bucket, using server-side Xet copies."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .schema import REPO_ID, REVISION, canonical_json

BUCKET_ID = "HuggingEnvs/NayanaOCR_Corpus_2025_bucket"


def mirror(bucket_id=BUCKET_ID, revision=REVISION, *, copy_missing=True):
    api = HfApi()
    api.bucket_info(bucket_id)  # The caller selects an existing bucket.
    source = [
        f
        for f in api.list_repo_tree(
            REPO_ID, repo_type="dataset", revision=revision, recursive=True
        )
        if hasattr(f, "size")
    ]
    target = {
        f.path: f
        for f in api.list_bucket_tree(bucket_id, recursive=True)
        if getattr(f, "type", None) == "file"
    }
    mismatches = [
        f.path
        for f in source
        if f.path in target
        and f.xet_hash
        and (f.xet_hash != target[f.path].xet_hash or f.size != target[f.path].size)
    ]
    if mismatches:
        raise ValueError(
            f"Bucket differs from pinned source; refusing overwrite: {mismatches[:10]}"
        )
    missing = [f for f in source if f.path not in target]
    if missing and not copy_missing:
        raise ValueError(f"Bucket is incomplete: {[f.path for f in missing]}")
    for start in range(0, len(missing), 100):
        chunk = missing[start : start + 100]
        copies = [("dataset", REPO_ID, f.xet_hash, f.path) for f in chunk if f.xet_hash]
        adds = [
            (
                hf_hub_download(
                    REPO_ID, f.path, repo_type="dataset", revision=revision
                ),
                f.path,
            )
            for f in chunk
            if not f.xet_hash
        ]
        api.batch_bucket_files(bucket_id, copy=copies or None, add=adds or None)
        print(
            f"Copied {min(start + 100, len(missing))}/{len(missing)} missing files",
            flush=True,
        )
    target = {
        f.path: f
        for f in api.list_bucket_tree(bucket_id, recursive=True)
        if getattr(f, "type", None) == "file"
    }
    for f in source:
        actual = target.get(f.path)
        if (
            actual is None
            or actual.size != f.size
            or (f.xet_hash and actual.xet_hash != f.xet_hash)
        ):
            raise ValueError(f"Copy verification failed: {f.path}")
    files = [
        {
            "path": f.path,
            "size": f.size,
            "xet_hash": target[f.path].xet_hash,
            "sha256": f.lfs.sha256 if f.lfs else None,
        }
        for f in sorted(source, key=lambda f: f.path)
    ]
    result = {
        "source": REPO_ID,
        "revision": revision,
        "bucket_id": bucket_id,
        "source_license": "cc-by-nc-4.0",
        "files": files,
        "source_bytes": sum(f["size"] for f in files),
        "parquet_files": sum(f["path"].endswith(".parquet") for f in files),
    }
    result["inventory_id"] = hashlib.sha256(canonical_json(result).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=BUCKET_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = mirror(args.bucket, args.revision, copy_missing=not args.verify_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
