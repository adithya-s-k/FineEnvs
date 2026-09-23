"""Publish the ASR server and attach its evaluation snapshot from the bucket.

The environment serves a prepared snapshot rather than an indexed corpus, so a deployment
has to put one somewhere the Space can read. The snapshot is uploaded into the same bucket
that holds the source and mounted read only, which keeps the Space image code-only and
lets several Spaces share one published snapshot.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume
from multilingual_asr.data.catalog import Catalog
from multilingual_asr.data.evalset import load as load_evalset
from multilingual_asr.data.schema import BUCKET_ID


def resolve_bucket_id(bucket_id):
    """Return the bucket's current canonical name.

    An organization rename leaves HTTP reads working through a redirect but does not move
    a Space's volume mount, which then fails every read with EIO. This cost the OCR
    environment a silently broken deployment; resolve the live name for the mount.
    """
    import requests
    from huggingface_hub import get_token

    token = get_token()
    response = requests.get(
        f"https://huggingface.co/api/buckets/{bucket_id}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=60,
    )
    response.raise_for_status()
    current = response.json().get("id")
    if not current:
        raise ValueError(f"Could not resolve the current name of bucket {bucket_id!r}")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--evalset", type=Path, help="Frozen set this snapshot serves")
    parser.add_argument("--bucket-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Reuse an already published snapshot of the same id",
    )
    args = parser.parse_args()

    catalog = Catalog(args.snapshot)
    api = HfApi()
    bucket = args.bucket_id or resolve_bucket_id(BUCKET_ID)
    prefix = f"openenv/snapshots/{catalog.snapshot_id}"

    if args.evalset:
        frozen = load_evalset(args.evalset)
        served = {
            catalog.at(s, i)["task_id"]
            for s in ("train", "validation", "test")
            for i in range(catalog.count(s))
        }
        missing = [e["task_id"] for e in frozen["tasks"] if e["task_id"] not in served]
        if missing:
            raise ValueError(
                f"Snapshot is missing {len(missing)} of {frozen['size']} "
                f"{frozen['name']} tasks; rebuild it with asr-prepare --evalset"
            )
        print(
            f"snapshot serves all {frozen['size']} tasks of {frozen['name']}",
            flush=True,
        )

    if not args.skip_upload:
        print(f"uploading snapshot to {bucket}/{prefix}", flush=True)
        api.upload_folder(
            repo_id=bucket,
            repo_type="bucket",
            folder_path=str(args.snapshot),
            path_in_repo=prefix,
            commit_message=f"ASR evaluation snapshot {catalog.snapshot_id[:12]}",
        )

    project = Path(__file__).resolve().parents[1] / "envs" / "multilingual_asr"
    with tempfile.TemporaryDirectory(prefix="asr-space-") as directory:
        staging = Path(directory)
        shutil.copytree(
            project,
            staging,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.egg-info",
                "tests",
                "snapshot",
            ),
        )
        counts = catalog.manifest["counts"]
        with (staging / "README.md").open("a") as card:
            card.write("\n## Served snapshot\n\n")
            card.write(
                f"{sum(c['tasks'] for c in counts):,} tasks across "
                f"{len(catalog.languages)} FLEURS languages.\n\n"
            )
            card.write(f"Snapshot: `{catalog.snapshot_id}`.\n")
        api.create_repo(
            args.space_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message=f"Serve FLEURS ASR snapshot {catalog.snapshot_id[:12]}",
        )

    runtime = api.space_info(args.space_id).runtime
    volumes = [v for v in (runtime.volumes or []) if v.mount_path != "/snapshot"]
    volumes.append(
        Volume(type="bucket", source=bucket, mount_path="/snapshot", read_only=True)
    )
    if [v.to_dict() for v in volumes] != [v.to_dict() for v in (runtime.volumes or [])]:
        api.set_space_volumes(args.space_id, volumes=volumes)

    current = api.get_space_variables(args.space_id)
    for key, value in {
        "ASR_SNAPSHOT": f"/snapshot/{prefix}",
        "ASR_MAX_SESSIONS": str(args.sessions),
    }.items():
        if key not in current or current[key].value != value:
            api.add_space_variable(args.space_id, key, value)

    result = {
        "space_id": args.space_id,
        "commit": commit.oid,
        "snapshot_id": catalog.snapshot_id,
        "bucket_id": bucket,
        "snapshot_path": f"/snapshot/{prefix}",
        "languages": len(catalog.languages),
        "tasks": sum(c["tasks"] for c in catalog.manifest["counts"]),
        "sessions": args.sessions,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
