"""Publish the ASR server and attach the FLEURS bucket read only.

The Space runs the same code as a local checkout and reads the same bucket, so what it
serves is what a training or evaluation run sees: all splits, every indexed language, with
audio fetched on demand. Its image carries code and the corpus manifest only.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token
from multilingual_asr.data.evalset import load as load_evalset
from multilingual_asr.data.schema import BUCKET_ID, REPO_ID


def resolve_bucket_id(bucket_id):
    """Return the bucket's current canonical name.

    An organization rename leaves HTTP reads working through a redirect but does not move
    a Space's volume mount, which then fails every read with EIO. That cost the OCR
    environment a silently broken deployment, so the live name is resolved for the mount
    while the recorded provenance is left alone.
    """
    import requests

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
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--evalset", type=Path, action="append", default=[])
    parser.add_argument("--bucket-id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.corpus_manifest.read_text())
    if manifest["config"]["source"] != REPO_ID or manifest.get("status") != "ready":
        parser.error("Publish a finalized FLEURS corpus index")
    api = HfApi()
    bucket = args.bucket_id or resolve_bucket_id(BUCKET_ID)

    # Every index must already be in the bucket: the Space fetches them lazily and would
    # otherwise fail at the first task rather than at deploy time.
    prefix = f"openenv/indexes/{manifest['snapshot_id']}"
    expected = {
        f"{prefix}/{info['path']}": info["size"]
        for info in manifest["indexes"].values()
    }
    available = {
        row.path: row.size for row in api.get_bucket_paths_info(bucket, list(expected))
    }
    missing = {p: s for p, s in expected.items() if available.get(p) != s}
    if missing:
        raise ValueError(
            f"{len(missing)} index file(s) are absent or the wrong size in {bucket}; "
            "publish the index before deploying"
        )
    print(f"verified {len(expected)} index files in {bucket}", flush=True)

    for path in args.evalset:
        frozen = load_evalset(path)
        print(f"evaluation set {frozen['name']}: {frozen['size']} tasks", flush=True)

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
        shutil.copyfile(args.corpus_manifest, staging / "corpus-manifest.json")
        for path in args.evalset:
            shutil.copyfile(path, staging / path.name)
        if not (staging / "README.md").exists():
            raise ValueError(
                "The environment needs a README.md with Space frontmatter; without "
                "sdk: docker the Space is created with no SDK and lands in CONFIG_ERROR"
            )
        tasks = sum(c["tasks"] for c in manifest["counts"])
        with (staging / "README.md").open("a") as card:
            card.write("\n## Served corpus\n\n")
            card.write(
                f"{tasks:,} indexed tasks across {len(manifest['config']['languages'])} "
                "FLEURS languages, in every published split.\n\n"
            )
            card.write(
                f"Snapshot: `{manifest['snapshot_id']}`. Audio stays in the bucket and is "
                "fetched on demand.\n"
            )
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
            commit_message=f"Serve FLEURS corpus {manifest['snapshot_id'][:12]} from the bucket",
        )

    runtime = api.space_info(args.space_id).runtime
    volumes = [v for v in (runtime.volumes or []) if v.mount_path != "/fleurs"]
    volumes.append(
        Volume(type="bucket", source=bucket, mount_path="/fleurs", read_only=True)
    )
    if [v.to_dict() for v in volumes] != [v.to_dict() for v in (runtime.volumes or [])]:
        api.set_space_volumes(args.space_id, volumes=volumes)

    current = api.get_space_variables(args.space_id)
    settings = {
        "FLEURS_CORPUS_MANIFEST": "/app/corpus-manifest.json",
        "FLEURS_SOURCE_ROOT": "/fleurs",
        "FLEURS_CACHE_DIR": "/tmp/fleurs-cache",
        "ASR_MAX_SESSIONS": str(args.sessions),
    }
    for key, value in settings.items():
        if key not in current or current[key].value != value:
            api.add_space_variable(args.space_id, key, value)
    # A stale ASR_SNAPSHOT would win over the corpus manifest and serve the wrong data.
    if "ASR_SNAPSHOT" in current:
        api.delete_space_variable(args.space_id, "ASR_SNAPSHOT")
        print("removed the stale ASR_SNAPSHOT variable", flush=True)

    result = {
        "space_id": args.space_id,
        "commit": commit.oid,
        "snapshot_id": manifest["snapshot_id"],
        "bucket_id": bucket,
        "languages": len(manifest["config"]["languages"]),
        "tasks": tasks,
        "splits": sorted({c["split"] for c in manifest["counts"]}),
        "evalsets": [p.name for p in args.evalset],
        "sessions": args.sessions,
        "bundled": "code, corpus manifest and evaluation sets; indexes and audio fetched lazily",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
