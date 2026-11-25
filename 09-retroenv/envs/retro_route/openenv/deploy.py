#!/usr/bin/env python3
"""Stage this folder with ../core into a Docker Space layout, and optionally publish it.

    python deploy.py --stage-only --stage-dir /tmp/retroenv-space   # build locally
    docker build -t retroenv /tmp/retroenv-space
    python deploy.py --repo FineEnvs/retroenv --public              # the FineEnvs Space

The Space runs the same code as a local checkout and serves the same pinned snapshot: the
image carries code and corpus-manifest.json only. The serving bucket named in the manifest
is mounted read-only at /data, and prepare.py copies and checks the snapshot from there at
startup. Every file the manifest names must already be in the bucket, so a Space never
starts against a half-published snapshot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_ROOT = HERE.parent
MANIFEST = HERE / "corpus-manifest.json"
MOUNT = "/data"
IGNORE = shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache", ".venv", ".env", "prepared", "tests")


def stage(target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ENV_ROOT / "core", target / "core", ignore=IGNORE)
    shutil.copytree(HERE, target / "openenv", ignore=IGNORE)
    shutil.copy2(HERE / "Dockerfile", target / "Dockerfile")
    shutil.copy2(HERE / "README.md", target / "README.md")
    shutil.copy2(ENV_ROOT / ".dockerignore", target / ".dockerignore")
    return target


def live_bucket(api, bucket_id: str) -> str:
    """The bucket's current name. An organization rename keeps HTTP reads working through a
    redirect but breaks a Space's volume mount, so the mount uses the live name."""
    return api.bucket_info(bucket_id).id


def check_bucket(api, bucket: str, manifest: dict) -> None:
    entries = [*manifest["files"], *manifest.get("evalsets", {}).values()]
    expected = {entry["path"]: entry["size"] for entry in entries}
    found = {f.path: f.size for f in api.get_bucket_paths_info(bucket, list(expected))}
    missing = sorted(path for path, size in expected.items() if found.get(path) != size)
    if missing:
        raise SystemExit(f"{bucket} lacks or has wrong sizes for {missing[:10]}; publish the snapshot first")
    print(f"verified {len(expected)} snapshot files in {bucket}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="Space id, e.g. FineEnvs/retroenv")
    parser.add_argument("--public", action="store_true", help="create the Space public (default private)")
    parser.add_argument("--concurrency", type=int, default=64, help="WebSocket sessions per process")
    parser.add_argument("--toolset", choices=("full", "unaided"), default="full")
    parser.add_argument("--hardware", help="Space hardware flavor, e.g. cpu-upgrade (default: leave as is)")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "ready":
        raise SystemExit(f"{MANIFEST} is not a finished snapshot; run dataset/publish_bucket.py")
    staged = stage(args.stage_dir or Path(tempfile.mkdtemp(prefix="retroenv-space-")))
    print(f"staged {staged}")
    if args.stage_only:
        return 0
    if not args.repo:
        parser.error("--repo is required to publish")

    from huggingface_hub import HfApi, Volume

    api = HfApi()
    bucket = live_bucket(api, manifest["bucket_id"])
    check_bucket(api, bucket, manifest)
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=staged,
        commit_message=f"Serve RetroEnv snapshot {manifest['snapshot_id'][:12]}",
    )

    runtime = api.space_info(args.repo).runtime
    current = list(runtime.volumes or []) if runtime else []
    volumes = [v for v in current if v.mount_path != MOUNT]
    volumes.append(Volume(type="bucket", source=bucket, mount_path=MOUNT, read_only=True))
    if [v.to_dict() for v in volumes] != [v.to_dict() for v in current]:
        api.set_space_volumes(args.repo, volumes=volumes)

    settings = {
        "MAX_CONCURRENT_ENVS": str(args.concurrency),
        "ENABLE_WEB_INTERFACE": "true",
        "RETROENV_TOOLSET": args.toolset,
        "RETROENV_BUCKET_ROOT": MOUNT,
    }
    existing = api.get_space_variables(args.repo)
    for key, value in settings.items():
        if key not in existing or existing[key].value != value:
            api.add_space_variable(args.repo, key, value)
    # Either of these would take precedence over the snapshot and serve something else.
    for stale in ("RETROENV_TASKS_REPO", "RETROENV_BENCHMARK_DIR"):
        if stale in existing:
            api.delete_space_variable(args.repo, stale)
            print(f"removed the stale {stale} variable")
    if args.hardware:
        api.request_space_hardware(args.repo, args.hardware)

    print(
        json.dumps(
            {
                "space": f"https://huggingface.co/spaces/{args.repo}",
                "commit": commit.oid,
                "snapshot_id": manifest["snapshot_id"],
                "bucket": bucket,
                "splits": {name: info["tasks"] for name, info in manifest["splits"].items()}
                | {name: info["tasks"] for name, info in manifest["evalsets"].items()},
                "sessions": args.concurrency,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
