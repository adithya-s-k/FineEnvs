"""Publish the full-corpus server and attach its existing source bucket read-only."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.evalset import INDEX_CACHE_BYTES
from nayana_ocr.data.schema import REPO_ID


def resolve_bucket_id(bucket_id):
    """Return the bucket's current canonical id.

    A manifest records the bucket name as published, and that name is hashed into the
    snapshot identity, so it must never be rewritten. An organization rename leaves HTTP
    requests working through a 307 redirect but does **not** move a Space's volume mount:
    the mount keeps the old source and every read fails with EIO. Resolve the live name
    for the mount while leaving the recorded provenance untouched.
    """
    import requests

    response = requests.get(
        f"https://huggingface.co/api/buckets/{bucket_id}",
        headers={"Authorization": f"Bearer {get_token()}"} if get_token() else {},
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
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--bucket-id",
        help="Mount source override; defaults to the manifest bucket resolved to its current name",
    )
    parser.add_argument(
        "--evalset",
        type=Path,
        action="append",
        default=[],
        help="Frozen set to ship and serve as its own split; repeat for several",
    )
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=16,
        help="Measured optimum; 32 halves throughput and pushes p95 toward the timeout",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--judge-config",
        type=Path,
        help="Passing HF Inference Providers calibration report from verify_judge.py",
    )
    args = parser.parse_args()
    manifest = json.loads(args.corpus_manifest.read_text())
    if manifest["config"]["source"] != REPO_ID:
        parser.error("Publish a finalized Nayana corpus index, not synthetic fixtures")
    api = HfApi()
    mount_bucket = args.bucket_id or resolve_bucket_id(manifest["bucket_id"])
    judge_variables = {}
    if args.judge_config:
        config = json.loads(args.judge_config.read_text())
        from nayana_ocr.server.judge import GemmaJudge

        judge = GemmaJudge(config["model"], config["provider"])
        if (
            config.get("status") != "passed"
            or config.get("policy_id") != judge.policy_id
        ):
            raise ValueError(
                "Calibrate this exact judge model/provider/rubric before deployment"
            )
        judge_variables = {
            "NAYANA_JUDGE_MODEL": judge.model,
            "NAYANA_JUDGE_PROVIDER": judge.provider,
        }
    prefix = f"openenv/indexes/{manifest['snapshot_id']}"
    expected = {
        f"{prefix}/{v['path']}": v["size"] for v in manifest["indexes"].values()
    }
    expected[f"{prefix}/manifest.json"] = args.corpus_manifest.stat().st_size
    available = {
        r.path: r.size for r in api.get_bucket_paths_info(mount_bucket, list(expected))
    }
    if available != expected:
        raise ValueError(
            "Publish and verify every index file in the bucket before deploying"
        )
    project = Path(__file__).resolve().parents[1] / "envs" / "nayana_ocr"
    with tempfile.TemporaryDirectory(prefix="nayana-space-") as directory:
        staging = Path(directory)
        catalog = CorpusCatalog(args.corpus_manifest, staging / "validation-cache")
        catalog.close()  # Validate the manifest identity without loading any images/indexes.
        shutil.rmtree(staging / "validation-cache")
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
                "snapshot",
                "tests",
            ),
        )
        shutil.copyfile(args.corpus_manifest, staging / "corpus-manifest.json")
        # The server serves each set beside the manifest as a split of its own, so what
        # is shipped here is exactly what the deployment exposes.
        from nayana_ocr.data.evalset import eval_split_name
        from nayana_ocr.data.evalset import load as load_evalset

        served = {}
        for path in args.evalset:
            record = load_evalset(path, manifest["snapshot_id"])
            served[path.name] = eval_split_name(record)
            shutil.copyfile(path, staging / path.name)
            print(
                f"evaluation set {path.name} -> split {served[path.name]} "
                f"({record['size']} tasks)",
                flush=True,
            )
        with (staging / "README.md").open("a") as card:
            card.write("\n## Served corpus\n\n")
            card.write(
                f"{sum(manifest['pages'].values()):,} pages across {len(manifest['pages'])} languages; "
            )
            card.write(
                f"{sum(c['tasks'] for c in manifest['counts']):,} indexed tasks.\n\n"
            )
            card.write(
                f"Snapshot: `{manifest['snapshot_id']}`. Image bounds are validated when a task is loaded.\n"
            )
            if served:
                card.write("\nFrozen evaluation splits: ")
                card.write(", ".join(f"`{name}`" for name in sorted(served.values())))
                card.write(".\n")
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
            folder_path=staging,
            delete_patterns=["snapshot/*"],
            commit_message=f"Serve complete Nayana corpus {manifest['snapshot_id'][:12]} from mounted bucket",
        )
        # An upload adds and replaces but never removes, so a set from an earlier
        # deployment would linger. Two sets covering the same languages and split resolve
        # to one split name and the server refuses to start rather than serve whichever
        # won, so sets this deployment does not carry are deleted explicitly.
        for entry in api.list_repo_files(args.space_id, repo_type="space"):
            if (
                entry.startswith("eval-")
                and entry.endswith(".json")
                and entry not in served
            ):
                api.delete_file(entry, args.space_id, repo_type="space")
                print(f"removed stale evaluation set {entry}", flush=True)

        # Preserve unrelated mounts; this path is owned by the Nayana deployment.
        runtime = api.space_info(args.space_id).runtime
        volumes = [v for v in (runtime.volumes or []) if v.mount_path != "/corpus"]
        volumes.append(
            Volume(
                type="bucket",
                source=mount_bucket,
                mount_path="/corpus",
                read_only=True,
            )
        )
        if [v.to_dict() for v in volumes] != [
            v.to_dict() for v in (runtime.volumes or [])
        ]:
            api.set_space_volumes(args.space_id, volumes=volumes)
        current_variables = api.get_space_variables(args.space_id)
        if judge_variables:
            import os

            token = os.environ.get("NAYANA_JUDGE_TOKEN") or get_token()
            if not token:
                raise ValueError(
                    "Provide an HF Inference Providers token for the Space secret"
                )
            api.add_space_secret(args.space_id, "NAYANA_JUDGE_TOKEN", token)
        # State the index budget explicitly: it must hold every published language index
        # at once, or a multi-language sweep re-copies ~200 MB databases as it rotates.
        index_bytes = max(
            INDEX_CACHE_BYTES,
            int(sum(info["size"] for info in manifest["indexes"].values()) * 1.2),
        )
        for key, value in {
            "NAYANA_CORPUS_MANIFEST": "/app/corpus-manifest.json",
            "NAYANA_SOURCE_ROOT": "/corpus",
            "NAYANA_CACHE_DIR": "/tmp/nayana-cache",
            "NAYANA_INDEX_CACHE_BYTES": str(index_bytes),
            # Concurrent evaluation needs a session per worker and a judge slot per
            # session, or the surplus callers queue against the semaphore and fail.
            # Judge throughput was measured to peak at 16 and degrade beyond it.
            "NAYANA_MAX_SESSIONS": str(args.sessions),
            "NAYANA_JUDGE_CONCURRENCY": str(args.judge_concurrency),
            **judge_variables,
        }.items():
            if key not in current_variables or current_variables[key].value != value:
                api.add_space_variable(args.space_id, key, value)
        result = {
            "space_id": args.space_id,
            "commit": commit.oid,
            "snapshot_id": manifest["snapshot_id"],
            "bucket_id": manifest["bucket_id"],
            "mount_bucket": mount_bucket,
            "volumes": [v.to_dict() for v in volumes],
            "bundled": "code and corpus manifest only; indexes and images fetched lazily",
            "index_cache_bytes": index_bytes,
            "sessions": args.sessions,
            "judge_concurrency": args.judge_concurrency,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
