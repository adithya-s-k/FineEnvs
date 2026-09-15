"""Publish the full-corpus server and attach its existing source bucket read-only."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.schema import REPO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
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
        r.path: r.size
        for r in api.get_bucket_paths_info(manifest["bucket_id"], list(expected))
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
        # Preserve unrelated mounts; this path is owned by the Nayana deployment.
        runtime = api.space_info(args.space_id).runtime
        volumes = [v for v in (runtime.volumes or []) if v.mount_path != "/corpus"]
        volumes.append(
            Volume(
                type="bucket",
                source=manifest["bucket_id"],
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
        for key, value in {
            "NAYANA_CORPUS_MANIFEST": "/app/corpus-manifest.json",
            "NAYANA_SOURCE_ROOT": "/corpus",
            "NAYANA_CACHE_DIR": "/tmp/nayana-cache",
            **judge_variables,
        }.items():
            if key not in current_variables or current_variables[key].value != value:
                api.add_space_variable(args.space_id, key, value)
        result = {
            "space_id": args.space_id,
            "commit": commit.oid,
            "snapshot_id": manifest["snapshot_id"],
            "bucket_id": manifest["bucket_id"],
            "volumes": [v.to_dict() for v in volumes],
            "bundled": "code and corpus manifest only; indexes and images fetched lazily",
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
