"""Publish an explicitly selected, prepared window with the OpenEnv Docker Space."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from nayana_ocr.data.catalog import Catalog
from nayana_ocr.data.schema import REPO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    catalog = Catalog(args.snapshot)
    if catalog.manifest["config"]["source"] != REPO_ID:
        parser.error("Publish a prepared Nayana window, not synthetic fixtures")
    project = Path(__file__).resolve().parents[1] / "envs" / "nayana_ocr"
    with tempfile.TemporaryDirectory(prefix="nayana-space-") as directory:
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
                "snapshot",
                "tests",
            ),
        )
        shutil.copytree(
            args.snapshot,
            staging / "snapshot",
            ignore=shutil.ignore_patterns(".prepare.lock"),
        )
        # Keep the Space card tied to the selected window when a different one is published.
        provenance = {
            key: catalog.manifest[key]
            for key in (
                "snapshot_id",
                "schema_version",
                "datasets_version",
                "config",
                "source_license",
                "pages",
                "counts",
                "media_bytes",
            )
        }
        with (staging / "README.md").open("a") as card:
            card.write("\n## Bundled preview\n\n```json\n")
            card.write(json.dumps(provenance, indent=2) + "\n```\n")
        api = HfApi()
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
            commit_message=f"Nayana snapshot {catalog.manifest['snapshot_id'][:12]}",
        )
        print(f"https://huggingface.co/spaces/{args.space_id}")
        print(f"Space commit: {commit.oid}")


if __name__ == "__main__":
    main()
