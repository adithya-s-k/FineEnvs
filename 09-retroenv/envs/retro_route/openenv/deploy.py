#!/usr/bin/env python3
"""Stage this folder with ../core into a Docker Space layout, and optionally upload it.

    python deploy.py --stage-only --stage-dir /tmp/retroenv-space   # build locally
    docker build -t retroenv /tmp/retroenv-space
    python deploy.py --repo YOUR_ORG/retroenv --tasks-repo LiteFold/RetroEnv

The Space gets Dockerfile and README.md at its root, with core/ and openenv/
beside them, the same layout the Dockerfile builds locally from envs/retro_route.
The answer key is not uploaded with the Space: it downloads the task dataset
named by --tasks-repo at startup (an HF_TOKEN Space secret if that dataset is private).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_ROOT = HERE.parent
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="Space id, e.g. FineEnvs/retroenv")
    parser.add_argument(
        "--tasks-repo",
        help="task dataset with tasks-private/ and stocks/ (org/name[@revision]), e.g. LiteFold/RetroEnv",
    )
    parser.add_argument("--public", action="store_true", help="create the Space public (default private)")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--toolset", choices=("full", "unaided"), default="full")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    staged = stage(args.stage_dir or Path(tempfile.mkdtemp(prefix="retroenv-space-")))
    print(f"staged {staged}")
    if args.stage_only:
        return 0
    if not args.repo or not args.tasks_repo:
        parser.error("--repo and --tasks-repo are required to upload")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    for key, value in {
        "MAX_CONCURRENT_ENVS": str(args.concurrency),
        "ENABLE_WEB_INTERFACE": "true",
        "RETROENV_TASKS_REPO": args.tasks_repo,
        "RETROENV_TOOLSET": args.toolset,
    }.items():
        api.add_space_variable(args.repo, key, value)
    api.upload_folder(
        repo_id=args.repo, repo_type="space", folder_path=staged, commit_message="Deploy RetroEnv OpenEnv server"
    )
    print(f"https://huggingface.co/spaces/{args.repo}  (add HF_TOKEN as a Space secret if the task dataset is private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
