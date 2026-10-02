"""Upload this standalone directory to a Docker Space."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("Concurrency must be positive")
    api = HfApi()
    api.create_repo(
        args.repo,
        repo_type="space",
        space_sdk="docker",
        private=not args.public,
        exist_ok=True,
    )
    variables = api.get_space_variables(args.repo)
    concurrency = args.concurrency or (
        int(variables["MAX_CONCURRENT_ENVS"].value)
        if "MAX_CONCURRENT_ENVS" in variables
        else 40
    )
    api.add_space_variable(args.repo, "MAX_CONCURRENT_ENVS", str(concurrency))
    api.add_space_variable(args.repo, "ENABLE_WEB_INTERFACE", "true")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=[
            "*.py",
            "*.json",
            "*.toml",
            "*.txt",
            "*.sh",
            "*.yaml",
            "README.md",
            "Dockerfile",
            ".dockerignore",
            ".gitignore",
            ".gitattributes",
        ],
        ignore_patterns=[
            ".deps/**",
            ".venv/**",
            "prepared/**",
            ".archive/**",
            "**/__pycache__/**",
            "**/*.egg-info/**",
            ".env",
            ".git/**",
        ],
        commit_message="Deploy SmolDataEnvs Multi-harness | Harbor",
    )
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
