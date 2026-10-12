"""Publish a built release, known routes included, to one public Hub dataset.

    uv run python -m dataset.publish_release                  # LiteFold/RetroEnv
    uv run python -m dataset.publish_release --runs runs      # also the eval runs

HF_TOKEN must be able to write the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

MANIFEST_ROW = (
    "| `manifest.json`, `audit.json`, `checksums.json` | build provenance, leakage and solvability audit, SHA-256 |"
)
RUNS_ROW = "| `runs/` | model evaluation episodes (full trajectories and scores) |"


def release_card(card: str, *, runs: bool) -> str:
    if not runs or RUNS_ROW in card:
        return card
    if MANIFEST_ROW not in card:
        raise SystemExit("release README has no manifest row to append runs/ after")
    return card.replace(MANIFEST_ROW, f"{MANIFEST_ROW}\n{RUNS_ROW}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("data/release/RetroEnv-RL"))
    parser.add_argument("--repo", default="LiteFold/RetroEnv")
    parser.add_argument("--runs", type=Path, help="eval run directory to publish under runs/")
    args = parser.parse_args(argv)

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.release,
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=["README.md"],
        commit_message="RetroEnv-RL release",
    )
    card = release_card((args.release / "README.md").read_text(), runs=bool(args.runs))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
    if args.runs:
        api.upload_folder(
            folder_path=args.runs,
            path_in_repo="runs",
            repo_id=args.repo,
            repo_type="dataset",
            ignore_patterns=["*.log"],
            commit_message="Eval runs",
        )
    print(f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
