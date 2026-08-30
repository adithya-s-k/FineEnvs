#!/usr/bin/env python3
"""Ship an article or slide deck to a Hugging Face Space.

Sources live in this repo; artifacts live on the Hub. Uploads go over the HTTP
endpoint (`hf upload`) rather than `git push` to a Space remote, so no nested git
repos end up in this tree.

The SDK is read from the item's own README frontmatter — that file is the Space
card. `sdk: static` builds first and uploads `dist/`; anything else uploads the
directory as-is.

    python3 tools/deploy.py content/slides/rl-environments-101 HuggingEnvs/rl-environments-101-slides
    python3 tools/deploy.py content/articles/rl-environments-guide HuggingEnvs/rl-environments-guide
    python3 tools/deploy.py --dry-run content/slides/scaling-rl-amd HuggingEnvs/scaling-rl
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Never ship build artifacts to a Space. The Dockerfile installs and builds there,
# and node_modules alone is ~350MB.
EXCLUDE = [
    "**/node_modules/**",
    "**/dist/**",
    "**/.astro/**",
    "**/.git/**",
    "**/.DS_Store",
]


def card_path(src: Path) -> Path:
    p = src / "space.md"
    return p if p.exists() else src / "README.md"


def read_sdk(src: Path) -> str:
    """Pull `sdk:` out of the item's README frontmatter."""
    # `space.md` is the Space card when present, so the repo README can stay a repo README.
    readme = src / "space.md"
    if not readme.exists():
        readme = src / "README.md"
    if not readme.exists():
        sys.exit(f"error: no space.md or README.md in {src} — a Space needs a card with frontmatter")
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        sys.exit(f"error: {readme} has no YAML frontmatter — cannot tell which SDK to deploy as")
    frontmatter = text.split("---", 2)[1]
    match = re.search(r"^sdk:\s*(\S+)", frontmatter, re.M)
    if not match:
        sys.exit(f"error: {readme} frontmatter has no `sdk:` field")
    return match.group(1).strip()


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print(f"  $ {' '.join(cmd)}" + (f"   (in {cwd})" if cwd else ""))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"error: command failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="path to the article or deck, e.g. content/slides/rl-environments-101")
    ap.add_argument("space", help="target Space id, e.g. HuggingEnvs/rl-environments-101-slides")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--skip-build", action="store_true", help="reuse an existing dist/ for static decks")
    args = ap.parse_args()

    src = (REPO_ROOT / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    if not src.is_dir():
        sys.exit(f"error: {src} is not a directory")

    if shutil.which("hf") is None:
        sys.exit("error: the `hf` CLI is not on PATH — pip install -U huggingface_hub")

    card = card_path(src)
    sdk = read_sdk(src)
    print(f"→ {src.relative_to(REPO_ROOT)}  ({sdk} SDK)  →  {args.space}")

    if sdk == "static":
        if not args.skip_build:
            run(["npm", "run", "build"], cwd=src, dry_run=args.dry_run)
        upload_dir = src / "dist"
        if not args.dry_run and not upload_dir.is_dir():
            sys.exit(f"error: {upload_dir} not found — build did not produce a dist/")
        # The Space card is not part of dist/, so push it separately.
        run(["hf", "upload", args.space, str(upload_dir), ".", "--repo-type", "space"], dry_run=args.dry_run)
        run(["hf", "upload", args.space, str(card), "README.md", "--repo-type", "space"],
            dry_run=args.dry_run)
    else:
        cmd = ["hf", "upload", args.space, str(src), ".", "--repo-type", "space"]
        for pattern in EXCLUDE:
            cmd += ["--exclude", pattern]
        run(cmd, dry_run=args.dry_run)

    print(f"✓ https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
