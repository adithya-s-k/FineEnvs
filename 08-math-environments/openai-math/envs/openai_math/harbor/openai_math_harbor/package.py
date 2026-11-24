"""Assemble the folder that is published as the Hugging Face dataset.

Copies a generated dataset (adapter output) and adds the generator itself under generator/, so the
published repo can rebuild the image and regenerate the tasks on its own.

    python -m openai_math_harbor.package <adapter output dir> <stage dir>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

GENERATOR_ROOT = Path(__file__).parents[1]
GENERATOR_PARTS = ["README.md", "openai_math_harbor", "image", "tests"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def stage(dataset: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dataset, out, ignore=IGNORE)
    for part in GENERATOR_PARTS:
        src = GENERATOR_ROOT / part
        dst = out / "generator" / part
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


if __name__ == "__main__":
    stage(Path(sys.argv[1]), Path(sys.argv[2]))
    print("staged", sys.argv[2])
