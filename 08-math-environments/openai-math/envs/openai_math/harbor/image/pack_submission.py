#!/usr/bin/env python3
"""Pack the agent's proof into one file for the verifier.

Harbor runs this as a [[verifier.collect]] hook in the agent's sandbox after the agent finishes.
It clears /logs/artifacts and writes /logs/artifacts/submission.tar holding only regular
`Submission.lean` and `Submission/**.lean` files. Symlinks and other entries are skipped, never
followed, so only one plain file crosses to the verifier sandbox.

The agent's sandbox is untrusted, so the verifier re-checks every entry of the archive. This step
only keeps the transfer small and free of links.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
from pathlib import Path

WORKSPACE = Path("/workspace")
OUT_DIR = Path("/logs/artifacts")
MAX_TOTAL_BYTES = 768 << 20


def clear(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def main() -> None:
    clear(OUT_DIR)
    tmp = OUT_DIR / "submission.tar.part"
    total = 0
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        root = WORKSPACE / "Submission.lean"
        if regular(root):
            total += root.lstat().st_size
            if total > MAX_TOTAL_BYTES:
                raise SystemExit("submission exceeds the size limit")
            tar.add(root, arcname="Submission.lean", recursive=False)
        tree = WORKSPACE / "Submission"
        if tree.is_dir() and not tree.is_symlink():
            for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
                dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
                for name in sorted(filenames):
                    path = Path(dirpath) / name
                    if not name.endswith(".lean") or not regular(path):
                        continue
                    total += path.lstat().st_size
                    if total > MAX_TOTAL_BYTES:
                        raise SystemExit("submission exceeds the size limit")
                    tar.add(path, arcname=str(path.relative_to(WORKSPACE)), recursive=False)
    tmp.rename(OUT_DIR / "submission.tar")
    print(f"packed {total} bytes")


if __name__ == "__main__":
    main()
