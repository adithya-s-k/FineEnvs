"""Build the shared Daytona snapshot from image/Dockerfile.

Harbor's own auto-snapshot waits only 10 minutes, and downloading Mathlib takes longer, so the
snapshot is built once up front and passed to Harbor with ``--ek snapshot_template_name=<name>``.
The name carries a hash of image/, so a changed recipe gets a new snapshot.

    python -m openai_math_harbor.snapshot [--print-name]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

IMAGE_DIR = Path(__file__).parents[1] / "image"


def snapshot_name() -> str:
    h = hashlib.sha256()
    # Regular files only: a stray __pycache__ or editor backup must not change or break the hash.
    for path in sorted(IMAGE_DIR.iterdir()):
        if path.is_file() and not path.name.startswith(".") and not path.name.endswith((".pyc", "~")):
            h.update(path.name.encode() + b"\0" + path.read_bytes())
    return f"openai-math-lean-{h.hexdigest()[:12]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-name", action="store_true")
    args = ap.parse_args()
    name = snapshot_name()
    if args.print_name:
        print(name)
        return
    from daytona import CreateSnapshotParams, Daytona, Image, Resources

    d = Daytona()
    try:
        snap = d.snapshot.get(name)
        print(f"{name} already exists ({snap.state})")
        return
    except Exception:
        pass
    image = Image.from_dockerfile(IMAGE_DIR / "Dockerfile")
    d.snapshot.create(
        CreateSnapshotParams(name=name, image=image, resources=Resources(cpu=4, memory=8, disk=10)),
        on_logs=lambda line: print(line, flush=True),
        timeout=3 * 3600,
    )
    print(f"created {name}")


if __name__ == "__main__":
    main()
