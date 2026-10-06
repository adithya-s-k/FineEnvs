"""Publish the Space's data to the public bucket FineEnvs/PortSimEnv.

    tasks/<pack>/tasks.jsonl.gz  gzipped task packs (downloaded at build time, like the twin)
    twin/<file>               the 3D twin's data (the Space image downloads these at build time: the Hub keeps
                              binaries in LFS and a Docker Space build would otherwise copy pointer files)
    rollouts/<run>/...        eval rollouts shown in the viewer (the Space mounts the bucket read-only at /data)
    video/<file>.mp4          the demo video played in the article (results/video/, not in git)

    openenv/.venv/bin/python tools/publish_bucket.py [--run dock-eval50 ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TWIN = HERE / "openenv" / "berth_openenv" / "web" / "twin"
RUNS = HERE.parents[1] / "results" / "rollouts"
VIDEO = HERE.parents[1] / "results" / "video"
BUCKET = "FineEnvs/PortSimEnv"
TASK_FILES = ["tasks/dock-v1-train/tasks.jsonl.gz"]
TWIN_FILES = ["twin.json.gz", "terrain.png", "cover.png", "scenery.json.gz", "surface.webp", "SOURCES.md"]


def main(argv=None) -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--run", action="append", default=None, help="rollout run under results/rollouts (repeatable)")
    args = ap.parse_args(argv)
    api = HfApi()
    api.create_bucket(args.bucket, private=False, exist_ok=True)
    add = [(str(TWIN / f), f"twin/{f}") for f in TWIN_FILES if (TWIN / f).is_file()]
    add += [(str(HERE / p), p) for p in TASK_FILES]
    add += [(str(v), f"video/{v.name}") for v in sorted(VIDEO.glob("*.mp4"))]
    for run in args.run or ["dock-eval50"]:
        root = RUNS / run
        if not (root / "index.json").is_file():
            raise SystemExit(f"{root} has no index.json")
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".json", ".md"}:
                add.append((str(p), f"rollouts/{run}/{p.relative_to(root).as_posix()}"))
    for i in range(0, len(add), 200):
        api.batch_bucket_files(args.bucket, add=add[i:i + 200])
    print(f"{len(add)} files -> https://huggingface.co/buckets/{args.bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
