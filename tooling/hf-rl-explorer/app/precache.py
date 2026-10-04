"""Index datasets ahead of time: their indexes and packs, ready in STORAGE_DIR before anyone opens them.

    uv run python -m app.precache --featured --top 40          # the collections and the 40 trending Harbor datasets
    uv run python -m app.precache FineEnvs/some-dataset ...    # these
    hf buckets sync .local-data/indexes hf://buckets/<bucket>/indexes   # then copy them to the Space's bucket
    hf buckets sync .local-data/packs hf://buckets/<bucket>/packs

Public datasets only. With --use-token, files are downloaded with your HF token for its higher rate limits;
listings and access checks stay anonymous, so nothing private is read.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="dataset ids, org/name")
    ap.add_argument("--featured", action="store_true", help="every dataset in the collections")
    ap.add_argument("--top", type=int, default=0, help="the N trending Harbor datasets on the Hub")
    ap.add_argument("--use-token", action="store_true", help="download public files with your HF token")
    ap.add_argument("--force", action="store_true", help="rebuild even when an index is current")
    args = ap.parse_args()
    if args.use_token:
        from huggingface_hub import get_token

        os.environ["RLX_INDEX_TOKEN"] = get_token() or ""
    from . import catalog, config   # after the token is in the environment

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    specs = list(args.specs)
    if args.featured:
        specs += catalog.featured_datasets()
    if args.top:
        rows = sorted(catalog.environments(), key=lambda d: d["trending"] * 1e9 + d["downloads"], reverse=True)
        specs += [d["id"] for d in rows if d.get("kind") == "dataset" and d.get("framework", "harbor") == "harbor"][: args.top]
    seen, failed = set(), 0
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        t = time.time()
        try:
            meta = catalog.info(spec)
            if meta.get("framework") != "harbor" and not catalog.looks_harbor(spec, meta["sha"]):
                print(f"  rows     {spec}: read row by row (app/envs), nothing to index")
                continue
            if not args.force and catalog._read_index(spec, meta["sha"]) and catalog._read_pack(spec, meta["sha"]):
                print(f"  current  {spec}")
                continue
            job: dict = {}
            idx, pack = catalog.build_index(spec, meta, job)
            if pack is not None:
                catalog._write_pack(spec, meta["sha"], pack)
            catalog._write_index(idx)
            print(f"  indexed  {spec}: {len(idx['tasks']):,} tasks in {time.time() - t:.0f}s"
                  + ("" if idx["tasks"] else f" ({idx.get('note')})"), flush=True)
        except Exception as exc:  # noqa: BLE001 - one dataset failing leaves the rest
            failed += 1
            print(f"  failed   {spec}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    print(f"{len(seen) - failed} of {len(seen)} datasets ready in {config.STORAGE_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
