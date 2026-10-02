"""Build Sarvam Indic OCR Bench once and publish it to a bucket for serving.

Extracts every crop of the pinned dataset revision into a content-addressed layout,
writes the per-split indexes, credits the source, verifies the result end to end, and
syncs it to the bucket the environment reads from - mounted on a Space or a job, or
fetched on first use anywhere else. Run it as a CPU job: it moves ~730 MB and decodes
6,909 image headers, which is the network's work and not a laptop's.

    <bucket>/<revision[:12]>/
        manifest.json               provenance, counts, index checksums, citation
        README.md                   credit and license
        indic_ocr_bench_test.json   6,909 tasks
        indic_ocr_bench_small.json  1,173 tasks (a subset of test; crops are shared)
        assets/<sha256>             one file per distinct crop
"""

import argparse
import hashlib
import json
from pathlib import Path

from nayana_ocr.data import indic_ocr_bench as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Local build directory")
    parser.add_argument("--bucket", default=bench.BUCKET)
    parser.add_argument(
        "--private", action="store_true", help="Create the bucket private (default public)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and verify only")
    args = parser.parse_args()

    root = args.work
    splits = {}
    for split in bench.SPLITS:
        rows = bench.build_split(split, root)
        index = bench.index_path(root, split)
        splits[split] = {
            "directory": bench.SPLITS[split][0],
            "tasks": len(rows),
            "languages": len({r["language"] for r in rows}),
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        }
        print(f"{split}: {len(rows)} tasks, {splits[split]['languages']} languages", flush=True)

    # Read it back exactly as a deployment will - in place, every crop checksummed - so a
    # bad build fails here rather than on the Space.
    served = bench.BenchCatalog(root / "unused-cache", root=root, fallback=False)
    for split in bench.SPLITS:
        for index in range(served.count(split)):
            task = served.at(split, index)
            served.asset_bytes(task["asset_sha256"], task["task_id"])
    assets = list((root / bench.VERSION / "assets").iterdir())
    size = sum(p.stat().st_size for p in assets)
    print(f"verified: {len(assets)} distinct crops, {size / 1e6:.0f} MB", flush=True)

    version_dir = root / bench.VERSION
    (version_dir / "manifest.json").write_text(json.dumps({
        "source": f"https://huggingface.co/datasets/{bench.REPO}",
        "repo": bench.REPO,
        "revision": bench.REVISION,
        "license": bench.LICENSE,
        "credit": "Sarvam AI, Sarvam Indic OCR Bench",
        "citation": bench.CITATION,
        "splits": splits,
        "distinct_assets": len(assets),
        "asset_bytes": size,
        "task_family": bench.FAMILY,
        "note": "Images and ground truth are unmodified. Served for evaluation only.",
    }, indent=2) + "\n")
    (version_dir / "README.md").write_text(f"""# Sarvam Indic OCR Bench - serving copy

This is a serving copy of [Sarvam Indic OCR Bench](https://huggingface.co/datasets/{bench.REPO})
by **Sarvam AI**, revision `{bench.REVISION}`, licensed **Apache-2.0**
(https://www.apache.org/licenses/LICENSE-2.0). Images and ground truth are unmodified; the
crops are stored by SHA-256 and indexed per split so an OpenEnv environment can serve them
for evaluation. All credit for the benchmark belongs to Sarvam AI. Please cite:

```bibtex
{bench.CITATION}
```
""")
    if args.dry_run:
        print("dry run: not publishing", flush=True)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_bucket(args.bucket, private=args.private, exist_ok=True)
    api.sync_bucket(str(version_dir), f"hf://buckets/{args.bucket}/{bench.VERSION}")
    # Confirm the indexes landed before anything is deployed against them.
    wanted = [f"{bench.VERSION}/{split}.json" for split in bench.SPLITS]
    present = {r.path for r in api.get_bucket_paths_info(args.bucket, wanted)}
    if set(wanted) - present:
        raise SystemExit(f"publish incomplete: missing {sorted(set(wanted) - present)}")
    print(f"published {bench.SNAPSHOT_ID} to hf://buckets/{args.bucket}/{bench.VERSION}")


if __name__ == "__main__":
    main()
