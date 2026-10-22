"""Fetch a bounded sample from the pinned public SWE source."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from urllib.request import urlopen

DATASET = "nebius/SWE-rebench-openhands-trajectories"
REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/generated/swe_native.jsonl",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error("Bounded setup sample: limit must be 1..1000")
    from datasets import load_dataset

    with urlopen(
        f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/tools.json", timeout=45
    ) as response:
        tools = json.load(response)
    dataset = load_dataset(DATASET, revision=REVISION, split="train", streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in itertools.islice(dataset, args.limit):
            row["tools"] = tools
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "rows": count,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "sampling": "first N streaming rows; not a representative benchmark sample",
    }
    Path(str(args.output) + ".source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
