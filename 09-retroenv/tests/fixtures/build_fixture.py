"""Cut a small, committed test release out of a full build.

    uv run python -m tests.fixtures.build_fixture --release data/release/RetroEnv-RL

Keeps a few standard tasks and one task of each variant per split, every library
reaction their known routes use, the most frequent templates, and a slice of the
stock, so server and integration tests run on real tasks without the full release.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from retroenv.reactions import iter_reactions

from dataset.pipeline.release import SPLITS, public_row, write_checksums, write_jsonl, write_library

STANDARD_PER_SPLIT = 4
TOP_TEMPLATES = 400
EXTRA_REACTIONS = 300
EXTRA_STOCK = 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("data/release/RetroEnv-RL"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "mini-release")
    args = parser.parse_args()
    rng = random.Random(0)
    source, out = args.release, args.output
    if out.exists():
        shutil.rmtree(out)

    rows_by_split = {}
    for split in SPLITS:
        rows = [json.loads(line) for line in (source / "tasks-private" / f"{split}.jsonl").open()]
        standard = [r for r in rows if r["variant"] == "standard"][:STANDARD_PER_SPLIT]
        variants = {}
        for row in rows:
            if row["variant"] != "standard" and row["variant"] not in variants:
                variants[row["variant"]] = row
        rows_by_split[split] = standard + list(variants.values())
    tasks = [row for rows in rows_by_split.values() for row in rows]

    keys = {
        (step["product"], tuple(step["reactants"]))
        for row in tasks
        for route in row["reference_routes"]
        for step in route["steps"]
    }
    library_rows, others = [], []
    for row in iter_reactions(source / "library" / "reactions.jsonl.gz"):
        (library_rows if (row["product_smiles"], tuple(row["reactant_smiles"])) in keys else others).append(row)
    library_rows += rng.sample([r for r in others if r["visible"]], EXTRA_REACTIONS)
    all_templates = json.loads((source / "library" / "templates.json").read_text())
    needed = {row["template"] for row in library_rows if row["template"]}
    kept = dict(Counter(all_templates).most_common(TOP_TEMPLATES)) | {
        t: all_templates[t] for t in needed if t in all_templates
    }
    visible = json.loads((source / "library" / "templates-visible.json").read_text())

    stock = (source / "stocks" / "paroutes-archive-leaves.smi").read_text().split()
    needed_stock = {
        m for row in tasks for route in row["reference_routes"] for s in route["steps"] for m in s["reactants"]
    }
    stock_slice = sorted(set(stock) & needed_stock | set(rng.sample(stock, EXTRA_STOCK)))

    for split, rows in rows_by_split.items():
        write_jsonl(out / "tasks-private" / f"{split}.jsonl", rows)
        write_jsonl(out / "tasks-public" / f"{split}.jsonl", (public_row(r) for r in rows))
    (out / "stocks").mkdir(parents=True)
    (out / "stocks" / "paroutes-archive-leaves.smi").write_text("\n".join(stock_slice) + "\n")
    reagents = json.loads((source / "library" / "reagents.json").read_text())
    write_library(out, library_rows, Counter(kept), Counter({t: n for t, n in visible.items() if t in kept}), reagents)
    manifest = json.loads((source / "manifest.json").read_text())
    (out / "manifest.json").write_text(
        json.dumps(
            {key: manifest[key] for key in ("schema_version", "task_schema", "stock", "library", "leakage", "design")},
            indent=1,
        )
    )
    write_checksums(out)
    print(
        f"wrote {out}: {len(tasks)} tasks, {len(library_rows)} reactions, {len(kept)} templates, {len(stock_slice)} stock"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
