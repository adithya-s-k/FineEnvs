#!/usr/bin/env python3
"""Freeze the 50-task final_eval subset of the held-out test splits, and core30 inside it.

    uv run python -m dataset.build_final_eval   # writes data/eval-final_eval.json and data/eval-core30.json

17 easy, 17 medium and 16 hard tasks by the release's own difficulty tier, drawn from
test_id and test_hard, standard tasks only (the constraint variants are covered by the full
splits). Within a tier the draw is spread over source split and shortest-route depth, and
prefers first reactions the tier has not used. The file pins the task ids, not the tasks:
the server resolves them against the release and serves them as the ``final_eval`` split.

core30 is 10 easy, 10 medium and 10 hard of those 50, drawn by the same rules: the quick board
every model is run on, nested in final_eval so its scores stay comparable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from retroenv.evalsets import load_evalset, make_evalset, select_tiered
from retroenv.store import TaskStore

ROOT = Path(__file__).resolve().parents[1]
NAME = "final_eval"
SEED = "retroenv-final-eval-v1"
QUOTAS = {"easy": 17, "medium": 17, "hard": 16}
CORE_NAME, CORE_SEED, CORE_QUOTAS = "core30", "retroenv-core30-v1", {"easy": 10, "medium": 10, "hard": 10}
SOURCES = ("test_id", "test_hard")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / f"eval-{NAME}.json")
    args = parser.parse_args(argv)

    store = TaskStore(args.release / "tasks-private", args.release / "stocks")
    candidates = [task for split in SOURCES for task in store.tasks(split) if task.variant == "standard"]
    chosen = select_tiered(candidates, QUOTAS, SEED)
    design = {
        "source_splits": list(SOURCES),
        "variant": "standard",
        "quotas": QUOTAS,
        "tier": "the release's difficulty.tier",
        "strata": "source split x shortest-route depth, visited round-robin from shallow to deep",
        "within_stratum": "sha256(seed:task_id); a first-reaction class the tier has not used is preferred",
    }
    record = make_evalset(NAME, chosen, seed=SEED, design=design)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    load_evalset(args.output)  # round-trip the id check
    core = select_tiered(chosen, CORE_QUOTAS, CORE_SEED)
    core_design = {**design, "nested_in": NAME, "quotas": CORE_QUOTAS}
    core_path = args.output.with_name(f"eval-{CORE_NAME}.json")
    core_path.write_text(json.dumps(make_evalset(CORE_NAME, core, seed=CORE_SEED, design=core_design), indent=2) + "\n")
    load_evalset(core_path)
    tasks = record["tasks"]
    print(
        json.dumps(
            {
                "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
                "evalset_id": record["evalset_id"],
                "tiers": Counter(t["tier"] for t in tasks),
                "source_splits": Counter(t["source_split"] for t in tasks),
                "min_depth": dict(sorted(Counter(t["min_depth"] for t in tasks).items())),
                "first_reactions": len({t["first_reaction"] for t in tasks}),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
