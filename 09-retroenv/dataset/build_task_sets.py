#!/usr/bin/env python3
"""Freeze the RL task sets: training, validation, the 100-task board and the held-out test.

    uv run python -m dataset.build_task_sets            # writes data/eval-{train12k,val150,core100,heldout300}.json

Each set pins task ids that already exist in the release and is served as a split of its own,
so a task scores exactly as it does in its source split. Four sets, mutually disjoint:

| Set | Size | Source | Tiers (easy/medium/hard) |
|---|---:|---|---|
| `train12k`    | 12,000 + 1,200 variants | train | 6000 / 4800 / 1200 |
| `val150`      | 150 | dev | 50 / 50 / 50 |
| `core100`     | 100 | test_id + test_hard | 34 / 33 / 33, a superset of final_eval and core30 |
| `heldout300`  | 300 | test_id + test_hard, disjoint from the board | 100 / 100 / 100 |
| `core150`     | 150 | test_id + test_hard | 50 / 50 / 50, a superset of core100 |

`train12k` lifts the hard tier from the split's own 2% to 10%: the base model passes about one
task in ten, so a draw at the natural rate would put almost no hard tasks in front of it, while
a balanced draw would spend most groups on tasks it never solves. The constraint variants ride
on top of the 12,000 because the depth, stock and forbidden-class rules are only learnable from
tasks that carry them, and the rare kinds are drawn first so each stays represented.

The boards nest: `core30` (30) in `final_eval` (50) in `core100` (100) in `core150` (150), each
keeping the previous one's tasks in their original order and appending the new ones. A board run
at any size stays comparable with every smaller one. `heldout300` is drawn before `core150` and
excluded from it, so growing the board never eats into the sealed test set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from retroenv.evalsets import load_evalset, make_evalset, select_tiered
from retroenv.models import RetroTask
from retroenv.store import TaskStore

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ("train12k", "retroenv-train12k-v1", {"easy": 6000, "medium": 4800, "hard": 1200})
VARIANTS = 1200
VAL = ("val150", "retroenv-val150-v1", {"easy": 50, "medium": 50, "hard": 50})
BOARD = ("core100", "retroenv-core100-v1", {"easy": 34, "medium": 33, "hard": 33})
HELDOUT = ("heldout300", "retroenv-heldout300-v1", {"easy": 100, "medium": 100, "hard": 100})
BOARD150 = ("core150", "retroenv-core150-v1", {"easy": 50, "medium": 50, "hard": 50})
TEST_SOURCES = ("test_id", "test_hard")
NESTED_BOARDS = ("core30", "final_eval", "core100", "core150")
STRATA = "source split x shortest-route depth, visited round-robin from shallow to deep"
WITHIN = "sha256(seed:task_id); a first-reaction class the tier has not used is preferred"


def seeded_order(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_round_robin(
    tasks: Iterable[RetroTask], key: Callable[[RetroTask], str], quota: int, seed: str
) -> list[RetroTask]:
    """Draw `quota` tasks one at a time from each group of `key`, so a scarce group survives.

    The constraint variants are grouped by kind rather than by tier: `diversity` and
    `forbidden_class` have only a couple of hundred tasks each against `max_depth`'s 3,659,
    and a proportional draw would leave almost none of them.
    """
    groups: dict[str, list[RetroTask]] = {}
    for task in tasks:
        groups.setdefault(key(task), []).append(task)
    for members in groups.values():
        members.sort(key=lambda task: seeded_order(seed, task.task_id))
    available = sum(len(members) for members in groups.values())
    if available < quota:
        raise ValueError(f"{available} candidates, fewer than the quota of {quota}")
    picked: list[RetroTask] = []
    while len(picked) < quota:
        for name in sorted(groups):
            if len(picked) == quota:
                break
            if groups[name]:
                picked.append(groups[name].pop(0))
    return picked


def extend_board(
    inherited: list[RetroTask], candidates: Iterable[RetroTask], quotas: dict[str, int], seed: str
) -> tuple[list[RetroTask], dict[str, int]]:
    """Keep an existing board in its original order and top it up to the new quotas."""
    held = {task.task_id for task in inherited}
    short = {tier: quotas[tier] - sum(t.difficulty.get("tier") == tier for t in inherited) for tier in quotas}
    if any(count < 0 for count in short.values()):
        raise ValueError(f"the inherited board already exceeds the quotas: {short}")
    added = select_tiered([t for t in candidates if t.task_id not in held], short, seed)
    return inherited + added, short


def write(path: Path, name: str, tasks: list[RetroTask], seed: str, design: dict[str, Any]) -> dict[str, Any]:
    record = make_evalset(name, tasks, seed=seed, design=design)
    path.write_text(json.dumps(record, indent=2) + "\n")
    load_evalset(path)  # round-trip the id check so a bad write fails here, not at serving time
    return record


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    tasks = record["tasks"]
    return {
        "size": record["size"],
        "evalset_id": record["evalset_id"][:16],
        "tiers": dict(sorted(Counter(t["tier"] for t in tasks).items())),
        "source_splits": dict(sorted(Counter(t["source_split"] for t in tasks).items())),
        "min_depth": dict(sorted(Counter(t["min_depth"] for t in tasks).items())),
        "first_reactions": len({t["first_reaction"] for t in tasks}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--install",
        action="store_true",
        help="also copy the sets into <release>/evalsets/ so a local server serves them",
    )
    args = parser.parse_args(argv)

    store = TaskStore(args.release / "tasks-private", args.release / "stocks")
    base = {"tier": "the release's difficulty.tier", "strata": STRATA, "within_stratum": WITHIN}
    report: dict[str, Any] = {}

    name, seed, quotas = TRAIN
    pool = list(store.tasks("train"))
    standard = select_tiered([t for t in pool if t.variant == "standard"], quotas, seed)
    variants = select_round_robin(
        [t for t in pool if t.variant != "standard"], lambda task: task.variant, VARIANTS, seed
    )
    design = {
        **base,
        "source_splits": ["train"],
        "variant": f"{sum(quotas.values())} standard plus {VARIANTS} constraint variants",
        "quotas": quotas,
        "variant_quota": VARIANTS,
        "variant_draw": "variant kinds visited round-robin, scarcest first, by sha256(seed:task_id)",
        "purpose": "RL rollouts; hard lifted to 10% of the standard draw from the split's own 2%",
    }
    report[name] = summarize(write(args.data / f"eval-{name}.json", name, standard + variants, seed, design))

    name, seed, quotas = VAL
    design = {**base, "source_splits": ["dev"], "variant": "standard", "quotas": quotas,
              "purpose": "checkpoint validation during training; never trained on, never reported"}
    val = select_tiered([t for t in store.tasks("dev") if t.variant == "standard"], quotas, seed)
    report[name] = summarize(write(args.data / f"eval-{name}.json", name, val, seed, design))

    candidates = [t for split in TEST_SOURCES for t in store.tasks(split) if t.variant == "standard"]
    by_id = {task.task_id: task for task in candidates}
    board_base = {**base, "source_splits": list(TEST_SOURCES), "variant": "standard"}

    # The board grows out of final_eval so the existing core30 and final_eval runs stay comparable.
    name, seed, quotas = BOARD
    inherited = [by_id[t["task_id"]] for t in load_evalset(args.data / "eval-final_eval.json")["tasks"]]
    board, short = extend_board(inherited, candidates, quotas, seed)
    design = {**board_base, "quotas": quotas, "extends": {"final_eval": len(inherited), "added": short},
              "purpose": "the iteration board; core30 and final_eval are nested in it, in their original order"}
    report[name] = summarize(write(args.data / f"eval-{name}.json", name, board, seed, design))

    name, seed, quotas = HELDOUT
    taken = {task.task_id for task in board}
    design = {**board_base, "quotas": quotas, "disjoint_from": BOARD[0],
              "purpose": "sealed final test; run once, not for iteration"}
    heldout = select_tiered([t for t in candidates if t.task_id not in taken], quotas, seed)
    report[name] = summarize(write(args.data / f"eval-{name}.json", name, heldout, seed, design))

    # Drawn after the sealed test and excluding it, so growing the board never eats into it.
    name, seed, quotas = BOARD150
    sealed = {task.task_id for task in heldout}
    larger, short = extend_board(board, [t for t in candidates if t.task_id not in sealed], quotas, seed)
    design = {**board_base, "quotas": quotas, "extends": {BOARD[0]: len(board), "added": short},
              "disjoint_from": HELDOUT[0],
              "purpose": "the iteration board at 150; core100 is nested in it, in its original order"}
    report[name] = summarize(write(args.data / f"eval-{name}.json", name, larger, seed, design))

    # core100 is a prefix of core150, so the largest board stands for both here.
    sets = {"train12k": standard + variants, "val150": val, BOARD150[0]: larger, HELDOUT[0]: heldout}
    ids = {name: {task.task_id for task in tasks} for name, tasks in sets.items()}
    for left, right in itertools.combinations(sorted(ids), 2):
        if ids[left] & ids[right]:
            raise SystemExit(f"{left} and {right} share {len(ids[left] & ids[right])} task(s)")
    chain = [{t["task_id"] for t in load_evalset(args.data / f"eval-{n}.json")["tasks"]} for n in NESTED_BOARDS]
    for smaller, bigger, names in zip(chain, chain[1:], zip(NESTED_BOARDS, NESTED_BOARDS[1:])):
        if not smaller <= bigger:
            raise SystemExit(f"{names[0]} must be nested in {names[1]}")

    if args.install:
        target = args.release / "evalsets"
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.data.glob("eval-*.json")):
            shutil.copy2(path, target / f"{load_evalset(path)['name']}.json")
        report["installed"] = str(target)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
