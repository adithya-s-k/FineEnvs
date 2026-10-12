#!/usr/bin/env python3
"""Cut the curated release: four splits, nothing else to choose between.

    uv run python -m dataset.build_curated                       # data/release/RetroEnv-Curated
    uv run python -m dataset.build_curated --publish LiteFold/RetroEnv-Curated

| Split | Tasks | Drawn from | easy / medium / hard |
|---|---:|---|---|
| `train` | 13,200 | RetroEnv-RL train | 6,000 / 4,800 / 1,200, plus 1,200 constraint variants |
| `validation` | 150 | RetroEnv-RL dev | 50 / 50 / 50 |
| `heldout` | 300 | RetroEnv-RL test_id + test_hard | 100 / 100 / 100 |
| `test` | 150 | RetroEnv-RL test_id + test_hard | 50 / 50 / 50 |

The selections are the frozen, hash-checked evaluation sets in `data/eval-*.json`; this script
only materialises them as a standalone release, so `RETROENV_BENCHMARK_DIR` can point here and
every tool, the server, the trainer and the eval harness work with no further indirection.

The training split keeps the name `train` on purpose: `RetroRouteSession` and `PrecedentIndex`
hide a task's own reactions, patents and near-duplicates from the tools only when
`task.split == "train"`, so renaming it would let a training rollout look up its own answer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from retroenv.evalsets import load_evalset
from retroenv.models import RetroTask
from retroenv.store import TaskStore

from dataset.pipeline.release import write_checksums

ROOT = Path(__file__).resolve().parents[1]
NAME = "RetroEnv-Curated"
# (split name, the frozen selection it materialises)
SETS = (("train", "train12k"), ("validation", "val150"), ("heldout", "heldout300"), ("test", "core150"))
SOURCE_SPLITS = ("train", "dev", "test_id", "test_hard")
COPIED = ("library", "stocks")

CARD = """---
license: cc-by-4.0
pretty_name: RetroEnv-Curated
task_categories: [reinforcement-learning, text-generation]
tags: [chemistry, retrosynthesis, tool-use, rl-environment, openenv]
size_categories: [10K<n<100K]
configs:
- config_name: tasks
  default: true
  data_files:
{public}
- config_name: tasks_with_known_routes
  data_files:
{private}
---

# RetroEnv-Curated

Four splits for training and evaluating retrosynthesis agents, cut from
[LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv). Each task gives a target
molecule, a depth budget and optionally a constraint; the agent plans with tools and submits
synthesis trees whose every leaf must be in the frozen stock. A deterministic verifier judges
each step against the reaction library, never against a hidden answer.

This is the whole dataset. There are no alternative subsets to pick between.

| Split | Tasks | easy / medium / hard | Use |
|---|---:|---|---|
{rows}

The four splits never share a task. Held-out targets share no patent, route molecule, reaction,
scaffold group or Tanimoto >= 0.90 near-duplicate with `train` or with each other, and library
reactions touching a held-out key are hidden from the agent's tools. Within a tier the draw is
spread over shortest-route depth and prefers a first reaction the tier has not used yet, so even
the 150-task splits cover the chemistry rather than one corner of it.

`train` carries {variants:,} constraint variants on top of its standard tasks (`max_depth`,
`restricted_stock`, `forbidden_class`, `diversity`), because those rules are only learnable from
tasks that impose them. `train` is also the one split name that must not change: the environment
hides a task's own reactions and precedents from the tools only for tasks whose split is
`train`.

| Path | Contents |
|---|---|
| `tasks-public/` | what a policy may see |
| `tasks-private/` | the same tasks with hidden known routes and difficulty labels |
| `stocks/{stock}.smi` | {molecules:,} building blocks |
| `library/` | corpus reactions, retro-templates, reagent whitelist |
| `manifest.json`, `checksums.json` | build provenance and SHA-256 |

## Use

```bash
hf download LiteFold/{name} --repo-type dataset --local-dir RetroEnv-Curated
RETROENV_BENCHMARK_DIR=RetroEnv-Curated uv run uvicorn retroenv_openenv.server:app --port 8000
```

Source: PaRoutes v2 (Genheden & Bjerrum, Zenodo 7341155, CC-BY-4.0), patent reactions from
USPTO. Known routes are public, so report results knowing a model may have seen them.
"""
SPLIT_USE = {
    "train": "RL rollouts and SFT",
    "validation": "checkpoint selection during training",
    "heldout": "sealed final test; run once",
    "test": "the iteration board",
}


def resolve(store: TaskStore, selection: dict[str, Any], split: str) -> list[RetroTask]:
    """The selection's tasks, in its order, relabelled into the curated split."""
    by_id = {task.task_id: task for source in SOURCE_SPLITS for task in store.tasks(source)}
    missing = [t["task_id"] for t in selection["tasks"] if t["task_id"] not in by_id]
    if missing:
        raise SystemExit(f"{selection['name']} names {len(missing)} task(s) the release lacks, e.g. {missing[0]}")
    return [dataclasses.replace(by_id[t["task_id"]], split=split) for t in selection["tasks"]]


def write_split(root: Path, split: str, tasks: Iterable[RetroTask]) -> None:
    tasks = list(tasks)
    for folder, hidden in (("tasks-private", True), ("tasks-public", False)):
        path = root / folder / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task.to_dict(include_hidden=hidden), sort_keys=True) + "\n")


def tier_counts(tasks: list[RetroTask]) -> dict[str, int]:
    counts = Counter(task.difficulty.get("tier") for task in tasks)
    return {tier: counts.get(tier, 0) for tier in ("easy", "medium", "hard")}


def write_card(root: Path, splits: dict[str, list[RetroTask]], manifest: dict[str, Any]) -> None:
    def files(folder: str) -> str:
        return "\n".join(f"  - split: {s}\n    path: {folder}/{s}.jsonl" for s, _ in SETS)

    rows = "\n".join(
        f"| `{split}` | {len(tasks):,} | "
        f"{' / '.join(str(n) for n in tier_counts(tasks).values())} | {SPLIT_USE[split]} |"
        for split, tasks in splits.items()
    )
    (root / "README.md").write_text(
        CARD.format(
            public=files("tasks-public"),
            private=files("tasks-private"),
            rows=rows,
            name=NAME,
            stock=manifest["stock"]["id"],
            molecules=manifest["stock"]["molecules"],
            variants=sum(1 for task in splits["train"] if task.variant != "standard"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "release" / NAME)
    parser.add_argument("--publish", help="Hub dataset repo to upload to, e.g. LiteFold/RetroEnv-Curated")
    args = parser.parse_args(argv)

    store = TaskStore(args.release / "tasks-private", args.release / "stocks")
    selections = {split: load_evalset(args.data / f"eval-{name}.json") for split, name in SETS}
    splits = {split: resolve(store, record, split) for split, record in selections.items()}

    seen: dict[str, str] = {}
    for split, tasks in splits.items():
        for task in tasks:
            if task.task_id in seen:
                raise SystemExit(f"{task.task_id} is in both {seen[task.task_id]} and {split}")
            seen[task.task_id] = split

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for split, tasks in splits.items():
        write_split(args.out, split, tasks)
    for folder in COPIED:
        shutil.copytree(args.release / folder, args.out / folder)

    manifest = json.loads((args.release / "manifest.json").read_text())
    manifest["splits"] = {
        split: {"tasks": len(tasks), "tiers": tier_counts(tasks), "use": SPLIT_USE[split]}
        for split, tasks in splits.items()
    }
    manifest["summary"] = {
        split: dict(sorted(Counter(task.variant for task in tasks).items())) for split, tasks in splits.items()
    }
    manifest["curated"] = {
        "source": "LiteFold/RetroEnv",
        "selections": {split: {"name": r["name"], "evalset_id": r["evalset_id"]} for split, r in selections.items()},
        "note": "task ids come from the frozen selections; rows are relabelled into the split that holds them",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    write_card(args.out, splits, manifest)
    write_checksums(args.out)

    report = {
        "out": str(args.out),
        "splits": {split: {"tasks": len(t), **tier_counts(t)} for split, t in splits.items()},
        "size_mb": round(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6, 1),
    }
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.publish, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=args.out,
            repo_id=args.publish,
            repo_type="dataset",
            commit_message=f"{NAME}: train, validation, heldout and test",
        )
        report["published"] = f"https://huggingface.co/datasets/{args.publish}"
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
