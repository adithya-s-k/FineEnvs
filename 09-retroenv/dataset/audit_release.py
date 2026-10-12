"""Audit a built release using only the runtime code paths.

    uv run python -m dataset.audit_release [--release data/release/RetroEnv-RL]

Checks: schema and public/private views, held-out vs train leakage (keys and
near-duplicates), library visibility, solvability of every held-out task (and a
train sample) by its own compliant known routes, constraint witnesses, prompt
hygiene, and reward-hacking probes. Writes audit.json and exits 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from retroenv.benchmark import Benchmark
from retroenv.classes import classify_step
from retroenv.leakage import LeakageRules, molecule_keys, reaction_key, task_keys, task_molecules
from retroenv.models import RetroTask
from retroenv.reactions import iter_reactions
from retroenv.store import TaskStore
from retroenv.verifier import compliant_references, known_routes_submission

from dataset.pipeline import similarity
from dataset.pipeline.paroutes import WORKERS

HELD_OUT = ("dev", "test_id", "test_hard")
TRAIN_SAMPLE = 2000
# Leakage passes fork from a light parent (tasks and rules only); never from a loaded Benchmark,
# whose reference-counted objects every forked worker would copy.
LEAKAGE_WORKERS = WORKERS


def check_schema(root: Path, store: TaskStore) -> dict:
    report: dict = {"splits": {}, "failures": []}
    ids: set[str] = set()
    for split in store.splits():
        tasks = store.tasks(split)
        public = [json.loads(line) for line in (root / "tasks-public" / f"{split}.jsonl").open()]
        if [row["task_id"] for row in public] != [task.task_id for task in tasks]:
            report["failures"].append(f"{split}: public and private task order differ")
        for row, task in zip(public, tasks):
            if row != task.to_dict(include_hidden=False):
                report["failures"].append(f"{task.task_id}: public row is not the private row minus hidden fields")
                break
        duplicate = ids & {task.task_id for task in tasks}
        if duplicate:
            report["failures"].append(f"{split}: task ids repeated across splits: {sorted(duplicate)[:3]}")
        ids |= {task.task_id for task in tasks}
        report["splits"][split] = {"tasks": len(tasks), "variants": dict(Counter(task.variant for task in tasks))}
    return report


_RULES: LeakageRules = LeakageRules()
_HELD_KEYS: frozenset = frozenset()


def _train_crossings(tasks: list[RetroTask]) -> Counter:
    crossings: Counter = Counter()
    for task in tasks:
        shared = task_keys(task, _RULES) & _HELD_KEYS
        if shared:
            crossings[sorted(shared)[0][0]] += 1
    return crossings


def _visible_crossings(rows: list[dict]) -> Counter:
    crossings: Counter = Counter()
    for row in rows:
        keys = molecule_keys([row["product_smiles"]], _RULES)
        keys |= {reaction_key(row["product_smiles"], row["reactant_smiles"])} | {("patent", p) for p in row["patents"]}
        shared = keys & _HELD_KEYS
        if shared:
            crossings[sorted(shared)[0][0]] += 1
    return crossings


def _parallel(function, items: list, workers: int = LEAKAGE_WORKERS) -> list:
    size = max(1, len(items) // (workers * 4))
    with Pool(workers) as pool:
        return pool.map(function, [items[i : i + size] for i in range(0, len(items), size)])


def check_leakage(root: Path, store: TaskStore, rules: LeakageRules) -> dict:
    global _RULES, _HELD_KEYS
    held = [t for split in HELD_OUT if split in store.splits() for t in store.tasks(split)]
    train = list(store.tasks("train"))
    failures = []
    parents: dict[str, set] = defaultdict(set)
    for task in held:
        parents[task.parent_id] |= task_keys(task, rules)
    owner: dict[tuple, str] = {}
    for parent, keys in parents.items():
        clash = next((key for key in keys if owner.get(key, parent) != parent), None)
        if clash:
            failures.append(f"held-out parents {owner[clash]} and {parent} share {clash}")
        owner.update(dict.fromkeys(keys, parent))
    _RULES, _HELD_KEYS = rules, frozenset(owner)
    crossings = sum(_parallel(_train_crossings, train), Counter())
    if crossings:
        failures.append(f"train tasks share held-out keys: {dict(crossings)}")
    held_molecules = sorted({m for task in held for m in task_molecules(task)})
    near = similarity.above(
        similarity.packed(held_molecules, LEAKAGE_WORKERS),
        similarity.packed(sorted({t.target_smiles for t in train}), LEAKAGE_WORKERS),
        rules.near_duplicate_threshold,
        LEAKAGE_WORKERS,
    )
    if near:
        failures.append(f"{len(near)} train targets are near-duplicates of held-out molecules")
    visible = [row for row in iter_reactions(root / "library" / "reactions.jsonl.gz") if row["visible"]]
    leaks = sum(_parallel(_visible_crossings, visible), Counter())
    if leaks:
        failures.append(f"train-visible library reactions share held-out keys: {dict(leaks)}")
    return {
        "held_out_parents": len(parents),
        "held_out_keys": len(owner),
        "held_out_molecules": len(held_molecules),
        "train_tasks": len(train),
        "train_visible_reactions": len(visible),
        "failures": failures,
    }


def check_solvability(benchmark: Benchmark, seed: int = 0) -> dict:
    """Every held-out task and a train sample must pass with its own compliant known routes."""
    store = benchmark.store
    sample = [t for split in HELD_OUT if split in store.splits() for t in store.tasks(split)]
    train = list(store.tasks("train"))
    sample += random.Random(seed).sample(train, min(TRAIN_SAMPLE, len(train)))
    session = benchmark.session()
    rewards, failures = defaultdict(list), []
    for task in sample:
        stock = store.stock(task.stock_id)
        session.reset(task, stock)
        kind = f"{task.split}/{task.variant}"
        if any(route.route_id in session.observation()["prompt"] for route in task.reference_routes):
            failures.append(f"{task.task_id} ({kind}): prompt mentions a hidden route id")
        score = session.emit_routes(known_routes_submission(task, stock))["score"]
        rewards[kind].append(score["reward"])
        if not score["valid"]:
            failures.append(f"{task.task_id} ({kind}): known routes do not pass: {score['hard_failures'][:3]}")
        forbidden = frozenset(task.constraints.forbidden_classes)
        for route in compliant_references(task, stock, forbidden) if forbidden else ():
            if {classify_step(step.product, step.reactants).name for step in route.steps} & forbidden:
                failures.append(f"{task.task_id} ({kind}): a compliant witness uses a forbidden class")
    return {
        "checked": len(sample),
        "mean_oracle_reward": {key: round(float(np.mean(values)), 4) for key, values in sorted(rewards.items())},
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def check_probes(benchmark: Benchmark) -> dict:
    """Reward-hacking probes on standard held-out tasks with at least two known steps."""
    store = benchmark.store
    tasks = [t for t in store.tasks("test_id") if t.variant == "standard"][:50]
    session = benchmark.session()
    results = defaultdict(list)
    for task in tasks:
        stock = store.stock(task.stock_id)
        oracle = known_routes_submission(task, stock)
        if not oracle["routes"]:
            continue
        one = {"routes": oracle["routes"][:1]}

        def score(submission):
            session.reset(task, stock)
            return session.emit_routes(submission)["score"]

        base = score(one)
        results["oracle_one_route"].append(base["reward"])
        results["duplicated_route"].append(score({"routes": one["routes"] * 3})["reward"])
        lying = json.loads(json.dumps(one))
        node = lying["routes"][0]
        node["children"][0]["children"] = [{"type": "mol", "smiles": "C1CC2CCC1C2", "in_stock": True, "children": []}]
        results["false_stock_claim"].append(score(lying)["reward"])
        truncated = json.loads(json.dumps(one))
        for child in truncated["routes"][0]["children"][0]["children"]:
            child["children"], child["in_stock"] = [], child["in_stock"]
        results["first_step_only"].append(score(truncated)["reward"])
        results["empty"].append(score({"routes": []})["reward"])
    summary = {key: round(float(np.mean(values)), 4) for key, values in results.items() if values}
    failures = []
    if summary.get("duplicated_route", 0) >= summary.get("oracle_one_route", 1) - 0.1:
        failures.append("duplicating a route is not penalised")
    if summary.get("false_stock_claim", 0) > 0.4:
        failures.append("false stock claims are not capped")
    if summary.get("empty", 1) > 0.0:
        failures.append("an empty submission earns reward")
    return {"tasks": len(tasks), "mean_reward": summary, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("data/release/RetroEnv-RL"))
    args = parser.parse_args(argv)
    store = TaskStore(args.release / "tasks-private", args.release / "stocks")
    report = {
        "schema": check_schema(args.release, store),
        "leakage": check_leakage(args.release, store, LeakageRules.from_manifest(args.release / "manifest.json")),
    }
    benchmark = Benchmark.load(args.release)  # loaded only after every forked pass has finished
    report["solvability"] = check_solvability(benchmark)
    report["probes"] = check_probes(benchmark)
    report["passed"] = not any(section["failures"] for section in report.values() if isinstance(section, dict))
    (args.release / "audit.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: (
                    v
                    if k == "passed"
                    else {kk: vv for kk, vv in v.items() if kk != "failures"} | {"failures": v["failures"][:5]}
                )
                for k, v in report.items()
            },
            indent=1,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
