#!/usr/bin/env python3
"""Independently audit a generated RetroEnv benchmark before model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from retroenv.environment import RetroRouteSession
from retroenv.graph import routes_to_submission
from retroenv.retrieval import PrecedentIndex
from retroenv.store import TaskStore
from retroenv.taskgen import SPLITS, audit_splits
from retroenv.verifier import RouteVerifier


PRIVATE_ONLY_KEYS = {
    "reference_routes",
    "route_id",
    "reaction_id",
    "reaction_smarts",
    "source",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _find_private_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in PRIVATE_ONLY_KEYS:
                found.append(child_path)
            found.extend(_find_private_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_private_keys(child, f"{path}[{index}]"))
    return found


def _cross_split_near_duplicates(tasks: Iterable[Any], threshold: float) -> list[dict[str, Any]]:
    values = list(tasks)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    products = [
        sorted(
            {
                step.product
                for route in task.reference_routes
                for step in route.steps
            }
            | {task.target_smiles}
        )
        for task in values
    ]
    fingerprints = [
        [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in task_products]
        for task_products in products
    ]
    overlaps: list[dict[str, Any]] = []
    for right in range(1, len(values)):
        for left in range(right):
            if values[left].split == values[right].split:
                continue
            best = (0.0, "", "")
            for right_index, right_fp in enumerate(fingerprints[right]):
                similarities = DataStructs.BulkTanimotoSimilarity(
                    right_fp, fingerprints[left]
                )
                for left_index, similarity in enumerate(similarities):
                    if similarity > best[0]:
                        best = (
                            float(similarity),
                            products[left][left_index],
                            products[right][right_index],
                        )
            if best[0] >= threshold:
                overlaps.append(
                    {
                        "left_task": values[left].task_id,
                        "left_split": values[left].split,
                        "left_product": best[1],
                        "right_task": values[right].task_id,
                        "right_split": values[right].split,
                        "right_product": best[2],
                        "tanimoto": round(best[0], 6),
                    }
                )
    return overlaps


def audit(root: Path, *, expected_eval_tasks: int = 20, threshold: float = 0.90) -> dict[str, Any]:
    private_dir = root / "tasks-private"
    public_dir = root / "tasks-public"
    stocks_dir = root / "stocks"
    store = TaskStore(private_dir, stocks_dir)
    verifier = RouteVerifier()
    tasks = list(store.iter_all())
    failures: list[str] = []
    counts = {split: len(store.tasks(split)) for split in SPLITS}
    if counts["eval"] != expected_eval_tasks:
        failures.append(
            f"eval contains {counts['eval']} tasks, expected {expected_eval_tasks}"
        )

    split_audit = audit_splits(tasks)
    if not split_audit["passed"]:
        failures.append("exact target/scaffold/route/reaction/source split leakage")
    near_duplicates = _cross_split_near_duplicates(tasks, threshold)
    if near_duplicates:
        failures.append(
            f"{len(near_duplicates)} cross-split route-product pairs have "
            f"Tanimoto >= {threshold}"
        )

    private_by_id = {task.task_id: task for task in tasks}
    public_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows = _read_jsonl(public_dir / f"{split}.jsonl")
        public_rows.extend(rows)
        expected_ids = {task.task_id for task in store.tasks(split)}
        actual_ids = {str(row.get("task_id", "")) for row in rows}
        if actual_ids != expected_ids:
            failures.append(f"{split}: public/private task IDs differ")
        for row in rows:
            leaked = _find_private_keys(row)
            if leaked:
                failures.append(
                    f"{split}/{row.get('task_id')}: public keys leak at {leaked[:3]}"
                )
            task = private_by_id.get(str(row.get("task_id", "")))
            if task and row != task.to_dict(include_references=False):
                failures.append(f"{split}/{task.task_id}: public task does not match private shell")

    reference_routes = 0
    reference_steps = 0
    oracle_failures = 0
    route_failures = 0
    for task in tasks:
        stock = store.stock(task.stock_id)
        reference_routes += len(task.reference_routes)
        reference_steps += sum(len(route.steps) for route in task.reference_routes)
        first_cuts = {
            tuple(sorted(route.steps[0].reactants))
            for route in task.reference_routes
            if route.steps
        }
        if not task.min_routes <= len(task.reference_routes) <= task.max_routes:
            failures.append(
                f"{task.task_id}: {len(task.reference_routes)} references outside "
                f"{task.min_routes}..{task.max_routes}"
            )
        if len(first_cuts) < task.min_routes:
            failures.append(f"{task.task_id}: insufficient distinct reference first cuts")
        if any(not 1 <= len(route.steps) <= task.max_steps for route in task.reference_routes):
            failures.append(f"{task.task_id}: reference route violates step cap")
        for route in task.reference_routes:
            submission = {
                "route": [step.to_dict(include_evidence=False) for step in route.steps]
            }
            if not verifier.score_route(task, submission, stock).valid:
                route_failures += 1
        oracle = routes_to_submission(
            task.target_smiles,
            task.reference_routes[: task.max_routes],
            stock,
            source="private-audit-oracle",
        )
        score = verifier.score_submission(task, oracle, stock)
        if not score.valid or score.reward != 1.0:
            oracle_failures += 1
    if route_failures:
        failures.append(f"{route_failures} private reference routes fail replay")
    if oracle_failures:
        failures.append(f"{oracle_failures} tasks fail graph oracle replay")

    train_index = PrecedentIndex(store.tasks("train"))
    non_train_ids = {
        task.task_id for split in ("dev", "eval", "stress") for task in store.tasks(split)
    }
    leaked_precedents = sorted(
        {record.task_id for record in train_index.records} & non_train_ids
    )
    if leaked_precedents:
        failures.append(f"precedent index contains non-train task IDs: {leaked_precedents[:3]}")

    eval_task = store.tasks("eval")[0]
    session = RetroRouteSession(precedent_index=train_index)
    opening_a = session.reset(
        eval_task, store.stock(eval_task.stock_id), episode_id="determinism-audit"
    )
    opening_b = session.reset(
        eval_task, store.stock(eval_task.stock_id), episode_id="determinism-audit"
    )
    if opening_a != opening_b:
        failures.append("session reset is not deterministic with a fixed episode ID")
    leaked_opening = _find_private_keys(opening_a)
    if leaked_opening:
        failures.append(f"environment opening leaks private keys at {leaked_opening[:3]}")
    stock_results = session.stock_retrieve("class:amine", limit=10_000).get("results", [])
    if len(stock_results) > session.max_search_results:
        failures.append("stock retrieval exceeded the configured result cap")

    result = {
        "schema_version": "retro-benchmark-audit-v1",
        "passed": not failures,
        "root": str(root),
        "counts": counts,
        "tasks": len(tasks),
        "public_tasks": len(public_rows),
        "reference_routes": reference_routes,
        "reference_steps": reference_steps,
        "train_precedents": len(train_index.records),
        "split_audit": split_audit,
        "near_duplicate_threshold": threshold,
        "cross_split_near_duplicates": near_duplicates,
        "reference_route_replay_failures": route_failures,
        "oracle_graph_replay_failures": oracle_failures,
        "failures": failures,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--expected-eval-tasks", type=int, default=20)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(
        args.benchmark_dir,
        expected_eval_tasks=args.expected_eval_tasks,
        threshold=args.near_duplicate_threshold,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
