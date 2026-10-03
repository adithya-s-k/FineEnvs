"""Offline Pass@k and route-quality evaluation against private task files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .store import TaskStore
from .verifier import RouteVerifier


def evaluate(
    store: TaskStore,
    prediction_rows: Iterable[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 8),
    splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_splits = splits or tuple(store.splits())
    unknown_splits = sorted(set(selected_splits) - set(store.splits()))
    if unknown_splits:
        raise ValueError(f"unknown evaluation split(s): {unknown_splits}")
    tasks = {
        task.task_id: task
        for split in selected_splits
        for task in store.tasks(split)
    }
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_rows: list[str] = []
    for row in prediction_rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in tasks:
            raise ValueError(f"prediction references unknown task_id: {task_id!r}")
        attempts = row.get("attempts")
        if attempts is None and any(key in row for key in ("route", "routes", "submission")):
            attempts = [row]
        if not isinstance(attempts, list):
            raise ValueError(f"{task_id}: attempts must be a list")
        if predictions[task_id]:
            duplicate_rows.append(task_id)
        predictions[task_id].extend(attempts)
    if duplicate_rows:
        raise ValueError(f"duplicate prediction rows for task(s): {sorted(set(duplicate_rows))[:5]}")

    verifier = RouteVerifier()
    episodes: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        stock = store.stock(task.stock_id)
        scored = []
        for attempt in predictions.get(task_id, []):
            if not isinstance(attempt, dict):
                result = verifier.score_submission(task, attempt, stock)
            elif "submission" in attempt:
                result = verifier.score_submission(task, attempt["submission"], stock)
            elif "routes" in attempt or attempt.get("type") == "mol":
                result = verifier.score_submission(task, attempt, stock)
            else:
                # Keep the v0 flat-route prediction format readable so old
                # experiment artifacts remain evaluable.
                route = attempt.get("route", attempt)
                if isinstance(route, dict) and (
                    "routes" in route or route.get("type") == "mol"
                ):
                    result = verifier.score_submission(task, route, stock)
                else:
                    result = verifier.score_route(task, route, stock)
            scored.append(
                {
                    "score": result,
                    "tool_calls": int(attempt.get("tool_calls", 0) or 0),
                    "invalid_proposals": int(attempt.get("invalid_proposals", 0) or 0),
                }
            )
        episodes.append({"task": task, "attempts": scored})

    total = len(episodes)
    pass_at = {
        f"pass@{k}": (
            sum(any(item["score"].valid for item in episode["attempts"][:k]) for episode in episodes)
            / total
            if total
            else 0.0
        )
        for k in ks
    }
    pass_counts = {
        k: sum(
            any(item["score"].valid for item in episode["attempts"][:k])
            for episode in episodes
        )
        for k in ks
    }
    first_results = [
        episode["attempts"][0]["score"] if episode["attempts"] else None
        for episode in episodes
    ]
    all_attempts = [item for episode in episodes for item in episode["attempts"]]
    successful = [item for item in all_attempts if item["score"].valid]
    tool_calls_success = [item["tool_calls"] for item in successful if item["tool_calls"] > 0]
    total_tool_calls = sum(item["tool_calls"] for item in all_attempts)
    total_invalid = sum(item["invalid_proposals"] for item in all_attempts)

    result = {
        "schema_version": "retro-eval-v1",
        "tasks": total,
        "tasks_with_predictions": sum(bool(episode["attempts"]) for episode in episodes),
        **{key: round(value, 6) for key, value in pass_at.items()},
        "pass_ci95": {
            f"pass@{k}": _wilson_interval(pass_counts[k], total) for k in ks
        },
        "top1_route_validity": _mean(
            float(score.valid) if score is not None else 0.0 for score in first_results
        ),
        "top1_reward": _mean(
            score.reward if score is not None else 0.0 for score in first_results
        ),
        "top1_parse_validity": _mean(
            score.components.get("parse_validity", float(score.valid))
            if score is not None
            else 0.0
            for score in first_results
        ),
        "top1_molecule_validity": _mean(
            score.metrics.get("molecule_validity", float(score.valid))
            if score is not None
            else 0.0
            for score in first_results
        ),
        "top1_graph_validity": _mean(
            score.metrics.get("graph_validity", float(score.valid))
            if score is not None
            else 0.0
            for score in first_results
        ),
        "top1_step_validity": _mean(
            score.metrics.get("step_validity", 0.0) if score is not None else 0.0
            for score in first_results
        ),
        "top1_reference_similarity": _mean(
            score.metrics.get("reference_similarity", 0.0) if score is not None else 0.0
            for score in first_results
        ),
        "top1_building_block_completion": _mean(
            score.metrics.get("building_block_completion", 0.0) if score is not None else 0.0
            for score in first_results
        ),
        "top1_exact_reference_match": _mean(
            float(score.metrics.get("exact_reference_match", False))
            if score is not None
            else 0.0
            for score in first_results
        ),
        "top1_verified_route_diversity": _mean(
            score.metrics.get("verified_route_diversity", 0.0)
            if score is not None
            else 0.0
            for score in first_results
        ),
        "average_successful_steps": _mean(
            item["score"].metrics.get("step_count", 0) for item in successful
        ),
        "average_tool_calls_per_success": _mean(tool_calls_success),
        "invalid_proposal_rate": (
            round(total_invalid / total_tool_calls, 6) if total_tool_calls else 0.0
        ),
        "attempts": len(all_attempts),
        "successful_attempts": len(successful),
        "verification_tiers": {
            tier: sum(item["score"].verification_tier == tier for item in all_attempts)
            for tier in ("dataset_supported", "template_supported", "rejected")
        },
        "by_split": _by_split(episodes, ks),
    }
    return result


def _by_split(episodes: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task"].split].append(episode)
    return {
        split: {
            "tasks": len(items),
            **{
                f"pass@{k}": round(
                    sum(any(attempt["score"].valid for attempt in item["attempts"][:k]) for item in items)
                    / len(items),
                    6,
                )
                for k in ks
            },
        }
        for split, items in sorted(grouped.items())
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(statistics.fmean(items), 6) if items else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a 95% Wilson score interval for a binomial success rate."""
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--stocks-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--k", nargs="+", type=int, default=(1, 8))
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(k < 1 for k in args.k):
        parser.error("k values must be positive")
    result = evaluate(
        TaskStore(args.tasks_dir, args.stocks_dir),
        _read_jsonl(args.predictions),
        ks=tuple(args.k),
        splits=tuple(args.splits) if args.splits else None,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
