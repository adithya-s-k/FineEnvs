#!/usr/bin/env python3
"""Merge deterministic task-range shards and recompute one complete model report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retroenv.evaluation import evaluate
from retroenv.store import TaskStore


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--stocks-dir", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    store = TaskStore(args.tasks_dir, args.stocks_dir)
    expected_tasks = list(store.tasks(args.split))
    expected_ids = [task.task_id for task in expected_tasks]
    expected_set = set(expected_ids)
    predictions: dict[str, dict[str, Any]] = {}
    episodes: dict[tuple[str, int], dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    invariant: dict[str, Any] | None = None
    varying_keys = {"start_index", "tasks", "task_ids"}

    for shard in args.shard:
        manifest_path = shard.with_suffix(".run.json")
        episode_path = shard.with_suffix(".episodes.jsonl")
        if not manifest_path.exists() or not episode_path.exists():
            raise ValueError(f"incomplete shard sidecars for {shard}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in manifest.items() if key not in varying_keys}
        if invariant is None:
            invariant = comparable
        elif comparable != invariant:
            raise ValueError(f"shard configuration differs: {shard}")
        manifests.append(manifest)
        for row in _read_jsonl(shard):
            task_id = str(row.get("task_id", ""))
            if task_id in predictions:
                raise ValueError(f"duplicate task prediction across shards: {task_id}")
            predictions[task_id] = row
        for row in _read_jsonl(episode_path):
            key = (str(row.get("task_id", "")), int(row.get("sample_index", 0)))
            if key in episodes:
                raise ValueError(f"duplicate episode across shards: {key}")
            episodes[key] = row

    missing = sorted(expected_set - set(predictions))
    extra = sorted(set(predictions) - expected_set)
    if missing or extra:
        raise ValueError(
            f"shards do not exactly cover {args.split}: missing={missing[:3]}, extra={extra[:3]}"
        )
    prediction_rows = [predictions[task_id] for task_id in expected_ids]
    episode_rows = sorted(
        episodes.values(),
        key=lambda row: (expected_ids.index(row["task_id"]), int(row.get("sample_index", 0))),
    )
    _write_jsonl(args.output, prediction_rows)
    _write_jsonl(args.output.with_suffix(".episodes.jsonl"), episode_rows)

    merged_manifest = {
        **(invariant or {}),
        "start_index": 0,
        "tasks": len(expected_tasks),
        "task_ids": expected_ids,
        "merged_shards": [
            {
                "path": str(path),
                "start_index": manifest["start_index"],
                "tasks": manifest["tasks"],
            }
            for path, manifest in zip(args.shard, manifests)
        ],
    }
    args.output.with_suffix(".run.json").write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = evaluate(store, prediction_rows, ks=(1,), splits=(args.split,))
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
