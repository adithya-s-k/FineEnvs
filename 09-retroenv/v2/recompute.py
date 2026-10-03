#!/usr/bin/env python3
"""Recompute v2 graphs and scores from saved raw emit_graph calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retroenv.store import TaskStore

from v2.run_pilot import BENCHMARK, ROOT, normalize_graph_argument, write_jsonl
from v2.verifier import verify_graph


def _last_emission(transcript: list[dict[str, Any]]) -> Any:
    for message in reversed(transcript):
        if message.get("role") != "assistant":
            continue
        for call in reversed(message.get("tool_calls") or []):
            function = call.get("function") or {}
            if function.get("name") != "emit_graph":
                continue
            arguments = json.loads(function.get("arguments") or "{}")
            return arguments.get("graph")
    raise ValueError("episode has no emit_graph call")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=ROOT / "v2" / "runs" / "opus-v2.episodes.jsonl",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    store = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
    tasks = {task.task_id: task for task in store.tasks("eval")}
    for row in rows:
        task = tasks[row["task_id"]]
        try:
            graph, parse_quality, warnings = normalize_graph_argument(
                _last_emission(row.get("transcript") or [])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            graph, parse_quality, warnings = {}, 0.0, [f"invalid emitted graph: {exc}"]
        row["graph"] = graph
        row["normalization_warnings"] = warnings
        row["score"] = verify_graph(
            task,
            graph,
            store.stock(task.stock_id),
            row.get("evidence") or [],
            parse_quality=parse_quality,
            parse_errors=warnings,
        )
    write_jsonl(args.path, rows)
    for row in rows:
        print(
            json.dumps(
                {
                    "task": row["task_label"],
                    "reward": row["score"]["reward"],
                    "valid": row["score"]["valid"],
                    "warnings": row.get("normalization_warnings") or [],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
