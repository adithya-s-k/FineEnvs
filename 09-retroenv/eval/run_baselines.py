#!/usr/bin/env python3
"""Materialize deterministic sanity baselines and score them offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retroenv.evaluation import evaluate
from retroenv.graph import routes_to_submission
from retroenv.store import TaskStore


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=Path("sample/tasks-private"))
    parser.add_argument("--stocks-dir", type=Path, default=Path("sample/stocks"))
    parser.add_argument("--output-dir", type=Path, default=Path("sample/baselines"))
    args = parser.parse_args()

    store = TaskStore(args.tasks_dir, args.stocks_dir)
    rows: dict[str, list[dict]] = {
        "oracle_ceiling": [],
        "one_route_ablation": [],
        "empty_graph_floor": [],
    }
    for task in store.iter_all():
        stock = store.stock(task.stock_id)
        references = task.reference_routes[: task.max_routes]
        rows["oracle_ceiling"].append(
            {
                "task_id": task.task_id,
                "attempts": [
                    {
                        "submission": routes_to_submission(
                            task.target_smiles,
                            references,
                            stock,
                            source="private-reference-oracle",
                        )
                    }
                ],
            }
        )
        rows["one_route_ablation"].append(
            {
                "task_id": task.task_id,
                "attempts": [
                    {
                        "submission": routes_to_submission(
                            task.target_smiles,
                            references[:1],
                            stock,
                            source="one-reference-ablation",
                        )
                    }
                ],
            }
        )
        rows["empty_graph_floor"].append(
            {
                "task_id": task.task_id,
                "attempts": [{"submission": {"routes": []}}],
            }
        )

    reports = {}
    for name, prediction_rows in rows.items():
        path = args.output_dir / f"{name}.jsonl"
        _write_jsonl(path, prediction_rows)
        reports[name] = evaluate(store, prediction_rows, ks=(1,))
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            key: report[key]
            for key in (
                "pass@1",
                "top1_reward",
                "top1_route_validity",
                "top1_graph_validity",
                "top1_step_validity",
                "top1_verified_route_diversity",
            )
        }
        for name, report in reports.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
