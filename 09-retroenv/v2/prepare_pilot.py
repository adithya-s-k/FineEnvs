#!/usr/bin/env python3
"""Freeze the three-task public/private v2 pilot inputs."""

from __future__ import annotations

import json
from pathlib import Path

from retroenv.store import TaskStore

from v2.run_pilot import BENCHMARK, ROOT, TASK_INDICES, TASK_LABELS


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    store = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
    all_eval = store.tasks("eval")
    tasks = [all_eval[index] for index in TASK_INDICES]
    output = ROOT / "v2" / "tasks"
    private_rows = [task.to_dict(include_references=True) for task in tasks]
    public_rows = []
    for task, label in zip(tasks, TASK_LABELS, strict=True):
        row = task.to_dict(include_references=False)
        row["pilot_label"] = label
        public_rows.append(row)
    _write_jsonl(output / "pilot-private.jsonl", private_rows)
    _write_jsonl(output / "pilot-public.jsonl", public_rows)
    manifest = {
        "schema_version": "retro-pilot-manifest-v2",
        "source_benchmark": str(BENCHMARK.relative_to(ROOT)),
        "source_split": "eval",
        "source_indices": list(TASK_INDICES),
        "task_ids": [task.task_id for task in tasks],
        "tasks": len(tasks),
        "references_private": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
