#!/usr/bin/env python3
"""Build a comparable model-board table from completed RetroEnv run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("eval/benchmark_models.json"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = {row["id"]: row["label"] for row in config["models"]}
    expected_tasks = int(config["run"]["tasks"])
    rows: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    task_sequences: set[tuple[str, ...]] = set()
    failures: list[str] = []
    for run_path in sorted(args.runs_dir.glob("*.run.json")):
        stem = run_path.name.removesuffix(".run.json")
        report_path = args.runs_dir / f"{stem}.report.json"
        episode_path = args.runs_dir / f"{stem}.episodes.jsonl"
        if not report_path.exists():
            failures.append(f"{stem}: missing report")
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episodes = _read_jsonl(episode_path)
        task_sequences.add(tuple(run["task_ids"]))
        predicted = int(report["tasks_with_predictions"])
        if predicted != expected_tasks:
            failures.append(f"{stem}: only {predicted}/{expected_tasks} tasks predicted")
        usage_cost = sum(
            float((row.get("usage") or {}).get("reported_cost_usd") or 0.0)
            for row in episodes
        )
        latency = sum(
            float((row.get("usage") or {}).get("latency_seconds") or 0.0)
            for row in episodes
        )
        model = run["requested_model"]
        if model in seen_models:
            failures.append(f"duplicate completed run for model: {model}")
        seen_models.add(model)
        rows.append(
            {
                "label": labels.get(model, model),
                "model": model,
                "tasks": int(report["tasks"]),
                "tasks_with_predictions": predicted,
                "pass@1": report["pass@1"],
                "pass@1_ci95": report.get("pass_ci95", {}).get("pass@1", [0.0, 0.0]),
                "mean_reward": report["top1_reward"],
                "route_validity": report["top1_route_validity"],
                "graph_validity": report["top1_graph_validity"],
                "step_validity": report["top1_step_validity"],
                "stock_completion": report["top1_building_block_completion"],
                "exact_reference_match": report["top1_exact_reference_match"],
                "route_diversity": report["top1_verified_route_diversity"],
                "invalid_proposal_rate": report["invalid_proposal_rate"],
                "reported_cost_usd": round(usage_cost, 6),
                "latency_seconds": round(latency, 3),
            }
        )
    if len(task_sequences) > 1:
        failures.append("run manifests use different task sequences")
    missing_models = [
        row["id"] for row in config["models"] if row["id"] not in seen_models
    ]
    if missing_models:
        failures.append("missing configured model runs: " + ", ".join(missing_models))
    rows.sort(key=lambda row: (-row["pass@1"], -row["mean_reward"], row["label"]))
    result = {
        "schema_version": "retro-model-board-results-v1",
        "expected_tasks": expected_tasks,
        "complete": not failures,
        "failures": failures,
        "models": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")

    header = (
        "| Model | Tasks | Pass@1 (95% CI) | Any exact route | Reward | Graph | "
        "Steps | Stock | Diversity | Cost | Latency |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        low, high = row["pass@1_ci95"]
        lines.append(
            f"| {row['label']} | {row['tasks_with_predictions']}/{row['tasks']} | "
            f"{row['pass@1']:.3f} [{low:.3f}, {high:.3f}] | "
            f"{row['exact_reference_match']:.3f} | "
            f"{row['mean_reward']:.3f} | {row['graph_validity']:.3f} | "
            f"{row['step_validity']:.3f} | {row['stock_completion']:.3f} | "
            f"{row['route_diversity']:.3f} | ${row['reported_cost_usd']:.3f} | "
            f"{row['latency_seconds']:.1f}s |\n"
        )
    markdown = "".join(lines)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if args.require_complete and failures else 0


if __name__ == "__main__":
    sys.exit(main())
