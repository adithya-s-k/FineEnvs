#!/usr/bin/env python3
"""Project board cost from measured RetroEnv episode token counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("eval/benchmark_models.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    task_count = int(config["run"]["tasks"])
    attempts = int(config["run"]["attempts"])
    rows: list[dict[str, Any]] = []
    for model in config["models"]:
        proxy_path = Path(model["token_proxy"])
        episodes = _read_jsonl(proxy_path)
        if not episodes:
            raise ValueError(f"token proxy contains no episodes: {proxy_path}")
        mean_prompt = sum(row["usage"]["prompt_tokens"] for row in episodes) / len(episodes)
        mean_completion = sum(
            row["usage"]["completion_tokens"] for row in episodes
        ) / len(episodes)
        episodes_planned = task_count * attempts
        estimated = episodes_planned * (
            mean_prompt * float(model["prompt_usd_per_token"])
            + mean_completion * float(model["completion_usd_per_token"])
        )
        rows.append(
            {
                "label": model["label"],
                "id": model["id"],
                "planned_episodes": episodes_planned,
                "proxy": str(proxy_path),
                "proxy_episodes": len(episodes),
                "mean_proxy_prompt_tokens": round(mean_prompt, 1),
                "mean_proxy_completion_tokens": round(mean_completion, 1),
                "estimated_cost_usd": round(estimated, 4),
            }
        )
    result = {
        "schema_version": "retro-board-cost-projection-v1",
        "method": "mean measured tokens per proxy episode times current token prices",
        "warning": "Projection only; routing, caching, reasoning tokens, and task difficulty vary.",
        "run": config["run"],
        "models": rows,
        "estimated_total_cost_usd": round(
            sum(row["estimated_cost_usd"] for row in rows), 4
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
