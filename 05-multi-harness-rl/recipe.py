"""Shared configuration, task identity and reward policy."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
MODES = ("whitebox", "opencode", "multi-harness")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def merge_config(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def config(path=None, **overrides):
    result = json.loads((ROOT / "configs/default.json").read_text())
    if path:
        merge_config(result, json.loads(Path(path).read_text()))
    result.update({k: v for k, v in overrides.items() if v is not None})
    if result["mode"] not in MODES or result["model"] not in result["models"]:
        raise ValueError("Unknown mode or model")
    for key in ("passes", "num_generations", "max_steps", "save_steps", "eval_steps", "eval_concurrency",
                "batch_size", "gradient_accumulation_steps", "token_budget", "max_output_tokens"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if result["num_generations"] < 2:
        raise ValueError("GRPO needs multiple rollouts of each task")
    if result["max_inflight"] < result["num_generations"]:
        raise ValueError("Inflight capacity must fit a complete group")
    if result["max_outstanding_rollouts"] < 2 * result["num_generations"]:
        raise ValueError("Outstanding capacity must fit two complete groups")
    if result["eval_steps"] % result["save_steps"]:
        raise ValueError("Evaluation cadence must be a multiple of checkpoint cadence")
    if result["top_p"] != 1.0 or result["top_k"] != 0:
        raise ValueError("Training requires full-vocabulary sampling for logprob consistency")
    if not 0 <= result["efficiency_weight"] <= 0.1:
        raise ValueError("Efficiency weight must be between zero and 0.1")
    result["profile"] = result["models"][result["model"]]
    return result


def whitebox_step_limit(cfg, task_count):
    rollouts_per_update = cfg["batch_size"] * cfg["gradient_accumulation_steps"]
    if rollouts_per_update % cfg["num_generations"]:
        raise ValueError("Whitebox update must contain complete GRPO groups")
    tasks_per_update = rollouts_per_update // cfg["num_generations"]
    available = task_count // tasks_per_update
    if not available:
        raise ValueError("Whitebox schedule is smaller than one update")
    return min(cfg["max_steps"], available)


def task_rows(split):
    rows = json.loads((ROOT / "data" / f"{split}.json").read_text())
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task identity")
    return rows


def eval_rows(rows, limit=None, *, stratified=False, seed=0):
    rows = sorted(rows, key=lambda row: row["name"])
    if limit is None:
        return rows
    if not 1 <= limit <= len(rows):
        raise ValueError("Eval limit must fit the fixed test set")
    if not stratified:
        return rows[:limit]
    strata = {d: [r for r in rows if r["difficulty"] == d] for d in sorted({r["difficulty"] for r in rows})}
    exact = {d: limit * len(group) / len(rows) for d, group in strata.items()}
    counts = {d: int(n) for d, n in exact.items()}
    for d in sorted(strata, key=lambda d: (-(exact[d] - counts[d]), d))[:limit - sum(counts.values())]:
        counts[d] += 1
    rng = random.Random(seed)
    selected = []
    for d, group in strata.items():
        selected.extend(rng.sample(group, counts[d]))
    return sorted(selected, key=lambda row: row["name"])


def schedule(cfg, rows):
    catalog = {name: i for i, name in enumerate(sorted(r["name"] for r in rows))}
    harnesses = cfg["harnesses"] if cfg["mode"] == "multi-harness" else [cfg["mode"]]
    groups = []
    for epoch in range(cfg["passes"]):
        for i, row in enumerate(rows):
            groups.append({"group_id": len(groups), "pass": epoch, "task_name": row["name"],
                           "task_index": catalog[row["name"]],
                           "harness": harnesses[(i + epoch) % len(harnesses)]})
    return groups


def resume_state(groups, state):
    if state["execution_schedule_sha256"] != digest(groups):
        raise ValueError("Cannot resume with a different task/harness schedule")
    committed = set(state.get("admitted_rollouts", {}).values())
    if any(type(g) is not int or not 0 <= g < len(groups) for g in committed):
        raise ValueError("Invalid committed group")
    return {**state, "settled_groups": sorted(committed)}


def reward(correctness, calls, *, verified, weight=0.1, budget=15):
    if correctness is None:
        return None
    if correctness not in (0, 1):
        raise ValueError("Expected binary correctness")
    if not 0 <= weight <= 0.1 or budget <= 0:
        raise ValueError("Invalid reward parameters")
    if not verified or calls is None:
        return float(correctness)
    if type(calls) is not int or calls < 0:
        raise ValueError("Invalid verified tool count")
    return float(correctness) * (1 + weight * budget / (budget + calls)) if calls else float(correctness)


def summary(records, expected):
    graded = [r for r in records if r.get("correctness") in (0, 1) and not r.get("error")]
    successes = sum(r["correctness"] for r in graded)
    return {"expected": expected, "graded": len(graded), "correct": successes,
            "pass_at_1": successes / len(graded) if graded else None,
            "coverage": len(graded) / expected if expected else 0,
            "complete": len(graded) == expected,
            "mean_shaped_reward": mean(r.get("reward") for r in graded),
            "mean_tool_calls": mean(r.get("tool_calls") for r in graded),
            "mean_generated_tokens": mean(r.get("generated_tokens") for r in graded)}


def mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(values) / len(values) if values else None
