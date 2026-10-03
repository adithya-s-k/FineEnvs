#!/usr/bin/env python3
"""Build the static, public-safe data bundle for the RetroEval web app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "retroeval-v1"
RUNS = BENCHMARK / "model-runs" / "board-v1"
PUBLIC_TASKS = BENCHMARK / "tasks-public" / "eval.jsonl"
BOARD = RUNS / "board.json"
OUTPUT = Path(__file__).with_name("data.js")

STEMS = {
    "anthropic/claude-opus-5.5": "opus",
    "anthropic/claude-sonnet-5": "sonnet",
    "qwen/qwen3.8-max-0902": "qwenmax",
    "deepseek/deepseek-v4-pro-0813": "pro",
    "openai/gpt-5.6-sol": "sol",
    "deepseek/deepseek-v4.1-flash": "flash",
    "openai/gpt-5.6-luna": "luna",
    "qwen/qwen3.8-27b": "qwen",
}

PROMPT_TEMPLATE = """Plan retrosyntheses for the target {target} in at most {max_steps} step(s).
Return {min_routes} to {max_routes} molecule/reaction trees by calling emit_routes
exactly once. The stock is not in this prompt: stock_retrieve is its only access
surface and a leaf may claim in_stock=true only after an exact lookup hit.

Each tree must alternate molecule -> reaction -> molecule. A molecule has
{"type":"mol","smiles":"...","in_stock":false,"children":[]}. A reaction has
type="reaction", is_reaction=true, metadata (explanation, reaction_class,
confidence as a numeric value from 0 to 1, literature as a list, precursor_roles
as an object), and molecule children. Distinct routes
must differ at the first cut. Every item in submission.routes must be the root
molecule object directly: never wrap it in route, root, tree, or route_id keys.
Validate proposed cuts before emitting. Dataset
support is not proof that a reaction will work experimentally."""

SYSTEM_PROMPT = """You are a retrosynthesis planning agent. Use tools, validate cuts, confirm every stock leaf by exact lookup, and finish only with emit_routes. You have at most 16 model turns and must reserve the final turn for emit_routes even if the routes are incomplete. Do not reveal chain-of-thought; put short evidence-based explanations in reaction metadata."""

REWARD_PARTS = [
    {"key": "parse_validity", "label": "Parse", "weight": 0.05, "description": "The answer follows the route-graph schema."},
    {"key": "molecule_validity", "label": "Molecules", "weight": 0.10, "description": "Every submitted SMILES parses into a valid structure."},
    {"key": "graph_validity", "label": "Graph", "weight": 0.10, "description": "Trees alternate molecule and reaction nodes and remain connected."},
    {"key": "step_correctness", "label": "Steps", "weight": 0.20, "description": "Cuts are supported and conserve chemical structure."},
    {"key": "stock_correctness", "label": "Stock", "weight": 0.10, "description": "Terminal leaves are available and stock claims are honest."},
    {"key": "reference_similarity", "label": "Similarity", "weight": 0.10, "description": "Best overlap with any hidden reference route."},
    {"key": "exact_route_match", "label": "Exact route", "weight": 0.10, "description": "At least one submitted route exactly matches a reference."},
    {"key": "verified_route_diversity", "label": "Diversity", "weight": 0.15, "description": "Verified routes use distinct first disconnections."},
    {"key": "route_set_compliance", "label": "Route count", "weight": 0.10, "description": "The answer returns the requested number of route trees."},
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_result(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"summary": str(raw)[:300]}
    if not isinstance(value, dict):
        return {"summary": str(value)[:300]}

    keep = (
        "valid",
        "support",
        "returned",
        "truncated",
        "formula",
        "molecular_weight",
        "rings",
        "formal_charge",
        "reaction_class",
        "classification",
        "confidence",
        "matched_reaction_id",
        "in_stock",
        "reward",
        "verification_tier",
        "done",
        "error",
        "errors",
        "model_turns_remaining",
    )
    result = {key: value[key] for key in keep if key in value}
    if isinstance(value.get("checks"), dict):
        result["checks"] = value["checks"]
    if isinstance(value.get("hard_failures"), list):
        result["hard_failures"] = value["hard_failures"][:4]
    if isinstance(value.get("results"), list):
        result["results"] = value["results"][:3]
    if isinstance(value.get("conditions"), list):
        result["conditions"] = value["conditions"][:3]
    if isinstance(value.get("literature"), list):
        result["literature"] = value["literature"][:3]
    if isinstance(value.get("metrics"), dict):
        result["metrics"] = {
            key: value["metrics"][key]
            for key in ("route_count", "valid_routes", "exact_reference_match")
            if key in value["metrics"]
        }
    return result


def compact_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"raw": (raw or "")[:500]}
    if not isinstance(value, dict):
        return {"raw": str(value)[:500]}
    # emit_routes already appears as the final graph, so avoid duplicating it.
    if "submission" in value:
        submission = value.get("submission") or {}
        if isinstance(submission, str):
            try:
                submission = json.loads(submission)
            except json.JSONDecodeError:
                submission = {}
        if not isinstance(submission, dict):
            submission = {}
        return {"route_count": len(submission.get("routes") or [])}
    return value


def compact_transcript(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("role") == "tool" and row.get("tool_call_id"):
            tool_results[str(row["tool_call_id"])] = compact_result(row.get("content", ""))

    steps: list[dict[str, Any]] = []
    turn = 0
    for row in rows:
        if row.get("role") != "assistant":
            continue
        calls = row.get("tool_calls") or []
        if not calls:
            content = (row.get("content") or "").strip()
            if content:
                turn += 1
                steps.append({"turn": turn, "tool": "reasoning", "note": content[:360], "arguments": {}, "result": {}})
            continue
        for call in calls:
            turn += 1
            function = call.get("function") or {}
            call_id = str(call.get("id") or "")
            steps.append(
                {
                    "turn": turn,
                    "tool": function.get("name") or "unknown_tool",
                    "note": (row.get("content") or "").strip()[:360],
                    "arguments": compact_arguments(function.get("arguments") or "{}"),
                    "result": tool_results.get(call_id, {}),
                }
            )
    return steps


def compact_episode(row: dict[str, Any]) -> dict[str, Any]:
    score = row.get("online_score") or {}
    metrics = score.get("metrics") or {}
    usage = row.get("usage") or {}
    submission = row.get("submission") or {}
    if isinstance(submission, str):
        try:
            submission = json.loads(submission)
        except json.JSONDecodeError:
            submission = {}
    if not isinstance(submission, dict):
        submission = {}
    return {
        "task_id": row["task_id"],
        "valid": bool(score.get("valid")),
        "verification_tier": score.get("verification_tier", "rejected"),
        "reward": score.get("reward", 0),
        "components": score.get("components") or {},
        "metrics": {
            key: metrics.get(key)
            for key in (
                "parse_valid",
                "molecule_validity",
                "graph_validity",
                "step_validity",
                "building_block_completion",
                "reference_similarity",
                "exact_reference_match",
                "verified_route_diversity",
                "route_count",
                "valid_routes",
                "route_count_in_bounds",
            )
        },
        "route_results": metrics.get("route_results") or [],
        "hard_failures": score.get("hard_failures") or [],
        "tool_calls": row.get("tool_calls", 0),
        "invalid_proposals": row.get("invalid_proposals", 0),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": usage.get("reported_cost_usd", 0),
            "latency": usage.get("latency_seconds", 0),
        },
        "errors": row.get("errors") or [],
        "trajectory": compact_transcript(row.get("transcript") or []),
        "routes": submission.get("routes") or [],
    }


def main() -> int:
    tasks = read_jsonl(PUBLIC_TASKS)
    board = json.loads(BOARD.read_text(encoding="utf-8"))
    models: list[dict[str, Any]] = []
    for rank, board_row in enumerate(board["models"], 1):
        model_id = board_row["model"]
        stem = STEMS[model_id]
        episodes = read_jsonl(RUNS / f"{stem}.episodes.jsonl")
        run = json.loads((RUNS / f"{stem}.run.json").read_text(encoding="utf-8"))
        models.append(
            {
                "key": stem,
                "rank": rank,
                "label": board_row["label"],
                "model": model_id,
                "tool_choice": run["tool_choice"],
                "scores": board_row,
                "episodes": {row["task_id"]: compact_episode(row) for row in episodes},
            }
        )

    bundle = {
        "meta": {
            "benchmark": "RetroEval v1",
            "task_count": len(tasks),
            "model_count": len(models),
            "prompt_template": PROMPT_TEMPLATE,
            "system_prompt": SYSTEM_PROMPT,
            "reward_parts": REWARD_PARTS,
            "hard_gates": [
                "Valid route-graph JSON",
                "Connected route rooted at the target",
                "Supported reaction steps",
                "All terminal molecules found in stock",
                "Requested route count and distinct first cuts",
            ],
        },
        "tasks": tasks,
        "models": models,
    }
    serialized = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    OUTPUT.write_text("window.RETRO_DATA=" + serialized + ";\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
