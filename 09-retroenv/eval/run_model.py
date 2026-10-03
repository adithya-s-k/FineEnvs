#!/usr/bin/env python3
"""Run an OpenAI-compatible tool-calling model against RetroEnv tasks.

This is an evaluation/smoke harness, not the trainer. It drives the same pure
session used by the MCP server, records raw episodes, and recomputes metrics
offline from private tasks rather than trusting model- or endpoint-supplied
scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from retroenv.environment import RetroRouteSession
from retroenv.evaluation import evaluate
from retroenv.retrieval import PrecedentIndex
from retroenv.store import TaskStore


def _function(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
    defs: dict[str, Any] | None = None,
) -> dict:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
    if defs:
        parameters["$defs"] = defs
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


SMILES = {"type": "string", "description": "A SMILES string."}
REACTANTS = {"type": "array", "items": SMILES, "minItems": 1}
ROUTE_GRAPH_DEFS: dict[str, Any] = {
    "metadata": {
        "type": "object",
        "description": "Evidence and confidence for this disconnection.",
        "properties": {
            "source": {"type": "string"},
            "explanation": {"type": "string"},
            "reaction_class": {"type": "string"},
            "classification": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "policy_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "literature": {"type": "array", "items": {"type": "object"}},
            "conditions": {"type": "array", "items": {"type": "object"}},
            "precursor_roles": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": [
            "explanation",
            "reaction_class",
            "confidence",
            "literature",
            "precursor_roles",
        ],
        "additionalProperties": True,
    },
    "reaction": {
        "type": "object",
        "description": "A reaction whose children are its precursor molecules.",
        "properties": {
            "type": {"const": "reaction"},
            "is_reaction": {"const": True},
            "metadata": {"$ref": "#/$defs/metadata"},
            "children": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/mol"},
            },
        },
        "required": ["type", "is_reaction", "metadata", "children"],
        "additionalProperties": False,
    },
    "mol": {
        "type": "object",
        "description": "A molecule node. Expanded nodes have one reaction child and in_stock=false; leaves have no children.",
        "properties": {
            "type": {"const": "mol"},
            "smiles": {"type": "string", "minLength": 1},
            "in_stock": {"type": "boolean"},
            "children": {
                "type": "array",
                "maxItems": 1,
                "items": {"$ref": "#/$defs/reaction"},
            },
        },
        "required": ["type", "smiles", "in_stock", "children"],
        "additionalProperties": False,
    },
}
TOOLS = [
    _function("inspect_molecule", "Inspect a molecule with RDKit.", {"smiles": SMILES}, ["smiles"]),
    _function("pubchem_lookup", "Canonicalize a SMILES or query the frozen molecule cache.", {"query": {"type": "string"}}, ["query"]),
    _function(
        "stock_retrieve",
        "The only stock access. Search exact SMILES/InChIKey, class, SMARTS, or similarity; at most 20 results.",
        {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["auto", "exact", "inchikey", "class", "substructure", "similarity"], "default": "auto"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
        ["query"],
    ),
    _function(
        "reaction_precedent_search",
        "Find reaction analogues from training-visible records.",
        {"product_smiles": SMILES, "reaction_class": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10}},
    ),
    _function(
        "validate_disconnection",
        "Validate one proposed product-to-reactants cut against hidden evidence.",
        {"product_smiles": SMILES, "reactants": REACTANTS, "reaction_class": {"type": "string"}},
        ["product_smiles", "reactants"],
    ),
    _function(
        "reaction_class_lookup",
        "Name the class of an agent-supplied supported cut.",
        {"product_smiles": SMILES, "reactants": REACTANTS},
        ["product_smiles", "reactants"],
    ),
    _function(
        "reaction_conditions_search",
        "Find frozen reported conditions for a cut or analogue.",
        {"product_smiles": SMILES, "reactants": REACTANTS, "reaction_class": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}},
    ),
    _function(
        "search_literature",
        "Search frozen citation metadata attached to training precedents.",
        {"product_smiles": SMILES, "reaction_class": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}},
    ),
    _function(
        "emit_routes",
        "Terminal call. Emit the required renderable molecule/reaction trees.",
        {
            "submission": {
                "type": "object",
                "properties": {
                    "schema_version": {"type": "string"},
                    "routes": {
                        "type": "array",
                        "description": "Root molecule nodes directly; do not wrap them in route/root/tree objects.",
                        "items": {"$ref": "#/$defs/mol"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
                "required": ["routes"],
            }
        },
        ["submission"],
        defs=ROUTE_GRAPH_DEFS,
    ),
]
EMIT_TOOL = next(tool for tool in TOOLS if tool["function"]["name"] == "emit_routes")


def _dispatch(session: RetroRouteSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in session.TOOL_NAMES:
        return {"error": f"unknown tool {name!r}"}
    function = getattr(session, name)
    try:
        return function(**arguments)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def rollout(
    client: Any,
    model: str,
    session: RetroRouteSession,
    opening: dict[str, Any],
    max_turns: int,
    temperature: float,
    tool_choice: str = "required",
    max_empty_turns: int = 2,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": f"You are a retrosynthesis planning agent. Use tools, validate cuts, confirm every stock leaf by exact lookup, and finish only with emit_routes. You have at most {max_turns} model turns and must reserve the final turn for emit_routes even if the routes are incomplete. Do not reveal chain-of-thought; put short evidence-based explanations in reaction metadata.",
        },
        {"role": "user", "content": opening["prompt"]},
    ]
    transcript: list[dict[str, Any]] = []
    emitted: Any = None
    errors: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    reported_cost = 0.0
    resolved_models: set[str] = set()
    empty_turns = 0
    force_terminal = False
    started = time.perf_counter()
    for turn_index in range(max_turns):
        # A benchmark episode must always end in a scoreable graph attempt.
        # Restricting the final model turn is provider-independent and avoids
        # silently converting a long, otherwise useful rollout into an empty
        # submission merely because the model forgot to call emit_routes.
        terminal_turn = force_terminal or turn_index == max_turns - 1
        available_tools = [EMIT_TOOL] if terminal_turn else TOOLS
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=available_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                temperature=temperature,
                max_tokens=4096,
                extra_body={"usage": {"include": True}},
            )
        except Exception as exc:
            errors.append(f"API error: {type(exc).__name__}: {exc}")
            break
        if response.usage:
            prompt_tokens += int(response.usage.prompt_tokens or 0)
            completion_tokens += int(response.usage.completion_tokens or 0)
            extra = response.usage.model_extra or {}
            reported_cost += float(extra.get("cost") or 0.0)
        if response.model:
            resolved_models.add(str(response.model))
        if not response.choices:
            errors.append(
                "API returned no choices: "
                + json.dumps(response.model_dump(exclude_none=True), sort_keys=True)[:1000]
            )
            break
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            empty_turns += 1
            errors.append(
                f"no tool call on turn {turn_index + 1}: {(message.content or '')[:200]}"
            )
            assistant = {"role": "assistant", "content": message.content or ""}
            messages.append(assistant)
            transcript.append(assistant)
            if terminal_turn:
                break
            if empty_turns >= max_empty_turns:
                force_terminal = True
                recovery = {
                    "role": "user",
                    "content": (
                        "The empty-turn limit was reached. On the next turn call emit_routes "
                        "with the best route trees available; no other tool will be exposed."
                    ),
                }
            else:
                recovery = {
                    "role": "user",
                    "content": (
                        "No tool call was emitted. Continue by calling one of the available "
                        "tools now; on your final turn you must call emit_routes."
                    ),
                }
            messages.append(recovery)
            transcript.append(recovery)
            continue
        assistant = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump() for call in calls],
        }
        messages.append(assistant)
        transcript.append(assistant)
        for call in calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"invalid tool arguments: {exc}"}
                errors.append(result["error"])
            else:
                result = _dispatch(session, call.function.name, arguments)
                if call.function.name == "emit_routes":
                    emitted = arguments.get("submission")
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(
                    {
                        **result,
                        "model_turns_remaining": max_turns - len(
                            [item for item in transcript if item.get("role") == "assistant"]
                        ),
                    },
                    sort_keys=True,
                ),
            }
            messages.append(tool_message)
            transcript.append(tool_message)
        if session.done:
            break
    if emitted is None:
        emitted = {"routes": []}
    return {
        "submission": emitted,
        "tool_calls": session.tool_calls,
        "invalid_proposals": sum(
            not row.get("valid", False) for row in session.validations
        ),
        "online_score": session.final_score,
        "errors": errors,
        "transcript": transcript,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "reported_cost_usd": round(reported_cost, 8),
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
        "resolved_models": sorted(resolved_models),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tasks-dir", type=Path, default=Path("sample/tasks-private"))
    parser.add_argument("--stocks-dir", type=Path, default=Path("sample/stocks"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="required")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-empty-turns",
        type=int,
        default=2,
        help="After this many empty assistant turns, expose one terminal emit-only turn.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed attempts from output and its episode sidecar.",
    )
    parser.add_argument(
        "--max-reported-cost-usd",
        type=float,
        help="Stop scheduling new attempts once provider-reported cost reaches this cap.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.attempts < 1:
        parser.error("attempts must be positive")
    if args.max_turns < 1:
        parser.error("max-turns must be positive")
    if args.request_timeout <= 0:
        parser.error("request-timeout must be positive")
    if args.max_empty_turns < 1:
        parser.error("max-empty-turns must be positive")
    if args.max_reported_cost_usd is not None and args.max_reported_cost_usd <= 0:
        parser.error("max-reported-cost-usd must be positive")
    episode_path = args.output.with_suffix(".episodes.jsonl")
    run_path = args.output.with_suffix(".run.json")
    if (args.output.exists() or episode_path.exists() or run_path.exists()) and not args.resume:
        parser.error("output already exists; pass --resume or choose a new path")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("install the eval extra: uv sync --extra eval") from exc
    client = OpenAI(
        base_url=args.endpoint,
        api_key=os.getenv(args.api_key_env) or "dummy",
        timeout=args.request_timeout,
        max_retries=2,
    )
    store = TaskStore(args.tasks_dir, args.stocks_dir)
    tasks = store.tasks(args.split)[args.start_index : args.start_index + args.limit]
    if not tasks:
        parser.error("selected task range is empty")
    precedent_index = PrecedentIndex(store.tasks("train"))
    stock_ids = sorted({task.stock_id for task in tasks})
    run_config = {
        "schema_version": "retro-model-run-v1",
        "requested_model": args.model,
        "endpoint": args.endpoint,
        "split": args.split,
        "start_index": args.start_index,
        "tasks": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "attempts_per_task": args.attempts,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "tool_choice": args.tool_choice,
        "request_timeout_seconds": args.request_timeout,
        "max_empty_turns": args.max_empty_turns,
        "tasks_sha256": _sha256(args.tasks_dir / f"{args.split}.jsonl"),
        "stocks_sha256": {
            stock_id: _sha256(args.stocks_dir / f"{stock_id}.smi")
            for stock_id in stock_ids
        },
        "tool_schema_sha256": hashlib.sha256(
            json.dumps(TOOLS, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if args.resume and run_path.exists():
        previous_config = json.loads(run_path.read_text(encoding="utf-8"))
        if previous_config != run_config:
            raise ValueError("resume configuration differs from the recorded run manifest")
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prior_predictions = _read_jsonl(args.output) if args.resume else []
    prediction_by_task: dict[str, dict[str, Any]] = {}
    for row in prior_predictions:
        task_id = str(row.get("task_id", ""))
        if task_id in prediction_by_task:
            raise ValueError(f"duplicate task in resume file: {task_id}")
        prediction_by_task[task_id] = row
    selected_ids = {task.task_id for task in tasks}
    unknown_ids = sorted(set(prediction_by_task) - selected_ids)
    if unknown_ids:
        raise ValueError(f"resume file contains tasks outside this run: {unknown_ids[:3]}")
    episode_rows = _read_jsonl(episode_path) if args.resume else []
    reported_cost = sum(
        float((row.get("usage") or {}).get("reported_cost_usd") or 0.0)
        for row in episode_rows
    )
    stopped_for_cost = False
    for task_index, task in enumerate(tasks):
        row = prediction_by_task.setdefault(task.task_id, {"task_id": task.task_id, "attempts": []})
        attempts = row.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"{task.task_id}: resume attempts must be a list")
        if len(attempts) > args.attempts:
            raise ValueError(
                f"{task.task_id}: resume file has {len(attempts)} attempts, "
                f"but --attempts={args.attempts}"
            )
        for sample_index in range(len(attempts), args.attempts):
            if (
                args.max_reported_cost_usd is not None
                and reported_cost >= args.max_reported_cost_usd
            ):
                stopped_for_cost = True
                break
            session = RetroRouteSession(precedent_index=precedent_index)
            opening = session.reset(
                task,
                store.stock(task.stock_id),
                episode_id=f"{args.model}:{task.task_id}:{sample_index}",
            )
            attempt = rollout(
                client,
                args.model,
                session,
                opening,
                args.max_turns,
                args.temperature,
                args.tool_choice,
                args.max_empty_turns,
            )
            attempts.append(
                {
                    key: attempt[key]
                    for key in ("submission", "tool_calls", "invalid_proposals")
                }
            )
            episode_rows.append(
                {
                    "task_id": task.task_id,
                    "requested_model": args.model,
                    "split": args.split,
                    "task_index": task_index,
                    "sample_index": sample_index,
                    **attempt,
                }
            )
            reported_cost += float(attempt["usage"]["reported_cost_usd"] or 0.0)
            ordered_predictions = [
                prediction_by_task[item.task_id]
                for item in tasks
                if item.task_id in prediction_by_task
            ]
            _atomic_write_jsonl(args.output, ordered_predictions)
            _atomic_write_jsonl(episode_path, episode_rows)
        if stopped_for_cost:
            break

    prediction_rows = [
        prediction_by_task[task.task_id]
        for task in tasks
        if task.task_id in prediction_by_task
    ]
    _atomic_write_jsonl(args.output, prediction_rows)
    _atomic_write_jsonl(episode_path, episode_rows)
    report = evaluate(
        store,
        prediction_rows,
        ks=tuple(sorted({1, args.attempts})),
        splits=(args.split,),
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if stopped_for_cost:
        print(
            f"Stopped before scheduling another attempt: provider-reported cost "
            f"${reported_cost:.4f} reached the configured cap.",
            file=sys.stderr,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
