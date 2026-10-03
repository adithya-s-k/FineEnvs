#!/usr/bin/env python3
"""Run Claude Opus on three explicit evidence-DAG RetroEnv v2 tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from retroenv.environment import RetroRouteSession
from retroenv.retrieval import PrecedentIndex
from retroenv.store import TaskStore

from v2.verifier import verify_graph


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "retroeval-v1"
TASK_INDICES = (0, 4, 10)
TASK_LABELS = (
    "A · mixed unary/binary chemistry",
    "B · two convergent binary routes",
    "C · three-step complex target",
)


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
EVIDENCE_IDS = {"type": "array", "items": {"type": "string"}}
DAG_DEFS: dict[str, Any] = {
    "molecule_metadata": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": [
                    "target",
                    "intermediate",
                    "starting_material",
                    "reagent",
                    "catalyst",
                    "solvent",
                ],
            },
            "rationale": {"type": "string", "minLength": 1},
            "stock_status": {
                "type": "string",
                "enum": ["confirmed", "not_in_stock", "not_applicable"],
            },
            "evidence_ids": EVIDENCE_IDS,
        },
        "required": ["role", "rationale", "stock_status", "evidence_ids"],
        "additionalProperties": False,
    },
    "reaction_metadata": {
        "type": "object",
        "properties": {
            "reaction_class": {"type": "string", "minLength": 1},
            "rationale": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "evidence_ids": EVIDENCE_IDS,
        },
        "required": [
            "reaction_class",
            "rationale",
            "confidence",
            "conditions",
            "evidence_ids",
        ],
        "additionalProperties": False,
    },
    "node": {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"const": "molecule"},
                    "smiles": SMILES,
                    "metadata": {"$ref": "#/$defs/molecule_metadata"},
                },
                "required": ["id", "type", "smiles", "metadata"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"const": "reaction"},
                    "metadata": {"$ref": "#/$defs/reaction_metadata"},
                },
                "required": ["id", "type", "metadata"],
                "additionalProperties": False,
            },
        ]
    },
    "edge": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "source": {"type": "string", "minLength": 1},
            "target": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": ["reactant", "reagent", "catalyst", "solvent", "product"],
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "minLength": 1},
                    "rationale": {"type": "string", "minLength": 1},
                    "evidence_ids": EVIDENCE_IDS,
                },
                "required": ["role", "rationale", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "required": ["id", "source", "target", "type", "metadata"],
        "additionalProperties": False,
    },
    "route": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "reaction_node_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "terminal_node_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["id", "reaction_node_ids", "terminal_node_ids", "rationale"],
        "additionalProperties": False,
    },
    "graph": {
        "type": "object",
        "properties": {
            "schema_version": {"const": "retro-evidence-dag-v2"},
            "target_node_id": {"type": "string", "minLength": 1},
            "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}, "minItems": 3},
            "edges": {"type": "array", "items": {"$ref": "#/$defs/edge"}, "minItems": 2},
            "routes": {"type": "array", "items": {"$ref": "#/$defs/route"}, "minItems": 2, "maxItems": 2},
        },
        "required": ["schema_version", "target_node_id", "nodes", "edges", "routes"],
        "additionalProperties": False,
    },
}

TOOLS = [
    function_tool("inspect_molecule", "Inspect a molecule with RDKit. The response receives an evidence_id.", {"smiles": SMILES}, ["smiles"]),
    function_tool(
        "stock_retrieve",
        "Only stock access. Use mode=exact before claiming a terminal reactant is available. The response receives an evidence_id.",
        {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["auto", "exact", "inchikey", "class", "substructure", "similarity"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["query", "mode"],
    ),
    function_tool(
        "reaction_precedent_search",
        "Search training-visible reaction precedents. The response receives an evidence_id.",
        {"product_smiles": SMILES, "reaction_class": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
    ),
    function_tool(
        "validate_disconnection",
        "Validate one product/reactant cut. Cite its evidence_id on the reaction, product edge, and reactant edges.",
        {"product_smiles": SMILES, "reactants": REACTANTS, "reaction_class": {"type": "string"}},
        ["product_smiles", "reactants"],
    ),
    function_tool(
        "reaction_class_lookup",
        "Name an agent-supplied supported cut. Cite its evidence_id on the reaction node.",
        {"product_smiles": SMILES, "reactants": REACTANTS},
        ["product_smiles", "reactants"],
    ),
    function_tool(
        "reaction_conditions_search",
        "Retrieve frozen conditions for a supplied cut. Cite its evidence_id on the reaction and reagent/condition edges.",
        {"product_smiles": SMILES, "reactants": REACTANTS, "reaction_class": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
        ["product_smiles", "reactants"],
    ),
    function_tool(
        "emit_graph",
        "Terminal call. Emit one shared evidence-bearing DAG containing exactly two route views.",
        {"graph": {"$ref": "#/$defs/graph"}},
        ["graph"],
        defs=DAG_DEFS,
    ),
]
EMIT_TOOL = TOOLS[-1]

SYSTEM_PROMPT = """You are a retrosynthesis planning agent building an explicit evidence DAG.
Use concise evidence-grounded rationales, not private chain-of-thought. Every tool response has an evidence_id.
Graph direction is precursor/reagent -> reaction -> product. Atom-contributing materials use reactant edges.
Use reagent/catalyst/solvent edges only for chemically relevant condition inputs; do not turn every solvent into a route precursor.
Every node and edge needs a short rationale and appropriate evidence_ids. A reaction must cite matching validation,
class, and conditions evidence. Every terminal reactant must cite an exact stock hit. Reuse molecule nodes by SMILES.
Batch independent tool calls when useful. Finish with emit_graph containing exactly two connected routes with distinct
first disconnections."""


def task_prompt(task: Any, label: str, max_turns: int) -> str:
    return f"""{label}
Target SMILES: {task.target_smiles}
Maximum reactions per route: {task.max_steps}
Required route views: exactly 2
Maximum model turns: {max_turns}

Build one shared acyclic graph, not two nested trees. Alternative routes may converge on the same target node and
may reuse a shared intermediate molecule node. The private reference routes are not visible. Stock is available only
through stock_retrieve. First discover and validate cuts, classes, conditions, and exact terminal stock; then emit_graph.
Dataset support means evidence-backed plausibility, not experimental proof."""


def dispatch(session: RetroRouteSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in {
        "inspect_molecule",
        "stock_retrieve",
        "reaction_precedent_search",
        "validate_disconnection",
        "reaction_class_lookup",
        "reaction_conditions_search",
    }:
        return {"error": f"unknown tool {name!r}"}
    try:
        return getattr(session, name)(**arguments)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def normalize_graph_argument(value: Any) -> tuple[dict[str, Any], float, list[str]]:
    """Normalize provider tool arguments without relaxing the graph schema.

    Some OpenAI-compatible Anthropic routes serialize a nested schema object
    as a JSON string even though the outer function arguments are valid JSON.
    Preserve and parse that object; the deterministic verifier still owns all
    semantic validation.
    """

    if isinstance(value, dict):
        return value, 1.0, []
    if isinstance(value, str):
        decoder = json.JSONDecoder()
        decoded, end = decoder.raw_decode(value)
        if isinstance(decoded, dict):
            trailing = value[end:].strip()
            if not trailing:
                return decoded, 0.75, ["graph object was JSON-encoded as a string"]
            if trailing and set(trailing) == {"}"}:
                return decoded, 0.5, [
                    "graph object was JSON-encoded as a string with trailing closing braces"
                ]
    raise ValueError("emit_graph.graph must be an object or a JSON-encoded object")


def parse_graph_argument(value: Any) -> dict[str, Any]:
    """Compatibility helper returning only the normalized graph."""

    return normalize_graph_argument(value)[0]


def rollout(
    client: Any,
    model: str,
    task: Any,
    stock: frozenset[str],
    precedent_index: PrecedentIndex,
    label: str,
    max_turns: int,
) -> dict[str, Any]:
    session = RetroRouteSession(max_tool_calls=48, precedent_index=precedent_index)
    session.reset(task, stock, episode_id=f"v2:{task.task_id}")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt(task, label, max_turns)},
    ]
    transcript: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    graph: dict[str, Any] | None = None
    score: dict[str, Any] | None = None
    errors: list[str] = []
    prompt_tokens = completion_tokens = 0
    reported_cost = 0.0
    resolved_models: set[str] = set()
    empty_turns = 0
    force_terminal = False
    started = time.perf_counter()

    for turn_index in range(max_turns):
        terminal = force_terminal or turn_index == max_turns - 1
        available_tools = [EMIT_TOOL] if terminal else TOOLS
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0,
                max_tokens=8192,
                extra_body={"usage": {"include": True}},
            )
        except Exception as exc:
            errors.append(f"API error: {type(exc).__name__}: {exc}")
            break
        if response.usage:
            prompt_tokens += int(response.usage.prompt_tokens or 0)
            completion_tokens += int(response.usage.completion_tokens or 0)
            reported_cost += float((response.usage.model_extra or {}).get("cost") or 0.0)
        if response.model:
            resolved_models.add(str(response.model))
        if not response.choices:
            errors.append("API returned no choices")
            break
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            empty_turns += 1
            errors.append(f"no tool call on turn {turn_index + 1}: {(message.content or '')[:180]}")
            assistant = {"role": "assistant", "content": message.content or ""}
            messages.append(assistant)
            transcript.append(assistant)
            if terminal:
                break
            if empty_turns >= 2:
                force_terminal = True
                recovery = {"role": "user", "content": "Call emit_graph now with the best complete evidence DAG available."}
            else:
                recovery = {"role": "user", "content": "Continue with one available tool call. Finish with emit_graph."}
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
                arguments = {}
                result = {"error": f"invalid tool arguments: {exc}"}
                errors.append(result["error"])
            else:
                if call.function.name == "emit_graph":
                    try:
                        graph, parse_quality, parse_warnings = normalize_graph_argument(
                            arguments.get("graph")
                        )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        graph = {}
                        parse_quality = 0.0
                        parse_warnings = [f"invalid emitted graph: {exc}"]
                        errors.append(f"invalid emitted graph: {exc}")
                    score = verify_graph(
                        task,
                        graph,
                        stock,
                        ledger,
                        parse_quality=parse_quality,
                        parse_errors=parse_warnings,
                    )
                    result = {"done": True, "score": score}
                else:
                    result = dispatch(session, call.function.name, arguments)
                    evidence_id = f"ev_{len(ledger) + 1:03d}"
                    ledger.append(
                        {
                            "evidence_id": evidence_id,
                            "tool": call.function.name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    result = {**result, "evidence_id": evidence_id}
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(result, sort_keys=True),
            }
            messages.append(tool_message)
            transcript.append(tool_message)
        if graph is not None:
            break

    if graph is None:
        graph = {}
        score = verify_graph(task, graph, stock, ledger)
    return {
        "task_id": task.task_id,
        "task_label": label,
        "model": model,
        "prompt": task_prompt(task, label, max_turns),
        "graph": graph,
        "score": score,
        "evidence": ledger,
        "transcript": transcript,
        "errors": errors,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reported_cost_usd": round(reported_cost, 8),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "resolved_models": sorted(resolved_models),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="anthropic/claude-opus-5.5")
    parser.add_argument("--endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument(
        "--task-indices",
        default=",".join(str(index) for index in TASK_INDICES),
        help="Comma-separated eval-task indices; the pilot defines labels for 0,4,10.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "v2" / "runs" / "opus-v2.episodes.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(
        base_url=args.endpoint,
        api_key=os.getenv(args.api_key_env) or "dummy",
        timeout=180,
        max_retries=2,
    )
    store = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
    all_eval = store.tasks("eval")
    selected_indices = tuple(
        int(value.strip()) for value in args.task_indices.split(",") if value.strip()
    )
    if not selected_indices:
        parser.error("--task-indices must contain at least one index")
    label_by_index = dict(zip(TASK_INDICES, TASK_LABELS, strict=True))
    try:
        tasks = [all_eval[index] for index in selected_indices]
        labels = [label_by_index[index] for index in selected_indices]
    except (IndexError, KeyError):
        parser.error("--task-indices must use the defined pilot indices: 0,4,10")
    precedent_index = PrecedentIndex(store.tasks("train"))
    prior = []
    if args.resume and args.output.exists():
        prior = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    done = {row["task_id"] for row in prior}
    rows = list(prior)
    for task, label in zip(tasks, labels, strict=True):
        if task.task_id in done:
            continue
        row = rollout(
            client,
            args.model,
            task,
            store.stock(task.stock_id),
            precedent_index,
            label,
            args.max_turns,
        )
        rows.append(row)
        write_jsonl(args.output, rows)
        print(
            json.dumps(
                {
                    "task": label,
                    "reward": row["score"]["reward"],
                    "valid": row["score"]["valid"],
                    "cost": row["usage"]["reported_cost_usd"],
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": "retro-evidence-dag-run-v2",
        "model": args.model,
        "task_indices": list(selected_indices),
        "task_ids": [task.task_id for task in tasks],
        "max_turns": args.max_turns,
        "temperature": 0,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "episodes": len(rows),
    }
    args.output.with_suffix(".run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
