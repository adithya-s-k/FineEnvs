#!/usr/bin/env python3
"""Build the static multi-model, three-task diagnostic payload."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from retroenv.chemistry import canonicalize_smiles
from retroenv.environment import PROMPT as LEGACY_USER_PROMPT
from retroenv.store import TaskStore
from retroenv.verifier import GRAPH_WEIGHTS, RouteVerifier
from v2.run_pilot import (
    BENCHMARK,
    ROOT,
    SYSTEM_PROMPT,
    TASK_INDICES,
    TASK_LABELS,
)
from v2.verifier import WEIGHTS as V2_WEIGHTS


HERE = Path(__file__).resolve().parent
V2_RUNS = ROOT / "v2" / "runs" / "opus-v2.episodes.jsonl"
LEGACY_ROOT = BENCHMARK / "model-runs" / "board-v1"
LEGACY_MODELS = (
    ("sonnet", "Claude Sonnet 5", "anthropic/claude-sonnet-5", "sonnet.episodes.jsonl"),
    ("luna", "GPT-5.6 Luna", "openai/gpt-5.6-luna", "luna.episodes.jsonl"),
    ("sol", "GPT-5.6 Sol", "openai/gpt-5.6-sol", "sol.episodes.jsonl"),
    ("qwenmax", "Qwen3.8 Max", "qwen/qwen3.8-max-0902", "qwenmax.episodes.jsonl"),
    ("qwen27", "Qwen3.8 27B", "qwen/qwen3.8-27b", "qwen.episodes.jsonl"),
    ("flash", "DeepSeek V4.1 Flash", "deepseek/deepseek-v4.1-flash", "flash.episodes.jsonl"),
    ("pro", "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro-0813", "pro.episodes.jsonl"),
)
EXTRA_V2_RUNS = {
    "sonnet": ROOT / "v2" / "runs" / "sonnet-v2-task-a.episodes.jsonl",
    "luna": ROOT / "v2" / "runs" / "luna-v2-task-a.episodes.jsonl",
    "sol": ROOT / "v2" / "runs" / "sol-v2-task-a.episodes.jsonl",
    "qwenmax": ROOT / "v2" / "runs" / "qwenmax-v2-task-a.episodes.jsonl",
}
V2_TOOLS = (
    "inspect_molecule",
    "reaction_precedent_search",
    "stock_retrieve",
    "validate_disconnection",
    "reaction_class_lookup",
    "reaction_conditions_search",
    "emit_graph",
)
LEGACY_TOOLS = (
    "inspect_molecule",
    "pubchem_lookup",
    "stock_retrieve",
    "reaction_precedent_search",
    "validate_disconnection",
    "reaction_class_lookup",
    "reaction_conditions_search",
    "search_literature",
    "emit_routes",
)
LEGACY_SYSTEM_PROMPT = (
    "You are a retrosynthesis planning agent. Use tools, validate cuts, confirm every "
    "stock leaf by exact lookup, and finish only with emit_routes. You have at most 16 "
    "model turns and must reserve the final turn for emit_routes even if the routes are "
    "incomplete. Do not reveal chain-of-thought; put short evidence-based explanations "
    "in reaction metadata."
)

V2_COMPONENT_LABELS = {
    "parse_validity": "Parse validity",
    "graph_structure": "Graph structure",
    "chemistry": "Chemistry",
    "evidence_grounding": "Evidence grounding",
    "reasoning_completeness": "Reasoning",
    "stock_closure": "Stock closure",
    "route_diversity": "Route diversity",
    "reference_coverage": "Reference coverage",
}
LEGACY_COMPONENT_LABELS = {
    "parse_validity": "Parse validity",
    "molecule_validity": "Molecule validity",
    "graph_validity": "Tree validity",
    "step_correctness": "Step correctness",
    "stock_correctness": "Stock correctness",
    "reference_similarity": "Reference similarity",
    "exact_route_match": "Exact-route match",
    "verified_route_diversity": "Route diversity",
    "route_set_compliance": "Route-set compliance",
}


def molecule_svg(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(260, 124)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def summarize_result(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("valid", "supported", "support", "reaction_class", "returned", "error"):
        if key in result:
            summary[key] = result[key]
    if result.get("errors"):
        summary["errors"] = result["errors"]
    if tool == "stock_retrieve":
        summary["hits"] = [item.get("smiles") for item in result.get("results") or []]
    elif tool == "reaction_conditions_search":
        summary["conditions"] = result.get("conditions") or []
    elif tool == "reaction_precedent_search":
        summary["top_precedents"] = [
            {
                "product": item.get("product_smiles"),
                "reactants": item.get("reactants"),
                "similarity": item.get("similarity"),
            }
            for item in (result.get("results") or [])[:3]
        ]
    return summary


def diagnosis(
    score: dict[str, Any], labels: dict[str, str]
) -> dict[str, list[str]]:
    components = score.get("components") or {}
    worked = [labels.get(key, key) for key, value in components.items() if value >= 0.999]
    failed = [
        f"{labels.get(key, key)} · {round(value * 100)}%"
        for key, value in components.items()
        if value < 0.999
    ]
    for detail in score.get("reaction_details") or []:
        if not detail.get("valid"):
            failed.append(f"{detail['reaction_node_id']} is not dataset/template supported")
    for detail in score.get("stock_details") or []:
        if not detail.get("valid"):
            failed.append(f"{detail['node_id']} lacks a valid exact-stock claim")
    if not components:
        failed.append("No scoreable submission")
    return {"worked": worked, "failed": failed}


def reference_shape(task: Any) -> list[dict[str, Any]]:
    return [
        {
            "steps": len(route.steps),
            "reactants_per_step": [len(step.reactants) for step in route.steps],
        }
        for route in task.reference_routes
    ]


def task_dataset(task: Any, stock_size: int) -> dict[str, Any]:
    return {
        "benchmark": "PaRoutes-derived RetroEval v1",
        "split": task.split,
        "stock_size": stock_size,
        "private_reference_routes": len(task.reference_routes),
        "reference_shape": reference_shape(task),
        "references_hidden_from_model": True,
    }


def drawings_for(graph: dict[str, Any]) -> dict[str, str | None]:
    return {
        node["id"]: molecule_svg(node["smiles"])
        for node in graph.get("nodes") or []
        if node.get("type") == "molecule" and node.get("smiles")
    }


def _canonical_or_raw(value: Any) -> str:
    text = str(value or "").strip()
    try:
        return canonicalize_smiles(text)
    except Exception:
        return text


def _stable_id(prefix: str, value: str, occupied: set[str]) -> str:
    base = f"{prefix}_{hashlib.sha1(value.encode()).hexdigest()[:8]}"
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def legacy_tree_to_dag(submission: Any, target_smiles: str) -> dict[str, Any]:
    """Convert a saved v1 nested tree for display without inventing evidence."""

    if isinstance(submission, str):
        try:
            submission = json.loads(submission)
        except json.JSONDecodeError:
            submission = {}
    raw_routes = submission.get("routes") if isinstance(submission, dict) else None
    if not isinstance(raw_routes, list):
        raw_routes = []

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    molecule_ids: dict[str, str] = {}
    occupied: set[str] = set()
    target = _canonical_or_raw(target_smiles)
    role_priority = {"starting_material": 1, "intermediate": 2, "target": 3}

    def molecule(node: dict[str, Any], role: str) -> str:
        smiles = _canonical_or_raw(node.get("smiles"))
        if smiles in molecule_ids:
            node_id = molecule_ids[smiles]
            metadata = nodes[node_id]["metadata"]
            if role_priority[role] > role_priority[metadata["role"]]:
                metadata["role"] = role
            if node.get("in_stock") is True and metadata["role"] == "starting_material":
                metadata["stock_status"] = "confirmed"
            return node_id
        node_id = _stable_id("m", smiles or f"invalid-{len(molecule_ids)}", occupied)
        molecule_ids[smiles] = node_id
        stock = node.get("in_stock") is True and role == "starting_material"
        rationale = (
            "Target supplied by the task."
            if role == "target"
            else (
                "Model marked this terminal molecule as in stock."
                if stock
                else "Molecule present in the model's legacy route tree."
            )
        )
        nodes[node_id] = {
            "id": node_id,
            "type": "molecule",
            "smiles": smiles,
            "metadata": {
                "role": role,
                "rationale": rationale,
                "stock_status": "confirmed" if stock else "not_applicable",
                "evidence_ids": [],
            },
        }
        return node_id

    route_rows: list[dict[str, Any]] = []
    for route_index, root in enumerate(raw_routes, 1):
        reaction_ids: list[str] = []
        terminal_ids: set[str] = set()

        def walk(
            node: Any, *, is_root: bool = False, ancestry: tuple[str, ...] = ()
        ) -> str | None:
            if not isinstance(node, dict):
                return None
            smiles = _canonical_or_raw(node.get("smiles"))
            if not smiles or smiles in ancestry:
                return None
            children = node.get("children") if isinstance(node.get("children"), list) else []
            role = (
                "target"
                if is_root or smiles == target
                else ("intermediate" if children else "starting_material")
            )
            molecule_id = molecule(node, role)
            if not children:
                terminal_ids.add(molecule_id)
                return molecule_id
            reaction = children[0] if isinstance(children[0], dict) else {}
            reaction_children = (
                reaction.get("children")
                if isinstance(reaction.get("children"), list)
                else []
            )
            metadata = (
                reaction.get("metadata")
                if isinstance(reaction.get("metadata"), dict)
                else {}
            )
            explanation = str(
                metadata.get("explanation") or "No reaction explanation was emitted."
            ).strip()
            reaction_class = str(
                metadata.get("reaction_class")
                or metadata.get("classification")
                or "unclassified"
            )
            reaction_id = _stable_id(
                f"r{route_index}",
                f"{smiles}:{len(reaction_ids)}:{reaction_class}",
                occupied,
            )
            confidence = metadata.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                confidence = 0.0
            conditions = metadata.get("conditions") or []
            if isinstance(conditions, dict):
                conditions = [conditions]
            nodes[reaction_id] = {
                "id": reaction_id,
                "type": "reaction",
                "metadata": {
                    "reaction_class": reaction_class,
                    "rationale": explanation,
                    "confidence": max(0.0, min(1.0, float(confidence))),
                    "conditions": [
                        json.dumps(item, sort_keys=True)
                        if isinstance(item, dict)
                        else str(item)
                        for item in conditions
                    ],
                    "evidence_ids": [],
                },
            }
            reaction_ids.append(reaction_id)
            precursor_roles = (
                metadata.get("precursor_roles")
                if isinstance(metadata.get("precursor_roles"), dict)
                else {}
            )
            for child in reaction_children:
                child_id = walk(child, ancestry=(*ancestry, smiles))
                if child_id is None:
                    continue
                child_smiles = nodes[child_id].get("smiles", "")
                role_text = (
                    precursor_roles.get(child.get("smiles"))
                    or precursor_roles.get(child_smiles)
                    or "reactant"
                )
                edge_id = _stable_id(
                    "e", f"{child_id}:{reaction_id}:{len(edges)}", occupied
                )
                edges.append(
                    {
                        "id": edge_id,
                        "source": child_id,
                        "target": reaction_id,
                        "type": "reactant",
                        "metadata": {
                            "role": str(role_text),
                            "rationale": f"Legacy precursor role: {role_text}.",
                            "evidence_ids": [],
                        },
                    }
                )
            product_edge_id = _stable_id(
                "e", f"{reaction_id}:{molecule_id}:{len(edges)}", occupied
            )
            edges.append(
                {
                    "id": product_edge_id,
                    "source": reaction_id,
                    "target": molecule_id,
                    "type": "product",
                    "metadata": {
                        "role": "forms product",
                        "rationale": explanation,
                        "evidence_ids": [],
                    },
                }
            )
            return molecule_id

        root_id = walk(root, is_root=True)
        if reaction_ids:
            route_rows.append(
                {
                    "id": f"route_{route_index}",
                    "reaction_node_ids": reaction_ids,
                    "terminal_node_ids": sorted(terminal_ids),
                    "rationale": "Display conversion of the saved v1 nested route tree.",
                }
            )
        if root_id and nodes[root_id].get("smiles") == target:
            nodes[root_id]["metadata"]["role"] = "target"

    return {
        "schema_version": "retro-display-dag-from-v1",
        "target_node_id": molecule_ids.get(target),
        "nodes": list(nodes.values()),
        "edges": edges,
        "routes": route_rows,
    }


def legacy_calls(
    transcript: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    calls: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    terminal_calls = 0
    for message in transcript:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                calls[str(call.get("id"))] = {
                    "tool": function.get("name") or "unknown",
                    "arguments": arguments,
                }
        if message.get("role") != "tool":
            continue
        match = calls.get(str(message.get("tool_call_id")), {})
        tool = str(message.get("name") or match.get("tool") or "unknown")
        if tool in {"emit_routes", "emit_graph"}:
            terminal_calls += 1
            continue
        try:
            result = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            result = {"error": "tool response was not valid JSON"}
        rows.append(
            {
                "index": len(rows) + 1,
                "evidence_id": f"call_{len(rows) + 1:03d}",
                "tool": tool,
                "arguments": match.get("arguments") or {},
                "summary": summarize_result(
                    tool, result if isinstance(result, dict) else {}
                ),
            }
        )
    return rows, terminal_calls


def v2_episode(
    row: dict[str, Any],
    task: Any,
    stock_size: int,
    *,
    model_id: str = "opus",
    model_label: str = "Claude Opus 5.5",
) -> dict[str, Any]:
    graph = dict(row.get("graph") or {})
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    graph.setdefault("routes", [])
    graph.setdefault("target_node_id", None)
    renderable = bool(
        graph.get("routes")
        and graph.get("edges")
        and any(node.get("type") == "reaction" for node in graph.get("nodes") or [])
    )
    calls = [
        {
            "index": index,
            "evidence_id": item["evidence_id"],
            "tool": item["tool"],
            "arguments": item.get("arguments") or {},
            "summary": summarize_result(item["tool"], item.get("result") or {}),
        }
        for index, item in enumerate(row.get("evidence") or [], 1)
    ]
    return {
        "task_id": row["task_id"],
        "model_id": model_id,
        "label": row["task_label"],
        "model": model_label,
        "requested_model": row["model"],
        "harness_version": "v2",
        "harness_label": "v2 evidence DAG",
        "task": task.to_dict(include_references=False),
        "dataset": task_dataset(task, stock_size),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": row["prompt"],
        "graph": graph,
        "graph_available": renderable,
        "graph_note": (
            "Native v2 evidence DAG. Node and edge citations are model-supplied and verifier-checked."
            if renderable
            else "A native v2 attempt was saved, but it did not contain a renderable molecule → reaction → product graph. Its verifier errors and full tool trajectory remain visible."
        ),
        "drawings": drawings_for(graph),
        "score": row["score"],
        "score_weights": V2_WEIGHTS,
        "component_labels": V2_COMPONENT_LABELS,
        "diagnosis": diagnosis(row["score"], V2_COMPONENT_LABELS),
        "evidence": calls,
        "evidence_contract": True,
        "trajectory_title": "Tool evidence ledger",
        "trajectory_copy": "Immutable evidence IDs can be traced from tool responses into graph claims.",
        "tools": list(V2_TOOLS),
        "terminal_calls": 1,
        "normalization_warnings": row.get("normalization_warnings") or [],
        "errors": row.get("errors") or [],
        "usage": row["usage"],
    }


def legacy_episode(
    row: dict[str, Any],
    model_id: str,
    model_label: str,
    task_label: str,
    task: Any,
    stock: frozenset[str],
) -> dict[str, Any]:
    graph = legacy_tree_to_dag(row.get("submission") or {}, task.target_smiles)
    score = RouteVerifier().score_submission(
        task, row.get("submission") or {}, stock
    ).to_dict()
    calls, terminal_calls = legacy_calls(row.get("transcript") or [])
    return {
        "task_id": row["task_id"],
        "model_id": model_id,
        "label": task_label,
        "model": model_label,
        "requested_model": row.get("requested_model"),
        "harness_version": "v1",
        "harness_label": "saved legacy v1 tree",
        "task": task.to_dict(include_references=False),
        "dataset": task_dataset(task, len(stock)),
        "system_prompt": LEGACY_SYSTEM_PROMPT,
        "user_prompt": LEGACY_USER_PROMPT.format(
            target=task.target_smiles,
            max_steps=task.max_steps,
            min_routes=task.min_routes,
            max_routes=task.max_routes,
        ),
        "graph": graph,
        "graph_available": bool(graph.get("routes")),
        "graph_note": (
            "Display-only DAG converted from the saved v1 nested tree. Reaction explanations and precursor roles are preserved; "
            "v1 did not require node/edge evidence citations."
            if graph.get("routes")
            else "No parseable route tree was emitted in this saved run. The failed tool trajectory is preserved below."
        ),
        "drawings": drawings_for(graph),
        "score": score,
        "score_weights": GRAPH_WEIGHTS,
        "component_labels": LEGACY_COMPONENT_LABELS,
        "diagnosis": diagnosis(score, LEGACY_COMPONENT_LABELS),
        "evidence": calls,
        "evidence_contract": False,
        "trajectory_title": "Legacy tool transcript",
        "trajectory_copy": "Recovered tool calls and responses from the saved v1 rollout; call IDs are display labels, not graph citations.",
        "tools": list(LEGACY_TOOLS),
        "terminal_calls": terminal_calls,
        "normalization_warnings": (
            ["legacy v1 tree converted to a flat DAG for display"]
            if graph.get("routes")
            else []
        ),
        "errors": row.get("errors") or [],
        "usage": row.get("usage") or {},
    }


def main() -> int:
    store = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
    all_eval = store.tasks("eval")
    selected_tasks = [all_eval[index] for index in TASK_INDICES]
    selected_ids = {task.task_id for task in selected_tasks}
    task_by_id = {task.task_id: task for task in selected_tasks}
    labels_by_id = {
        task.task_id: label
        for task, label in zip(selected_tasks, TASK_LABELS, strict=True)
    }
    models = [
        {
            "id": "opus",
            "label": "Claude Opus 5.5",
            "requested_model": "anthropic/claude-opus-5.5",
            "harness_version": "v2",
            "harness_label": "v2 evidence DAG",
        }
    ]
    models.extend(
        {
            "id": model_id,
            "label": label,
            "requested_model": requested,
            "harness_version": "v1",
            "harness_label": "saved legacy v1 tree",
        }
        for model_id, label, requested, _ in LEGACY_MODELS
    )

    episodes: list[dict[str, Any]] = []
    for line in V2_RUNS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task_id"] not in selected_ids:
            continue
        task = task_by_id[row["task_id"]]
        episodes.append(v2_episode(row, task, len(store.stock(task.stock_id))))

    native_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id, path in EXTRA_V2_RUNS.items():
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                native_rows[(model_id, row["task_id"])] = row

    for model_id, model_label, _, filename in LEGACY_MODELS:
        rows = [
            json.loads(line)
            for line in (LEGACY_ROOT / filename).read_text().splitlines()
            if line.strip()
        ]
        by_task = {row["task_id"]: row for row in rows}
        for task in selected_tasks:
            native = native_rows.get((model_id, task.task_id))
            if native is not None:
                episodes.append(
                    v2_episode(
                        native,
                        task,
                        len(store.stock(task.stock_id)),
                        model_id=model_id,
                        model_label=model_label,
                    )
                )
            else:
                episodes.append(
                    legacy_episode(
                        by_task[task.task_id],
                        model_id,
                        model_label,
                        labels_by_id[task.task_id],
                        task,
                        store.stock(task.stock_id),
                    )
                )

    payload = {
        "schema_version": "retro-diagnostic-web-v3",
        "title": "RetroEnv · multi-model three-task diagnostic",
        "models": models,
        "tasks": [
            {
                "task_id": task.task_id,
                "label": labels_by_id[task.task_id],
                "max_steps": task.max_steps,
            }
            for task in selected_tasks
        ],
        "episodes": episodes,
        "comparison_note": (
            "Task A uses native v2 attempts for Opus, Sonnet, Luna, Sol, and Qwen Max. Missing Task B/C v2 runs fall back to saved v1 trajectories. "
            "Each view is labeled; v1 and v2 rewards are not directly rank-comparable."
        ),
    }
    (HERE / "data.js").write_text(
        "window.RETRO_V2_DATA = "
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print(
        f"wrote {HERE / 'data.js'} with {len(models)} models and {len(episodes)} episodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
