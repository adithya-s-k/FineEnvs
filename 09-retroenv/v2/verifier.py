"""Deterministic verifier for the explicit evidence-bearing RetroEnv v2 DAG."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from retroenv.chemistry import (
    canonical_step_key,
    canonicalize_components,
    canonicalize_smiles,
)
from retroenv.models import RetroTask
from retroenv.verifier import RouteVerifier


WEIGHTS = {
    "parse_validity": 0.05,
    "graph_structure": 0.10,
    "chemistry": 0.25,
    "evidence_grounding": 0.20,
    "reasoning_completeness": 0.15,
    "stock_closure": 0.10,
    "route_diversity": 0.10,
    "reference_coverage": 0.05,
}

INPUT_EDGE_TYPES = {"reactant", "reagent", "catalyst", "solvent"}
EDGE_TYPES = INPUT_EDGE_TYPES | {"product"}
MOLECULE_ROLES = {
    "target",
    "intermediate",
    "starting_material",
    "reagent",
    "catalyst",
    "solvent",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {_text(item) for item in value if _text(item)}


def _canonical_cut(product: str, reactants: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    return canonicalize_smiles(product), canonicalize_components(reactants)


def _evidence_cut(row: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    arguments = row.get("arguments") or {}
    product = arguments.get("product_smiles")
    reactants = arguments.get("reactants")
    if not product or not reactants:
        return None
    try:
        return _canonical_cut(product, reactants)
    except Exception:
        return None


def _stock_evidence_matches(row: dict[str, Any], smiles: str) -> bool:
    if row.get("tool") != "stock_retrieve":
        return False
    arguments = row.get("arguments") or {}
    if arguments.get("mode") != "exact":
        return False
    result = row.get("result") or {}
    for hit in result.get("results") or []:
        try:
            if canonicalize_smiles(hit.get("smiles", "")) == smiles:
                return True
        except Exception:
            continue
    return False


def verify_graph(
    task: RetroTask,
    graph: Any,
    stock: Iterable[str],
    evidence_rows: list[dict[str, Any]],
    *,
    parse_quality: float = 1.0,
    parse_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Score one graph without trusting model-supplied evidence or scores."""

    errors: list[str] = list(parse_errors)
    parse_quality = max(0.0, min(1.0, float(parse_quality)))
    if not isinstance(graph, dict):
        return _empty_result("graph must be an object")
    if graph.get("schema_version") != "retro-evidence-dag-v2":
        errors.append("schema_version must be retro-evidence-dag-v2")

    raw_nodes = graph.get("nodes")
    raw_edges = graph.get("edges")
    raw_routes = graph.get("routes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return _empty_result("nodes must be a non-empty list")
    if not isinstance(raw_edges, list) or not raw_edges:
        return _empty_result("edges must be a non-empty list")
    if not isinstance(raw_routes, list):
        raw_routes = []
        errors.append("routes must be a list")

    evidence = {
        _text(row.get("evidence_id")): row
        for row in evidence_rows
        if _text(row.get("evidence_id"))
    }
    nodes: dict[str, dict[str, Any]] = {}
    molecules: dict[str, str] = {}
    molecule_by_smiles: dict[str, str] = {}
    reactions: dict[str, dict[str, Any]] = {}
    rationale_checks: list[bool] = []
    evidence_id_checks: list[bool] = []

    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        node_id = _text(node.get("id"))
        node_type = node.get("type")
        if not node_id:
            errors.append(f"nodes[{index}].id is required")
            continue
        if node_id in nodes:
            errors.append(f"duplicate node id: {node_id}")
            continue
        if node_type not in {"molecule", "reaction"}:
            errors.append(f"{node_id}: type must be molecule or reaction")
            continue
        nodes[node_id] = node
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        rationale_checks.append(bool(_text(metadata.get("rationale"))))
        cited = _ids(metadata.get("evidence_ids"))
        evidence_id_checks.extend(item in evidence for item in cited)
        if node_type == "molecule":
            try:
                smiles = canonicalize_smiles(node.get("smiles", ""))
            except Exception as exc:
                errors.append(f"{node_id}: invalid molecule: {exc}")
                continue
            if smiles in molecule_by_smiles:
                errors.append(
                    f"duplicate molecule {smiles}; reuse node {molecule_by_smiles[smiles]}"
                )
            molecule_by_smiles.setdefault(smiles, node_id)
            molecules[node_id] = smiles
            if metadata.get("role") not in MOLECULE_ROLES:
                errors.append(f"{node_id}: invalid molecule role")
        else:
            reactions[node_id] = node
            if not _text(metadata.get("reaction_class")):
                errors.append(f"{node_id}: reaction_class is required")
            confidence = metadata.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
                errors.append(f"{node_id}: confidence must be between 0 and 1")

    target_id = _text(graph.get("target_node_id"))
    if target_id not in molecules:
        errors.append("target_node_id must identify a molecule node")
    elif molecules[target_id] != canonicalize_smiles(task.target_smiles):
        errors.append("target_node_id does not match the task target")

    edges: dict[str, dict[str, Any]] = {}
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in nodes}
    edge_reasoning_checks: list[bool] = []
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            errors.append(f"edges[{index}] must be an object")
            continue
        edge_id = _text(edge.get("id"))
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        edge_type = edge.get("type")
        metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        if not edge_id or edge_id in edges:
            errors.append(f"edges[{index}]: id is missing or duplicated")
            continue
        if source not in nodes or target not in nodes:
            errors.append(f"{edge_id}: source and target must exist")
            continue
        if edge_type not in EDGE_TYPES:
            errors.append(f"{edge_id}: unsupported edge type")
            continue
        if edge_type == "product":
            type_ok = source in reactions and target in molecules
        else:
            type_ok = source in molecules and target in reactions
        if not type_ok:
            errors.append(f"{edge_id}: {edge_type} has invalid endpoint types")
        edges[edge_id] = edge
        incoming[target].append(edge)
        outgoing[source].append(edge)
        indegree[target] += 1
        cited = _ids(metadata.get("evidence_ids"))
        edge_reasoning_checks.extend(
            [bool(_text(metadata.get("role"))), bool(_text(metadata.get("rationale")))]
        )
        evidence_id_checks.extend(item in evidence for item in cited)

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    degrees = dict(indegree)
    while queue:
        node_id = queue.popleft()
        visited += 1
        for edge in outgoing.get(node_id, []):
            target = edge["target"]
            degrees[target] -= 1
            if degrees[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        errors.append("graph contains a cycle")

    connected_to_target: set[str] = set()
    if target_id in nodes:
        pending = [target_id]
        while pending:
            node_id = pending.pop()
            if node_id in connected_to_target:
                continue
            connected_to_target.add(node_id)
            pending.extend(edge["source"] for edge in incoming.get(node_id, []))
    fully_connected = connected_to_target == set(nodes)
    if not fully_connected:
        errors.append("every graph node must participate in a path to the target")

    chemistry_verifier = RouteVerifier()
    reaction_details: list[dict[str, Any]] = []
    chemistry_checks: list[bool] = []
    grounding_checks: list[bool] = []
    reaction_step_keys: dict[str, tuple[str, tuple[str, ...]]] = {}
    validation_evidence_by_reaction: dict[str, set[str]] = defaultdict(set)
    condition_evidence_by_reaction: dict[str, set[str]] = defaultdict(set)

    for reaction_id, node in reactions.items():
        input_edges = [edge for edge in incoming.get(reaction_id, []) if edge.get("type") in INPUT_EDGE_TYPES]
        reactant_edges = [edge for edge in input_edges if edge.get("type") == "reactant"]
        product_edges = [edge for edge in outgoing.get(reaction_id, []) if edge.get("type") == "product"]
        if not reactant_edges:
            errors.append(f"{reaction_id}: at least one reactant edge is required")
        if len(product_edges) != 1:
            errors.append(f"{reaction_id}: exactly one product edge is required")
        if not reactant_edges or len(product_edges) != 1:
            continue
        try:
            reactants = canonicalize_components(molecules[edge["source"]] for edge in reactant_edges)
            product = molecules[product_edges[0]["target"]]
        except KeyError:
            errors.append(f"{reaction_id}: reaction edges reference invalid molecules")
            continue
        metadata = node.get("metadata") or {}
        validation = chemistry_verifier.validate_step(
            task, product, reactants, metadata.get("reaction_class")
        )
        chemistry_checks.append(validation.valid)
        cut = (product, reactants)
        reaction_step_keys[reaction_id] = cut
        cited = _ids(metadata.get("evidence_ids"))
        matching = {
            evidence_id
            for evidence_id, row in evidence.items()
            if _evidence_cut(row) == cut
        }
        validation_ids = {
            evidence_id
            for evidence_id in matching
            if evidence[evidence_id].get("tool") == "validate_disconnection"
            and bool((evidence[evidence_id].get("result") or {}).get("valid"))
        }
        class_ids = {
            evidence_id
            for evidence_id in matching
            if evidence[evidence_id].get("tool") == "reaction_class_lookup"
            and bool((evidence[evidence_id].get("result") or {}).get("supported"))
        }
        # A supported lookup with an empty conditions list is still truthful
        # evidence that this source record does not report conditions.  It may
        # ground the reaction metadata, but cannot ground a reagent-like edge.
        condition_lookup_ids = {
            evidence_id
            for evidence_id in matching
            if evidence[evidence_id].get("tool") == "reaction_conditions_search"
            and bool((evidence[evidence_id].get("result") or {}).get("supported"))
            and "conditions" in (evidence[evidence_id].get("result") or {})
        }
        reported_condition_ids = {
            evidence_id
            for evidence_id in condition_lookup_ids
            if bool((evidence[evidence_id].get("result") or {}).get("conditions"))
        }
        validation_evidence_by_reaction[reaction_id] = validation_ids
        condition_evidence_by_reaction[reaction_id] = reported_condition_ids
        grounding_checks.extend(
            [
                bool(cited & validation_ids),
                bool(cited & class_ids),
                bool(cited & condition_lookup_ids),
            ]
        )
        for edge in reactant_edges + product_edges:
            edge_citations = _ids((edge.get("metadata") or {}).get("evidence_ids"))
            grounding_checks.append(bool(edge_citations & validation_ids))
        for edge in input_edges:
            if edge.get("type") in {"reagent", "catalyst", "solvent"}:
                edge_citations = _ids((edge.get("metadata") or {}).get("evidence_ids"))
                grounding_checks.append(bool(edge_citations & reported_condition_ids))
        reaction_details.append(
            {
                "reaction_node_id": reaction_id,
                "product": product,
                "reactants": list(reactants),
                "valid": validation.valid,
                "support": validation.support,
                "validation_evidence": sorted(validation_ids),
                "class_evidence": sorted(class_ids),
                "conditions_evidence": sorted(condition_lookup_ids),
                "reported_conditions_evidence": sorted(reported_condition_ids),
                "errors": list(validation.errors),
            }
        )

    stock_set = frozenset(canonicalize_smiles(item) for item in stock)
    produced_molecules = {
        edge["target"] for edge in edges.values() if edge.get("type") == "product"
    }
    terminal_reactants = {
        edge["source"]
        for edge in edges.values()
        if edge.get("type") == "reactant" and edge["source"] not in produced_molecules
    }
    stock_checks: list[bool] = []
    stock_details: list[dict[str, Any]] = []
    for node_id in sorted(terminal_reactants):
        smiles = molecules.get(node_id, "")
        metadata = (nodes.get(node_id) or {}).get("metadata") or {}
        cited = _ids(metadata.get("evidence_ids"))
        exact_ids = {
            evidence_id
            for evidence_id, row in evidence.items()
            if _stock_evidence_matches(row, smiles)
        }
        valid = (
            smiles in stock_set
            and metadata.get("role") == "starting_material"
            and metadata.get("stock_status") == "confirmed"
            and bool(cited & exact_ids)
        )
        stock_checks.append(valid)
        stock_details.append(
            {
                "node_id": node_id,
                "smiles": smiles,
                "valid": valid,
                "stock_evidence": sorted(exact_ids),
            }
        )

    route_details: list[dict[str, Any]] = []
    first_cuts: set[tuple[str, ...]] = set()
    exact_routes = 0
    route_validity_checks: list[bool] = []
    reference_sets = [
        {canonical_step_key(step.product, step.reactants) for step in route.steps}
        for route in task.reference_routes
    ]
    for index, route in enumerate(raw_routes):
        if not isinstance(route, dict):
            errors.append(f"routes[{index}] must be an object")
            continue
        reaction_ids = route.get("reaction_node_ids")
        if not isinstance(reaction_ids, list) or not reaction_ids:
            errors.append(f"routes[{index}].reaction_node_ids must be non-empty")
            continue
        reaction_ids = [_text(item) for item in reaction_ids]
        if len(reaction_ids) != len(set(reaction_ids)):
            errors.append(f"routes[{index}] repeats a reaction node")
        selected = {item for item in reaction_ids if item in reaction_step_keys}
        selected_steps = {
            canonical_step_key(*reaction_step_keys[item]) for item in selected
        }
        by_product = {product: (reaction_id, reactants) for reaction_id, (product, reactants) in ((item, reaction_step_keys[item]) for item in selected)}
        reachable: set[str] = set()

        def walk(product: str) -> None:
            match = by_product.get(product)
            if not match or match[0] in reachable:
                return
            reaction_id, reactants = match
            reachable.add(reaction_id)
            for reactant in reactants:
                walk(reactant)

        walk(canonicalize_smiles(task.target_smiles))
        target_match = by_product.get(canonicalize_smiles(task.target_smiles))
        cut = tuple(sorted(target_match[1])) if target_match else ()
        if cut:
            first_cuts.add(cut)
        exact = selected_steps in reference_sets
        exact_routes += int(exact)
        valid = (
            len(selected) == len(reaction_ids)
            and len(reaction_ids) == len(set(reaction_ids))
            and reachable == selected
            and len(selected) <= task.max_steps
            and bool(target_match)
            and all(
                next((item["valid"] for item in reaction_details if item["reaction_node_id"] == reaction_id), False)
                for reaction_id in selected
            )
        )
        route_validity_checks.append(valid)

        raw_terminal_ids = route.get("terminal_node_ids")
        if not isinstance(raw_terminal_ids, list) or not raw_terminal_ids:
            errors.append(f"routes[{index}].terminal_node_ids must be non-empty")
            declared_terminal_ids: set[str] = set()
        else:
            declared_terminal_ids = {_text(item) for item in raw_terminal_ids}
        route_products = {
            reaction_step_keys[item][0] for item in selected if item in reaction_step_keys
        }
        expected_terminal_ids = {
            molecule_by_smiles[reactant]
            for item in selected
            for reactant in reaction_step_keys[item][1]
            if reactant not in route_products and reactant in molecule_by_smiles
        }
        terminals_match = declared_terminal_ids == expected_terminal_ids
        if not terminals_match:
            errors.append(f"routes[{index}] terminal_node_ids do not match its route leaves")
            route_validity_checks[-1] = False
            valid = False
        route_details.append(
            {
                "route_id": _text(route.get("id")) or f"route_{index + 1}",
                "reaction_node_ids": reaction_ids,
                "valid": valid,
                "exact_reference_match": exact,
                "first_cut": list(cut),
                "connected_reactions": len(reachable),
                "terminal_node_ids_match": terminals_match,
            }
        )

    # Product molecules are claims too: cite the validation evidence of at
    # least one incoming reaction.  This prevents empty evidence arrays from
    # earning a perfect reasoning/grounding score.
    for molecule_id in sorted(produced_molecules):
        cited = _ids(((nodes.get(molecule_id) or {}).get("metadata") or {}).get("evidence_ids"))
        producer_ids = {
            edge["source"]
            for edge in incoming.get(molecule_id, [])
            if edge.get("type") == "product" and edge.get("source") in reactions
        }
        valid_evidence = set().union(
            *(validation_evidence_by_reaction[item] for item in producer_ids)
        ) if producer_ids else set()
        grounding_checks.append(bool(cited & valid_evidence))

    if target_id in nodes:
        target_metadata = nodes[target_id].get("metadata") or {}
        if target_metadata.get("role") != "target":
            errors.append("target molecule role must be target")

    route_count_ok = task.min_routes <= len(raw_routes) <= task.max_routes
    diversity = min(1.0, len(first_cuts) / max(1, task.min_routes))
    if not route_count_ok:
        errors.append(
            f"route count {len(raw_routes)} outside [{task.min_routes}, {task.max_routes}]"
        )
    if diversity < 1:
        errors.append("routes do not provide enough distinct first cuts")

    structural_checks = [
        bool(nodes),
        bool(edges),
        target_id in molecules,
        visited == len(nodes),
        fully_connected,
        route_count_ok,
        all(route_validity_checks) if route_validity_checks else False,
    ]
    components = {
        "parse_validity": parse_quality,
        "graph_structure": sum(structural_checks) / len(structural_checks),
        "chemistry": sum(chemistry_checks) / len(chemistry_checks) if chemistry_checks else 0.0,
        "evidence_grounding": sum(grounding_checks) / len(grounding_checks) if grounding_checks else 0.0,
        "reasoning_completeness": (
            sum(rationale_checks + edge_reasoning_checks + evidence_id_checks)
            / len(rationale_checks + edge_reasoning_checks + evidence_id_checks)
            if rationale_checks or edge_reasoning_checks or evidence_id_checks
            else 0.0
        ),
        "stock_closure": sum(stock_checks) / len(stock_checks) if stock_checks else 0.0,
        "route_diversity": diversity,
        "reference_coverage": exact_routes / len(raw_routes) if raw_routes else 0.0,
    }
    reward = sum(WEIGHTS[key] * value for key, value in components.items())
    hard_valid = (
        not errors
        and parse_quality == 1.0
        and all(chemistry_checks)
        and all(grounding_checks)
        and all(stock_checks)
        and all(route_validity_checks)
        and components["reasoning_completeness"] == 1.0
    )
    return {
        "schema_version": "retro-evidence-dag-score-v2",
        "valid": hard_valid,
        "reward": round(reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "errors": list(dict.fromkeys(errors)),
        "metrics": {
            "nodes": len(nodes),
            "molecules": len(molecules),
            "reactions": len(reactions),
            "edges": len(edges),
            "routes": len(raw_routes),
            "exact_routes": exact_routes,
            "distinct_first_cuts": len(first_cuts),
            "terminal_reactants": len(terminal_reactants),
        },
        "reaction_details": reaction_details,
        "stock_details": stock_details,
        "route_details": route_details,
    }


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "retro-evidence-dag-score-v2",
        "valid": False,
        "reward": 0.0,
        "components": {key: 0.0 for key in WEIGHTS},
        "errors": [reason],
        "metrics": {
            "nodes": 0,
            "molecules": 0,
            "reactions": 0,
            "edges": 0,
            "routes": 0,
            "exact_routes": 0,
            "distinct_first_cuts": 0,
            "terminal_reactants": 0,
        },
        "reaction_details": [],
        "stock_details": [],
        "route_details": [],
    }
