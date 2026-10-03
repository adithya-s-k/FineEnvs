from __future__ import annotations

import copy

import pytest

from retroenv.chemistry import canonical_step_key, canonicalize_smiles
from retroenv.store import TaskStore
from v2.run_pilot import BENCHMARK, TASK_INDICES, parse_graph_argument
from v2.verifier import verify_graph


def _oracle_graph(task, stock):
    nodes = {}
    edges = {}
    reactions = {}
    evidence = []
    evidence_number = 0

    def add_evidence(tool, arguments, result):
        nonlocal evidence_number
        evidence_number += 1
        evidence_id = f"ev_{evidence_number:03d}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "tool": tool,
                "arguments": arguments,
                "result": result,
            }
        )
        return evidence_id

    def molecule_id(smiles):
        canonical = canonicalize_smiles(smiles)
        node_id = f"m{len(nodes) + 1}"
        for existing_id, node in nodes.items():
            if node.get("smiles") == canonical:
                return existing_id
        nodes[node_id] = {
            "id": node_id,
            "type": "molecule",
            "smiles": canonical,
            "metadata": {
                "role": "starting_material",
                "rationale": "Dataset molecule participating in a validated route.",
                "stock_status": "confirmed",
                "evidence_ids": [],
            },
        }
        return node_id

    route_rows = []
    for route_index, route in enumerate(task.reference_routes[:2], 1):
        reaction_ids = []
        route_products = {canonicalize_smiles(step.product) for step in route.steps}
        route_reactants = {
            canonicalize_smiles(reactant)
            for step in route.steps
            for reactant in step.reactants
        }
        for step in route.steps:
            product = canonicalize_smiles(step.product)
            reactants = tuple(sorted(canonicalize_smiles(item) for item in step.reactants))
            key = canonical_step_key(product, reactants)
            reaction_id = reactions.get(key)
            if reaction_id is None:
                reaction_id = f"r{len(reactions) + 1}"
                reactions[key] = reaction_id
                arguments = {"product_smiles": product, "reactants": list(reactants)}
                validation_id = add_evidence(
                    "validate_disconnection", arguments, {"valid": True}
                )
                class_id = add_evidence(
                    "reaction_class_lookup",
                    arguments,
                    {"supported": True, "reaction_class": "unclassified"},
                )
                condition_id = add_evidence(
                    "reaction_conditions_search",
                    arguments,
                    {"supported": True, "conditions": list(step.conditions)},
                )
                nodes[reaction_id] = {
                    "id": reaction_id,
                    "type": "reaction",
                    "metadata": {
                        "reaction_class": "unclassified",
                        "rationale": "Exact dataset-supported transformation.",
                        "confidence": 1.0,
                        "conditions": [str(item) for item in step.conditions],
                        "evidence_ids": [validation_id, class_id, condition_id],
                    },
                }
                product_id = molecule_id(product)
                nodes[product_id]["metadata"]["role"] = (
                    "target" if product == canonicalize_smiles(task.target_smiles) else "intermediate"
                )
                nodes[product_id]["metadata"]["stock_status"] = "not_applicable"
                nodes[product_id]["metadata"]["evidence_ids"].append(validation_id)
                product_edge_id = f"e{len(edges) + 1}"
                edges[product_edge_id] = {
                    "id": product_edge_id,
                    "source": reaction_id,
                    "target": product_id,
                    "type": "product",
                    "metadata": {
                        "role": "forms product",
                        "rationale": "This validated step forms the product.",
                        "evidence_ids": [validation_id],
                    },
                }
                for reactant in reactants:
                    reactant_id = molecule_id(reactant)
                    edge_id = f"e{len(edges) + 1}"
                    edges[edge_id] = {
                        "id": edge_id,
                        "source": reactant_id,
                        "target": reaction_id,
                        "type": "reactant",
                        "metadata": {
                            "role": "atom-contributing precursor",
                            "rationale": "This molecule contributes atoms to the product.",
                            "evidence_ids": [validation_id],
                        },
                    }
            reaction_ids.append(reaction_id)

        terminal_smiles = route_reactants - route_products
        terminal_ids = []
        for smiles in sorted(terminal_smiles):
            node_id = molecule_id(smiles)
            terminal_ids.append(node_id)
            if not nodes[node_id]["metadata"]["evidence_ids"]:
                stock_id = add_evidence(
                    "stock_retrieve",
                    {"query": smiles, "mode": "exact"},
                    {"results": [{"smiles": smiles}]},
                )
                nodes[node_id]["metadata"]["evidence_ids"].append(stock_id)
        route_rows.append(
            {
                "id": f"route_{route_index}",
                "reaction_node_ids": reaction_ids,
                "terminal_node_ids": terminal_ids,
                "rationale": "A complete reference-backed route view.",
            }
        )

    target_id = molecule_id(task.target_smiles)
    graph = {
        "schema_version": "retro-evidence-dag-v2",
        "target_node_id": target_id,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "routes": route_rows,
    }
    return graph, evidence


@pytest.fixture(scope="module")
def pilot_cases():
    store = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")
    tasks = store.tasks("eval")
    return [(tasks[index], store.stock(tasks[index].stock_id)) for index in TASK_INDICES]


def test_all_three_pilot_tasks_have_reachable_perfect_graphs(pilot_cases):
    for task, stock in pilot_cases:
        graph, evidence = _oracle_graph(task, stock)
        result = verify_graph(task, graph, stock, evidence)
        assert result["valid"], result
        assert result["reward"] == 1.0
        assert result["metrics"]["exact_routes"] == 2


def test_missing_edge_reasoning_lowers_reward(pilot_cases):
    task, stock = pilot_cases[0]
    graph, evidence = _oracle_graph(task, stock)
    graph = copy.deepcopy(graph)
    graph["edges"][0]["metadata"]["rationale"] = ""
    result = verify_graph(task, graph, stock, evidence)
    assert not result["valid"]
    assert result["components"]["reasoning_completeness"] < 1.0


def test_orphan_node_is_rejected(pilot_cases):
    task, stock = pilot_cases[0]
    graph, evidence = _oracle_graph(task, stock)
    graph = copy.deepcopy(graph)
    graph["nodes"].append(
        {
            "id": "orphan",
            "type": "molecule",
            "smiles": "C",
            "metadata": {
                "role": "reagent",
                "rationale": "Disconnected molecule.",
                "stock_status": "not_applicable",
                "evidence_ids": [],
            },
        }
    )
    result = verify_graph(task, graph, stock, evidence)
    assert not result["valid"]
    assert "every graph node must participate in a path to the target" in result["errors"]


def test_provider_json_string_graph_is_normalized():
    assert parse_graph_argument('{"schema_version":"retro-evidence-dag-v2"}') == {
        "schema_version": "retro-evidence-dag-v2"
    }
    with pytest.raises(ValueError):
        parse_graph_argument("[]")
