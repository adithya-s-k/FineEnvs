from __future__ import annotations

import copy

from retroenv.graph import parse_submission, routes_to_submission
from retroenv.models import ReferenceRoute, RetroTask
from retroenv.retrieval import PrecedentIndex, StockIndex, molecule_lookup
from retroenv.verifier import RouteVerifier

from conftest import make_task


STOCK = {"CCO", "CC(=O)O"}


def test_reference_round_trip_is_renderable_and_scores_one():
    task = make_task()
    submission = routes_to_submission(
        task.target_smiles, task.reference_routes, STOCK, source="oracle-smoke-test"
    )
    parsed = parse_submission(submission)
    assert parsed.parse_valid is True
    assert parsed.trees[0].graph_valid is True
    result = RouteVerifier().score_submission(task, submission, STOCK)
    assert result.valid is True
    assert result.reward == 1.0
    assert result.components == {
        "parse_validity": 1.0,
        "molecule_validity": 1.0,
        "graph_validity": 1.0,
        "step_correctness": 1.0,
        "stock_correctness": 1.0,
        "reference_similarity": 1.0,
        "exact_route_match": 1.0,
        "verified_route_diversity": 1.0,
        "route_set_compliance": 1.0,
    }


def test_dense_reward_distinguishes_parse_and_chemistry_failures():
    verifier = RouteVerifier()
    task = make_task()
    invalid_json = verifier.score_submission(task, "not JSON", STOCK)
    empty_graph = verifier.score_submission(task, {"routes": []}, STOCK)
    assert invalid_json.reward == 0.0
    assert 0.0 < empty_graph.reward < 1.0
    assert empty_graph.components["parse_validity"] == 1.0
    assert empty_graph.valid is False


def test_route_count_cannot_be_gamed_with_non_molecule_wrappers():
    base = make_task()
    task = RetroTask(
        task_id="two-route-garbage-guard",
        mode="route_planning",
        target_smiles=base.target_smiles,
        max_steps=1,
        stock_id=base.stock_id,
        split="eval",
        reference_routes=base.reference_routes,
        min_routes=2,
        max_routes=2,
    )
    result = RouteVerifier().score_submission(task, {"routes": [{}, {}]}, STOCK)
    assert result.reward == 0.05
    assert result.components["route_set_compliance"] == 0.0
    assert result.components["graph_validity"] == 0.0
    assert result.valid is False


def test_false_stock_claim_loses_stock_reward():
    task = make_task()
    submission = routes_to_submission(task.target_smiles, task.reference_routes, STOCK)
    submission["routes"][0]["children"][0]["children"][0]["in_stock"] = False
    result = RouteVerifier().score_submission(task, submission, STOCK)
    assert result.valid is True
    assert result.components["stock_correctness"] < 1.0
    assert result.reward < 1.0


def test_two_distinct_reference_trees_satisfy_diversity_contract():
    first = make_task()
    alternate = make_task(reactants=("CCBr", "CC(=O)O", "O"))
    task = RetroTask(
        task_id="two-route-task",
        mode="route_planning",
        target_smiles=first.target_smiles,
        max_steps=1,
        stock_id="test_stock",
        split="train",
        reference_routes=(
            first.reference_routes[0],
            ReferenceRoute("alternate", alternate.reference_routes[0].steps, ()),
        ),
        min_routes=2,
        max_routes=2,
    )
    stock = {"CCO", "CCBr", "O", "CC(=O)O"}
    submission = routes_to_submission(task.target_smiles, task.reference_routes, stock)
    result = RouteVerifier().score_submission(task, submission, stock)
    assert result.valid is True
    assert result.metrics["valid_routes"] == 2
    assert result.metrics["verified_route_diversity"] == 1.0

    one_bad = copy.deepcopy(submission)
    second_precursors = one_bad["routes"][1]["children"][0]["children"]
    next(node for node in second_precursors if node["smiles"] == "CCBr")["smiles"] = "CCCl"
    incomplete = RouteVerifier().score_submission(task, one_bad, stock | {"CCCl"})
    assert incomplete.metrics["valid_routes"] == 1
    assert incomplete.valid is False


def test_stock_retrieval_is_exact_and_capped():
    index = StockIndex({"CCO", "CCCO", "CCCCO"})
    exact = index.retrieve("CCO", mode="exact", limit=20)
    assert [row["smiles"] for row in exact["results"]] == ["CCO"]
    similarity = index.retrieve("CCO", mode="similarity", limit=2)
    assert similarity["returned"] == 2
    assert similarity["results"][0]["smiles"] == "CCO"


def test_precedent_index_excludes_current_task():
    task = make_task()
    index = PrecedentIndex((task,))
    assert index.search(task_id=task.task_id, product_smiles=task.target_smiles)[
        "results"
    ] == []


def test_pubchem_name_lookup_uses_frozen_cache():
    result = molecule_lookup(
        "ethanol",
        {"ethanol": {"canonical_smiles": "OCC", "cid": 702}},
    )
    assert result["canonical_smiles"] == "CCO"
    assert result["cid"] == 702
    assert result["source"] == "frozen_pubchem_cache"
