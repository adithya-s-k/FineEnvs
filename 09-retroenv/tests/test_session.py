from __future__ import annotations

import json

from conftest import ESTER, STOCK, make_task, step
from retroenv.chemistry import canonicalize_components, canonicalize_smiles
from retroenv.environment import RetroRouteSession
from retroenv.graph import routes_to_submission
from retroenv.leakage import LeakageRules
from retroenv.models import ReferenceRoute
from retroenv.retrieval import PrecedentIndex, StockIndex


def rows(*reactions, visible=True, patent="US0000001"):
    out = []
    for product, reactants in reactions:
        product, reactants = canonicalize_smiles(product), list(canonicalize_components(reactants))
        out.append(
            {
                "product": product,
                "reactants": reactants,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "reaction_class": "esterification",
                "patents": [patent],
                "reagents": ["O"],
                "visible": visible,
            }
        )
    return out


def session(library, precedents=(), **kwargs):
    return RetroRouteSession(library=library, precedent_index=PrecedentIndex(precedents, LeakageRules()), **kwargs)


def test_the_opening_hides_known_routes_and_states_constraints(library):
    task = make_task(forbidden=("SNAr",), excluded=("CC(=O)O",), min_routes=2, variant="diversity")
    opening = session(library).reset(task, STOCK)
    text = json.dumps(opening)
    assert "route_0" not in text and "US0000000" not in text
    assert "SNAr" in opening["prompt"] and "CC(=O)O" in opening["prompt"]
    assert "at least 2 valid routes" in opening["prompt"]


def test_the_tool_budget_is_enforced_but_emit_stays_available(library):
    task = make_task()
    s = session(library, max_tool_calls=1)
    s.reset(task, STOCK)
    assert "results" in s.stock_retrieve("CCO", mode="exact")
    assert "budget" in s.inspect_molecule("CCO")["error"]
    submission = routes_to_submission(task.target_smiles, [ReferenceRoute("r", task.reference_routes[0].steps)], STOCK)
    assert s.emit_routes(submission)["score"]["valid"]


def test_a_train_task_cannot_confirm_its_own_reaction_by_corpus_lookup(library):
    s = session(library)
    s.reset(make_task(split="train"), STOCK)
    assert s.validate_disconnection(ESTER, ["CCO", "CC(=O)O"])["support"] == "reaction_template"
    s.reset(make_task(split="test_id"), STOCK)
    assert s.validate_disconnection(ESTER, ["CCO", "CC(=O)O"])["support"] == "known_precedent"


def test_excluded_building_blocks_are_absent_from_stock_search(library):
    s = session(library)
    s.reset(make_task(excluded=("CC(=O)O",)), STOCK)
    assert s.stock_retrieve("CC(=O)O", mode="exact")["results"] == []
    assert s.stock_retrieve("CCO", mode="exact")["returned"] == 1


def test_class_lookup_flags_forbidden_classes(library):
    s = session(library)
    s.reset(make_task(forbidden=("esterification",)), STOCK)
    result = s.reaction_class_lookup(ESTER, ["CCO", "CC(=O)O"])
    assert result["reaction_class"] == "esterification" and result["forbidden_in_this_task"]


def test_precedents_hide_a_train_tasks_own_keys_but_not_from_held_out_tasks(library):
    precedents = rows((ESTER, ("CCO", "CC(=O)O")), patent="US0000000")
    s = session(library, precedents)
    s.reset(make_task(split="train"), STOCK)
    assert s.reaction_precedent_search(ESTER)["results"] == []
    s.reset(make_task(split="test_id"), STOCK)
    assert s.reaction_precedent_search(ESTER)["returned"] == 1


def test_invisible_corpus_reactions_never_reach_the_tools(library):
    s = session(library, rows((ESTER, ("CCO", "CC(=O)O")), visible=False))
    s.reset(make_task(), STOCK)
    assert s.reaction_precedent_search(ESTER)["results"] == []


def test_stock_index_caps_results():
    index = StockIndex([f"C{'C' * i}O" for i in range(30)])
    result = index.retrieve("CO", mode="substructure", limit=50)
    assert result["returned"] == 20 and result["truncated"]


def test_a_second_emit_is_refused(library):
    task = make_task()
    s = session(library)
    s.reset(task, STOCK)
    s.emit_routes({"routes": []})
    assert "already complete" in s.emit_routes({"routes": []})["error"]


def test_unaided_toolset_drops_the_step_checker(library):
    s = session(library, toolset="unaided")
    s.reset(make_task(), STOCK)
    assert "validate_disconnection" not in s.tool_names
    assert "not available" in s.validate_disconnection(ESTER, ["CCO"])["error"]
    assert step(ESTER, "CCO", "CC(=O)O").product == canonicalize_smiles(ESTER)
