from __future__ import annotations

from retroenv.chemistry import canonicalize_components, canonicalize_smiles
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask
from retroenv.verifier import RouteVerifier

from conftest import make_task


STOCK = {"CCO", "CC(=O)O", "CO", "CCBr", "CCCl", "O"}


def test_exact_supported_route_scores_one():
    task = make_task()
    result = RouteVerifier().score_route(
        task,
        {"route": [{"product": "CCOC(C)=O", "reactants": ["CCO", "CC(=O)O"]}]},
        STOCK,
    )
    assert result.valid is True
    assert result.reward == 1.0
    assert result.verification_tier == "dataset_supported"
    assert result.metrics["exact_reference_match"] is True


def test_invalid_smiles_is_hard_zero():
    result = RouteVerifier().score_route(
        make_task(),
        {"route": [{"product": "not-smiles", "reactants": ["CCO"]}]},
        STOCK,
    )
    assert result.valid is False
    assert result.reward == 0.0


def test_product_leakage_is_hard_zero():
    result = RouteVerifier().score_route(
        make_task(),
        {
            "route": [
                {
                    "product": "CCOC(C)=O",
                    "reactants": ["CCOC(C)=O", "CCO"],
                }
            ]
        },
        STOCK | {"CCOC(C)=O"},
    )
    assert result.reward == 0.0
    assert any("leak" in reason for reason in result.hard_failures)


def test_missing_terminal_stock_is_hard_zero():
    result = RouteVerifier().score_route(
        make_task(),
        {"route": [{"product": "CCOC(C)=O", "reactants": ["CCO", "CC(=O)O"]}]},
        {"CCO"},
    )
    assert result.reward == 0.0
    assert result.metrics["building_block_completion"] == 0.5


def test_unsupported_structurally_valid_step_is_hard_zero():
    result = RouteVerifier().score_route(
        make_task(),
        {"route": [{"product": "CCOC(C)=O", "reactants": ["CO", "CC(=O)O"]}]},
        STOCK,
    )
    assert result.reward == 0.0
    assert result.step_results[0].support == "none"


def test_unreachable_extra_step_is_hard_zero():
    task = make_task()
    task = RetroTask(
        task_id=task.task_id,
        mode="route_planning",
        target_smiles=task.target_smiles,
        max_steps=2,
        stock_id=task.stock_id,
        split=task.split,
        reference_routes=task.reference_routes,
    )
    result = RouteVerifier().score_route(
        task,
        {
            "route": [
                {"product": "CCOC(C)=O", "reactants": ["CCO", "CC(=O)O"]},
                {"product": "COC=O", "reactants": ["CO", "C=O"]},
            ]
        },
        STOCK | {"C=O"},
    )
    assert result.reward == 0.0
    assert any("unreachable" in reason for reason in result.hard_failures)


def test_trusted_template_accepts_an_alternative_precursor():
    target = canonicalize_smiles("CCO")
    reference = ReactionStep(
        product=target,
        reactants=canonicalize_components(["CCCl", "O"]),
        reaction_id="substitution",
        reaction_smarts="[C:1][Cl,Br:2].[O:3]>>[C:1][O:3]",
    )
    task = RetroTask(
        task_id="template-task",
        mode="single_step",
        target_smiles=target,
        max_steps=1,
        stock_id="test_stock",
        split="eval",
        reference_routes=(ReferenceRoute("reference", (reference,), ()),),
    )
    result = RouteVerifier().score_route(
        task,
        {"route": [{"product": "CCO", "reactants": ["CCBr", "O"]}]},
        STOCK,
    )
    assert result.valid is True
    assert result.verification_tier == "template_supported"
    assert result.reward < 1.0


def test_reaction_class_conflict_is_rejected():
    result = RouteVerifier().score_route(
        make_task(),
        {
            "route": [
                {
                    "product": "CCOC(C)=O",
                    "reactants": ["CCO", "CC(=O)O"],
                    "reaction_class": "Suzuki coupling",
                }
            ]
        },
        STOCK,
    )
    assert result.reward == 0.0
    assert result.step_results[0].checks["reaction_class_compatible"] is False


def test_dataset_match_cannot_bypass_atom_inventory_gate():
    target = canonicalize_smiles("CCO")
    reference = ReactionStep(
        product=target,
        reactants=canonicalize_components(["CC"]),
        reaction_id="broken-observation",
        mapping_status="unmapped",
    )
    task = RetroTask(
        task_id="atom-inventory-task",
        mode="single_step",
        target_smiles=target,
        max_steps=1,
        stock_id="test_stock",
        split="eval",
        reference_routes=(ReferenceRoute("reference", (reference,), ()),),
    )
    result = RouteVerifier().score_route(
        task,
        {"route": [{"product": "CCO", "reactants": ["CC"]}]},
        {"CC"},
    )
    assert result.reward == 0.0
    assert result.step_results[0].support == "dataset_exact"
    assert result.step_results[0].checks["atom_inventory_conserved"] is False
    assert "O:1" in "; ".join(result.step_results[0].errors)
