from __future__ import annotations

import json

from retroenv.environment import RetroRouteSession
from retroenv.evaluation import evaluate
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask
from retroenv.store import TaskStore
from retroenv.taskgen import assign_strict_splits, audit_splits

from conftest import make_task


def test_observation_does_not_expose_references():
    session = RetroRouteSession()
    observation = session.reset(make_task(), {"CCO", "CC(=O)O"})
    rendered = json.dumps(observation)
    assert "reference_routes" not in rendered
    assert "rxn_fixture" not in rendered


def test_multiturn_validation_and_submission():
    session = RetroRouteSession(max_tool_calls=4)
    session.reset(make_task(), {"CCO", "CC(=O)O"})
    assert session.inspect_molecule("CCOC(C)=O")["valid"]
    candidate = session.propose_disconnection(
        "CCOC(C)=O", ["CCO", "CC(=O)O"], "esterification"
    )
    assert candidate["structurally_valid"]
    validation = session.validate_step(
        "CCOC(C)=O", ["CCO", "CC(=O)O"], "esterification"
    )
    assert validation["valid"]
    final = session.submit_route(
        {"route": [{"product": "CCOC(C)=O", "reactants": ["CCO", "CC(=O)O"]}]}
    )
    assert final["done"] is True
    assert final["score"]["reward"] == 1.0
    assert session.submit_route({"route": []})["error"]


def test_tool_budget_cannot_be_bypassed_but_submit_remains_available():
    session = RetroRouteSession(max_tool_calls=1)
    session.reset(make_task(), {"CCO", "CC(=O)O"})
    session.inspect_molecule("CCO")
    blocked = session.inspect_molecule("CCO")
    assert "budget exhausted" in blocked["error"]
    final = session.submit_route(
        {"route": [{"product": "CCOC(C)=O", "reactants": ["CCO", "CC(=O)O"]}]}
    )
    assert final["score"]["valid"] is True


def test_strict_split_keeps_same_scaffold_and_source_group_together():
    first = make_task(target="CCOC(C)=O")
    aromatic_a = make_task(target="CNC(=O)c1ccccc1", reactants=("CN", "O=C(O)c1ccccc1"))
    aromatic_b = make_task(target="CC(=O)Nc1ccccc1", reactants=("CC(=O)O", "Nc1ccccc1"))
    tasks = []
    for index, task in enumerate((first, aromatic_a, aromatic_b)):
        tasks.append(
            RetroTask(
                task_id=f"task-{index}",
                mode=task.mode,
                target_smiles=task.target_smiles,
                max_steps=task.max_steps,
                stock_id=task.stock_id,
                split="unassigned",
                reference_routes=task.reference_routes,
            )
        )
    split, manifest = assign_strict_splits(tasks, near_duplicate_threshold=0)
    by_id = {task.task_id: task.split for task in split}
    assert by_id["task-1"] == by_id["task-2"]
    assert manifest["audit"]["passed"] is True
    assert audit_splits(split)["overlaps"] == []


def test_strict_split_keeps_routes_with_a_shared_reaction_in_one_split():
    first = make_task(target="CCOC(C)=O")
    second = make_task(target="CNC(=O)c1ccccc1", reactants=("CN", "O=C(O)c1ccccc1"))
    tasks = []
    for index, task in enumerate((first, second)):
        original = task.reference_routes[0].steps[0]
        shared_step = ReactionStep(
            product=original.product,
            reactants=original.reactants,
            reaction_id="shared-reaction-id",
            mapping_status="complete",
        )
        tasks.append(
            RetroTask(
                task_id=f"shared-{index}",
                mode="route_planning",
                target_smiles=task.target_smiles,
                max_steps=1,
                stock_id=task.stock_id,
                split="unassigned",
                reference_routes=(
                    ReferenceRoute(f"route-{index}", (shared_step,), ()),
                ),
            )
        )
    split, manifest = assign_strict_splits(tasks, near_duplicate_threshold=0)
    assert len({task.split for task in split}) == 1
    assert manifest["audit"]["passed"] is True


def test_strict_split_keeps_shared_hidden_intermediate_in_one_split():
    shared_intermediate = ReactionStep(
        product="CCN",
        reactants=("CCBr", "N"),
        reaction_id="ethylamine-a",
    )
    first = RetroTask(
        task_id="intermediate-a",
        mode="route_planning",
        target_smiles="CCNC(=O)OCC",
        max_steps=2,
        stock_id="test_stock",
        split="unassigned",
        reference_routes=(
            ReferenceRoute(
                "route-intermediate-a",
                (
                    ReactionStep(
                        product="CCNC(=O)OCC",
                        reactants=("CCN", "CCOC(=O)Cl"),
                        reaction_id="carbamate-a",
                    ),
                    shared_intermediate,
                ),
                (),
            ),
        ),
    )
    second = RetroTask(
        task_id="intermediate-b",
        mode="route_planning",
        target_smiles="CCNC(=O)c1ccccc1",
        max_steps=2,
        stock_id="test_stock",
        split="unassigned",
        reference_routes=(
            ReferenceRoute(
                "route-intermediate-b",
                (
                    ReactionStep(
                        product="CCNC(=O)c1ccccc1",
                        reactants=("CCN", "O=C(Cl)c1ccccc1"),
                        reaction_id="amide-b",
                    ),
                    ReactionStep(
                        product="CCN",
                        reactants=("CCBr", "N"),
                        reaction_id="ethylamine-b",
                    ),
                ),
                (),
            ),
        ),
    )

    split, manifest = assign_strict_splits(
        [first, second], near_duplicate_threshold=0
    )
    assert len({task.split for task in split}) == 1
    assert manifest["audit"]["passed"] is True


def test_task_store_public_view_strips_private_routes(tmp_path):
    task = make_task()
    tasks_dir, stocks_dir = tmp_path / "tasks", tmp_path / "stocks"
    tasks_dir.mkdir()
    stocks_dir.mkdir()
    (tasks_dir / "train.jsonl").write_text(json.dumps(task.to_dict()) + "\n")
    (stocks_dir / "test_stock.smi").write_text("CCO\nCC(=O)O\n")
    store = TaskStore(tasks_dir, stocks_dir)
    public = store.public_task("train", 0)
    assert "reference_routes" not in public
    assert len(store.stock("test_stock")) == 2


def test_evaluation_keeps_missing_tasks_in_denominator(tmp_path):
    task = make_task()
    second = RetroTask(
        task_id="retro_missing",
        mode=task.mode,
        target_smiles=task.target_smiles,
        max_steps=task.max_steps,
        stock_id=task.stock_id,
        split="train",
        reference_routes=task.reference_routes,
    )
    tasks_dir, stocks_dir = tmp_path / "tasks", tmp_path / "stocks"
    tasks_dir.mkdir()
    stocks_dir.mkdir()
    (tasks_dir / "train.jsonl").write_text(
        json.dumps(task.to_dict()) + "\n" + json.dumps(second.to_dict()) + "\n"
    )
    (stocks_dir / "test_stock.smi").write_text("CCO\nCC(=O)O\n")
    result = evaluate(
        TaskStore(tasks_dir, stocks_dir),
        [
            {
                "task_id": task.task_id,
                "attempts": [
                    {
                        "route": {
                            "route": [
                                {
                                    "product": "CCOC(C)=O",
                                    "reactants": ["CCO", "CC(=O)O"],
                                }
                            ]
                        }
                    }
                ],
            }
        ],
        ks=(1,),
    )
    assert result["tasks"] == 2
    assert result["pass@1"] == 0.5
    assert result["top1_route_validity"] == 0.5
    assert result["top1_step_validity"] == 0.5
