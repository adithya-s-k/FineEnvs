from __future__ import annotations

from retroenv.chemistry import canonicalize_components, canonicalize_smiles
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask


def make_task(
    *,
    target: str = "CCOC(C)=O",
    reactants: tuple[str, ...] = ("CCO", "CC(=O)O"),
    reaction_smarts: str | None = None,
    split: str = "train",
) -> RetroTask:
    target = canonicalize_smiles(target)
    step = ReactionStep(
        product=target,
        reactants=canonicalize_components(reactants),
        reaction_class="esterification",
        reaction_id="rxn_fixture",
        reaction_smarts=reaction_smarts,
        mapping_status="complete",
    )
    return RetroTask(
        task_id="retro_fixture",
        mode="single_step",
        target_smiles=target,
        max_steps=1,
        stock_id="test_stock",
        split=split,
        reference_routes=(
            ReferenceRoute(
                route_id="route_fixture",
                steps=(step,),
                source=(
                    {
                        "name": "fixture",
                        "license": "CC0-1.0",
                        "record_id": "1",
                        "group_id": "group-1",
                    },
                ),
            ),
        ),
        difficulty={"scaffold": "acyclic"},
    )

