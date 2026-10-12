from __future__ import annotations

from pathlib import Path

import pytest
from retroenv.chemistry import canonicalize_components, canonicalize_smiles
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask, TaskConstraints
from retroenv.reactions import ReactionLibrary, free_key

FIXTURE_RELEASE = Path(__file__).parent / "fixtures" / "mini-release"

ESTER = "CCOC(C)=O"
NITRO_ESTER = "CCOC(=O)c1ccc([N+](=O)[O-])cc1"
AMINO_ESTER = "CCOC(=O)c1ccc(N)cc1"
STOCK = frozenset(
    canonicalize_smiles(s)
    for s in ("CCO", "CC(=O)O", "CC(=O)Cl", "O=C(O)c1ccc([N+](=O)[O-])cc1", "O=C(Cl)c1ccc([N+](=O)[O-])cc1", "O", "CN")
)
# Radius-1 retro-templates (and their corpus counts) extracted from PaRoutes reactions.
TEMPLATES = {
    "[C:4]-[O;H0;D2;+0:5]-[C;H0;D3;+0:1](-[C:2])=[O;D1;H0:3]>>O-[C;H0;D3;+0:1](-[C:2])=[O;D1;H0:3].[C:4]-[OH;D1;+0:5]": 399,
    "[C:4]-[O;H0;D2;+0:5]-[C;H0;D3;+0:1](-[C;D1;H3:2])=[O;D1;H0:3]>>Cl-[C;H0;D3;+0:1](-[C;D1;H3:2])=[O;D1;H0:3].[C:4]-[OH;D1;+0:5]": 124,
    "[NH2;D1;+0:1]-[c:2]>>O=[N+;H0;D3:1](-[O-])-[c:2]": 9687,
}
# Known corpus reactions; acyl chloride esterification is supported by its template alone.
CORPUS = [
    (ESTER, ("CCO", "CC(=O)O")),
    (NITRO_ESTER, ("CCO", "O=C(O)c1ccc([N+](=O)[O-])cc1")),
    (AMINO_ESTER, (NITRO_ESTER,)),
]


@pytest.fixture(scope="session")
def library() -> ReactionLibrary:
    corpus = [free_key(product, reactants) for product, reactants in CORPUS]
    return ReactionLibrary(corpus, TEMPLATES, ["O", "CCN(CC)CC"])


def step(product: str, *reactants: str) -> ReactionStep:
    return ReactionStep(product=canonicalize_smiles(product), reactants=canonicalize_components(reactants))


def make_task(
    *,
    target: str = ESTER,
    routes: tuple[tuple[ReactionStep, ...], ...] = ((step(ESTER, "CCO", "CC(=O)O"),),),
    max_depth: int = 3,
    split: str = "test_id",
    variant: str = "standard",
    min_routes: int = 1,
    forbidden: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
) -> RetroTask:
    return RetroTask(
        task_id="retro_fixture",
        parent_id="retro_fixture",
        variant=variant,
        target_smiles=canonicalize_smiles(target),
        stock_id="test_stock",
        split=split,
        max_depth=max_depth,
        min_routes=min_routes,
        constraints=TaskConstraints(forbidden_classes=forbidden, excluded_stock=excluded),
        reference_routes=tuple(
            ReferenceRoute(route_id=f"route_{i}", steps=steps, kind="patent", source=({"patent": f"US{i:07d}"},))
            for i, steps in enumerate(routes)
        ),
        difficulty={"min_depth": 1},
    )
