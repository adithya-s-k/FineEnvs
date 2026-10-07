"""Stage 6: constrained variants, each kept only with a stock-closed witness that satisfies it.

* ``max_depth``: the depth cap is the shortest known depth while the patent route is longer.
* ``forbidden_class``: a reliably labelled class used by the patent route may not be used.
* ``restricted_stock``: the patent route's largest building block is unavailable.
* ``diversity``: two (sometimes three) routes with different first disconnections are required.

Variants inherit their parent's split, so splitting happens before generation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool

from rdkit import Chem
from retroenv.chemistry import canonicalize_components
from retroenv.classes import CONSTRAINABLE_CLASSES

from .paroutes import WORKERS
from .solver import MAX_DEPTH, RouteGraph, Step, steps_depth, steps_leaves
from .targets import Candidate, stable_order

DEPTH_SLACK = 2
MAX_TASK_DEPTH = 10
VARIANT_SHARE = 0.15
MIN_BUILDING_BLOCK_HEAVY = 6


@dataclass
class TaskSpec:
    candidate: Candidate
    variant: str
    max_depth: int
    witnesses: list[list[Step]]
    min_routes: int = 1
    forbidden_classes: tuple[str, ...] = ()
    excluded_stock: tuple[str, ...] = ()
    constrained_min_depth: int = 0
    extra: dict = field(default_factory=dict)


def budget(depth: int) -> int:
    return min(MAX_TASK_DEPTH, depth + DEPTH_SLACK)


def valid_witness(target: str, steps: list[Step], stock: frozenset[str], max_depth: int) -> bool:
    return (
        bool(steps)
        and not any(target in reactants for _, reactants in steps)
        and steps_leaves(target, steps) <= stock
        and steps_depth(target, steps) <= max_depth
    )


_GRAPH: RouteGraph | None = None
_CLASS: dict[Step, str] = {}


def _heavy(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0


def _variants(candidate: Candidate) -> list[TaskSpec]:
    graph = _GRAPH
    assert graph is not None
    target, depth = candidate.target, candidate.min_depth
    patent = candidate.patent_routes[0]
    specs: list[TaskSpec] = []
    indexes = graph.subgraph(target)

    if patent.depth > depth:
        specs.append(TaskSpec(candidate, "max_depth", depth, [candidate.witness], constrained_min_depth=depth))

    # A constraint must break both obvious answers (the shortest witness and the patent route),
    # so the agent has to find a genuinely different route.
    known = (candidate.witness, patent.steps)
    used = [{_CLASS[step] for step in steps} for steps in known]
    root_first = sorted(candidate.witness, key=lambda step: step[0] != target)
    classes = [c for c in dict.fromkeys(_CLASS[s] for s in root_first) if c in CONSTRAINABLE_CLASSES and all(c in u for u in used)]
    for reaction_class in classes:
        solved = graph.solve(target, forbidden=frozenset({reaction_class}), indexes=indexes)
        if solved and solved[0] <= MAX_DEPTH and valid_witness(target, solved[1], graph.stock, budget(solved[0])):
            specs.append(
                TaskSpec(candidate, "forbidden_class", budget(solved[0]), [solved[1]],
                         forbidden_classes=(reaction_class,), constrained_min_depth=solved[0])
            )
            break

    blocks = [
        sorted((leaf for leaf in steps_leaves(target, steps) if _heavy(leaf) >= MIN_BUILDING_BLOCK_HEAVY), key=lambda leaf: (-_heavy(leaf), leaf))
        for steps in known
    ]
    shared = [leaf for leaf in blocks[0] if leaf in blocks[1]]
    options = [frozenset({leaf}) for leaf in shared[:3]]
    if blocks[0] and blocks[1]:
        options.append(frozenset({blocks[0][0], blocks[1][0]}))
    for excluded in options:
        solved = graph.solve(target, excluded=excluded, indexes=indexes)
        if solved and solved[0] <= MAX_DEPTH and valid_witness(target, solved[1], graph.stock - excluded, budget(solved[0])):
            specs.append(
                TaskSpec(candidate, "restricted_stock", budget(solved[0]), [solved[1]],
                         excluded_stock=tuple(sorted(excluded)), constrained_min_depth=solved[0])
            )
            break

    cuts = graph.first_cuts(target, budget(depth))
    witnesses, cut_sets = [], set()
    for _, index in cuts:
        reactants = canonicalize_components(graph.reactions[index].reactants, isomeric=False)
        if reactants in cut_sets:
            continue
        steps = graph.witness(target, first=index)
        if valid_witness(target, steps, graph.stock, budget(depth)):
            witnesses.append(steps)
            cut_sets.add(reactants)
    if len(witnesses) >= 2:
        wanted = 3 if len(witnesses) >= 3 and int(stable_order("diversity", target), 16) % 3 == 0 else 2
        chosen = witnesses[:wanted]
        specs.append(
            TaskSpec(candidate, "diversity", budget(depth), chosen, min_routes=wanted,
                     constrained_min_depth=max(steps_depth(target, steps) for steps in chosen))
        )
    return specs


def generate(candidates: list[Candidate], graph: RouteGraph, step_class: dict[Step, str],
             workers: int = WORKERS) -> list[TaskSpec]:
    global _GRAPH, _CLASS
    _GRAPH, _CLASS = graph, step_class
    with Pool(workers) as pool:
        return [spec for specs in pool.imap(_variants, candidates, chunksize=64) for spec in specs]


def standard(candidate: Candidate) -> TaskSpec:
    return TaskSpec(candidate, "standard", budget(candidate.min_depth), [candidate.witness],
                    constrained_min_depth=candidate.min_depth)


def cap_variants(specs: list[TaskSpec], split_of: dict[str, str]) -> tuple[list[TaskSpec], dict]:
    """Keep at most VARIANT_SHARE x parents of each variant type per split, in stable order."""
    parents = Counter(split_of.values())
    grouped: dict[tuple[str, str], list[TaskSpec]] = defaultdict(list)
    for spec in specs:
        grouped[(split_of[spec.candidate.task_id], spec.variant)].append(spec)
    kept, report = [], {}
    for (split, variant), group in sorted(grouped.items()):
        limit = math.ceil(VARIANT_SHARE * parents[split])
        group.sort(key=lambda spec: stable_order(variant, spec.candidate.target))
        kept.extend(group[:limit])
        report[f"{split}/{variant}"] = {"available": len(group), "kept": min(limit, len(group))}
    return kept, report
