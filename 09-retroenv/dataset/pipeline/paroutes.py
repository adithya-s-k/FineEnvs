"""Stage 1: flatten the PaRoutes archive into canonical reactions and routes.

Every reaction node becomes (product, reactants, mapped rsmi, patent); molecules
are canonicalized once in parallel. Routes keep only their step keys and leaves.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from retroenv.chemistry import canonicalize_smiles

# The container reports the host's cores and memory, not the pod's (32 vCPU, 125 GB); every
# forked worker can copy the parent's memory, so keep this small.
WORKERS = int(os.getenv("RETROENV_WORKERS", "8"))


@dataclass
class Reaction:
    """One stereo-aware unique step; provenance merged over every archive occurrence."""

    product: str
    reactants: tuple[str, ...]
    rsmi: str
    patents: set[str] = field(default_factory=set)
    routes: int = 0


@dataclass
class Route:
    index: int
    target: str
    patent: str
    steps: list[tuple[str, tuple[str, ...]]]
    leaves: set[str]
    depth: int


def _canonical_pair(smiles: str) -> tuple[str, str | None]:
    try:
        return smiles, canonicalize_smiles(smiles)
    except Exception:
        return smiles, None


def canonicalize_all(smiles: set[str], workers: int = WORKERS) -> dict[str, str | None]:
    with Pool(workers) as pool:
        return dict(pool.imap_unordered(_canonical_pair, sorted(smiles), chunksize=2000))


def _walk(node: dict, out: list[tuple[str, list[str], str, str]]) -> None:
    for reaction in node.get("children", []):
        children = reaction.get("children", [])
        metadata = reaction.get("metadata", {})
        patent = str(metadata.get("ID", "")).split(";")[0]
        out.append((node["smiles"], [child["smiles"] for child in children], metadata.get("rsmi", ""), patent))
        for child in children:
            _walk(child, out)


def _depth(node: dict) -> int:
    return max(
        (1 + max((_depth(c) for c in r.get("children", [])), default=0) for r in node.get("children", [])), default=0
    )


def _leaves(node: dict, out: set[str]) -> None:
    if not node.get("children"):
        out.add(node["smiles"])
    for reaction in node.get("children", []):
        for child in reaction.get("children", []):
            _leaves(child, out)


def load_archive(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def flatten(
    archive: list[dict],
) -> tuple[dict[tuple[str, tuple[str, ...]], Reaction], list[Route], dict[str, str | None]]:
    """Canonical unique reactions, canonical routes, and the raw->canonical molecule map."""
    raw_routes = []
    molecules: set[str] = set()
    for index, tree in enumerate(archive):
        steps: list[tuple[str, list[str], str, str]] = []
        _walk(tree, steps)
        leaves: set[str] = set()
        _leaves(tree, leaves)
        raw_routes.append((index, tree["smiles"], steps, leaves, _depth(tree)))
        molecules.add(tree["smiles"])
        for product, reactants, _, _ in steps:
            molecules.add(product)
            molecules.update(reactants)
    canonical = canonicalize_all(molecules)

    reactions: dict[tuple[str, tuple[str, ...]], Reaction] = {}
    routes: list[Route] = []
    for index, target, steps, leaves, depth in raw_routes:
        canonical_steps = []
        broken = canonical[target] is None
        patent = steps[0][3] if steps else ""
        for product, reactants, rsmi, step_patent in steps:
            product_c = canonical[product]
            reactants_c = [canonical[r] for r in reactants]
            if product_c is None or None in reactants_c or not reactants_c:
                broken = True
                continue
            key = (product_c, tuple(sorted(reactants_c)))
            reaction = reactions.get(key)
            if reaction is None:
                reaction = reactions[key] = Reaction(product_c, key[1], rsmi)
            reaction.patents.add(step_patent)
            reaction.routes += 1
            canonical_steps.append(key)
        canonical_leaves = {canonical[leaf] for leaf in leaves}
        if broken or None in canonical_leaves:
            continue
        routes.append(Route(index, canonical[target], patent, canonical_steps, canonical_leaves, depth))
    return reactions, routes, canonical


def group_routes(routes: list[Route]) -> dict[str, list[Route]]:
    by_target: dict[str, list[Route]] = defaultdict(list)
    for route in routes:
        by_target[route.target].append(route)
    return by_target
