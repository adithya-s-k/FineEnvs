"""Stage 4: candidate targets, their known routes, features and leakage keys.

Filters (from the PaRoutes study): 10-60 heavy atoms, no metal, not itself in
stock, a shortest stock-closed route of 2-8 reactions, and at least one patent
route made only of clean reactions. Patent routes are deduplicated by their
step set across patent families, keeping the union of their patents.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from retroenv.chemistry import scaffold_smiles

from . import similarity
from .paroutes import WORKERS, Route
from .solver import MAX_DEPTH, RouteGraph, Step, is_convergent, steps_depth

MIN_HEAVY, MAX_HEAVY = 10, 60
MIN_DEPTH = 2
MAX_PATENT_DEPTH = 10
PATENT_ROUTES_KEPT = 5
METALS = frozenset(
    "Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Cs Ba "
    "La Ce Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split()
)
NEAR_DUPLICATE = 0.90
NEIGHBOURS = 64


@dataclass
class PatentRoute:
    steps: list[Step]
    patents: list[str]
    depth: int


@dataclass
class Candidate:
    task_id: str
    target: str
    min_depth: int
    witness: list[Step]
    patent_routes: list[PatentRoute]
    features: dict = field(default_factory=dict)
    near_duplicates: list[str] = field(default_factory=list)


def task_id(target: str) -> str:
    return "retro_" + hashlib.sha256(f"retroenv:{target}".encode()).hexdigest()[:16]


def stable_order(salt: str, text: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()


def _molecule_features(smiles: str) -> tuple[str, dict]:
    mol = Chem.MolFromSmiles(smiles)
    ring_info = mol.GetRingInfo()
    complex_ring = (
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0
        or rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0
        or any(len(ring) >= 8 for ring in ring_info.AtomRings())
    )
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    return smiles, {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "metal": any(atom.GetSymbol() in METALS for atom in mol.GetAtoms()),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "complex_ring": complex_ring,
        "stereocentres": len(centres),
        "scaffold": scaffold_smiles(smiles),
    }


def molecule_features(molecules: set[str], workers: int = WORKERS) -> dict[str, dict]:
    with Pool(workers) as pool:
        return dict(pool.imap_unordered(_molecule_features, sorted(molecules), chunksize=500))


def patent_routes(routes: list[Route], clean: set[Step], stock: frozenset[str]) -> list[PatentRoute]:
    grouped: dict[frozenset[Step], PatentRoute] = {}
    for route in routes:
        if not route.steps or not set(route.steps) <= clean or not route.leaves <= stock:
            continue
        key = frozenset(route.steps)
        known = grouped.get(key)
        if known is None:
            depth = steps_depth(route.target, route.steps)
            if depth > MAX_PATENT_DEPTH:
                continue
            grouped[key] = PatentRoute(sorted(set(route.steps)), [route.patent], depth)
        elif route.patent not in known.patents:
            known.patents.append(route.patent)
    ranked = sorted(grouped.values(), key=lambda r: (r.depth, -len(r.patents), r.steps))
    for route in ranked:
        route.patents.sort()
    return ranked[:PATENT_ROUTES_KEPT]


def build_candidates(
    by_target: dict[str, list[Route]],
    graph: RouteGraph,
    clean: set[Step],
    template_count: dict[Step, int],
    features: dict[str, dict],
) -> tuple[list[Candidate], Counter]:
    funnel: Counter = Counter()
    candidates = []
    for target in sorted(by_target):
        funnel["targets"] += 1
        info = features[target]
        if not MIN_HEAVY <= info["heavy_atoms"] <= MAX_HEAVY:
            funnel["drop_heavy_atoms"] += 1
            continue
        if info["metal"]:
            funnel["drop_metal"] += 1
            continue
        if target in graph.stock:
            funnel["drop_target_in_stock"] += 1
            continue
        depth = graph.depth.get(target)
        if depth is None or not MIN_DEPTH <= depth <= MAX_DEPTH:
            funnel["drop_min_depth_outside_2_8" if depth else "drop_unsolvable_with_clean_reactions"] += 1
            continue
        known = patent_routes(by_target[target], clean, graph.stock)
        if not known:
            funnel["drop_no_clean_patent_route"] += 1
            continue
        witness = graph.witness(target)
        steps = witness + [step for route in known for step in route.steps]
        candidates.append(
            Candidate(
                task_id=task_id(target),
                target=target,
                min_depth=depth,
                witness=witness,
                patent_routes=known,
                features={
                    "heavy_atoms": info["heavy_atoms"],
                    "rings": info["rings"],
                    "complex_ring": info["complex_ring"],
                    "stereocentres": info["stereocentres"],
                    "scaffold": info["scaffold"],
                    "convergent": is_convergent(witness, graph.stock)
                    or any(is_convergent(route.steps, graph.stock) for route in known),
                    "rarest_template_count": min(template_count.get(step, 0) for step in steps),
                    "patent_min_depth": known[0].depth,
                    "patent_routes": len(known),
                },
            )
        )
        funnel["kept"] += 1
    return candidates, funnel


def add_neighbour_features(candidates: list[Candidate], workers: int = WORKERS) -> None:
    """Nearest other-patent Tanimoto (novelty) and near-duplicate targets (>= 0.90)."""
    fps = similarity.packed([c.target for c in candidates], workers)
    patents = [{p for r in c.patent_routes for p in r.patents} for c in candidates]
    for i, indexes, values in similarity.nearest(fps, NEIGHBOURS, workers):
        candidate = candidates[i]
        candidate.near_duplicates = [candidates[j].target for j, v in zip(indexes, values) if v >= NEAR_DUPLICATE]
        other = next((v for j, v in zip(indexes, values) if not patents[i] & patents[j]), values[-1])
        candidate.features["nn_similarity"] = round(float(other), 4)


def tier(features: dict, min_depth: int) -> str:
    points = (2 if min_depth >= 4 else 1 if min_depth == 3 else 0) + sum(
        (
            features["nn_similarity"] < 0.5,
            features["rarest_template_count"] <= 10,
            features["convergent"],
            features["complex_ring"],
        )
    )
    return "easy" if points <= 1 else "medium" if points <= 3 else "hard"
