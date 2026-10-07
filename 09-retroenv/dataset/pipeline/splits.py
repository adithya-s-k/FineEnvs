"""Stage 5: leakage-free held-out selection by blocklist.

Shared patents and intermediates join most PaRoutes targets into one giant
component, so whole components cannot be assigned to splits. Instead held-out
tasks are drawn one by one (test_hard, then test_id, then dev) against depth
quotas; each must share no leakage key or near-duplicate with any earlier
held-out task, and candidates that would block unusually many train targets are
skipped. Train is every remaining candidate that touches no held-out key and is
not a near-duplicate of any held-out molecule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from retroenv.chemistry import is_single_ring_scaffold

from . import similarity
from .paroutes import WORKERS
from .targets import NEAR_DUPLICATE, Candidate, stable_order

Key = tuple[str, str]
HELD_OUT = ("test_hard", "test_id", "dev")
SPLIT_SIZE = 1000
GENERIC_SCAFFOLD_TARGETS = 100
COST_PERCENTILE = 95
# Shares of each held-out split by shortest-route depth; unfilled strata fall back to neighbours.
DEPTH_QUOTAS = {
    "test_hard": {4: 0.40, 5: 0.30, 6: 0.18, 7: 0.08, 8: 0.04},
    "test_id": {2: 0.40, 3: 0.30, 4: 0.17, 5: 0.08, 6: 0.03, 7: 0.01, 8: 0.01},
    "dev": {2: 0.40, 3: 0.30, 4: 0.17, 5: 0.08, 6: 0.03, 7: 0.01, 8: 0.01},
}
HARD_RULE = {"min_depth": 4, "max_nn_similarity": 0.6, "rare_template": 10}


def is_hard(candidate: Candidate) -> bool:
    f = candidate.features
    return (
        candidate.min_depth >= HARD_RULE["min_depth"]
        and f["nn_similarity"] < HARD_RULE["max_nn_similarity"]
        and (f["convergent"] or f["complex_ring"] or f["rarest_template_count"] <= HARD_RULE["rare_template"])
    )


@dataclass
class ScaffoldRules:
    generic: frozenset[str]
    scaffold_of: dict[str, str]
    _single: dict[str, bool] = field(default_factory=dict)

    def key(self, smiles: str) -> str:
        scaffold = self.scaffold_of[smiles]
        if scaffold not in self._single:
            self._single[scaffold] = is_single_ring_scaffold(scaffold)
        if scaffold in self.generic or self._single[scaffold]:
            return f"exact:{smiles}"
        return scaffold


def generic_scaffolds(candidates: list[Candidate]) -> frozenset[str]:
    counts = Counter(c.features["scaffold"] for c in candidates)
    return frozenset(s for s, n in counts.items() if n >= GENERIC_SCAFFOLD_TARGETS and not s.startswith("acyclic:"))


def route_molecules(candidate: Candidate) -> set[str]:
    molecules = {candidate.target}
    molecules.update(product for product, _ in candidate.witness)
    for route in candidate.patent_routes:
        molecules.update(product for product, _ in route.steps)
    return molecules


def candidate_keys(candidate: Candidate, rules: ScaffoldRules, free_reaction: dict) -> set[Key]:
    """Route molecules, their scaffold groups, every known reaction, and the patents of the
    target's own patent routes (a witness reaction's other patents describe other targets)."""
    keys: set[Key] = set()
    for smiles in route_molecules(candidate):
        keys.add(("molecule", smiles))
        keys.add(("scaffold", rules.key(smiles)))
    steps = list(candidate.witness) + [step for route in candidate.patent_routes for step in route.steps]
    keys.update(("reaction", free_reaction[step]) for step in steps)
    for route in candidate.patent_routes:
        keys.update(("patent", patent) for patent in route.patents)
    return keys


@dataclass
class Assignment:
    split_of: dict[str, str]
    held_out_keys: set[Key]
    held_out_molecules: set[str]
    report: dict


def assign(candidates: list[Candidate], keys: dict[str, set[Key]], size: int = SPLIT_SIZE) -> Assignment:
    by_key: dict[Key, list[str]] = defaultdict(list)
    for candidate in candidates:
        for key in keys[candidate.task_id]:
            by_key[key].append(candidate.task_id)
    near = {c.task_id: set(c.near_duplicates) for c in candidates}
    target_of = {c.task_id: c.target for c in candidates}
    cost = {
        c.task_id: len({other for key in keys[c.task_id] for other in by_key[key]}) - 1 + len(near[c.task_id])
        for c in candidates
    }
    cap = float(np.percentile(list(cost.values()), COST_PERCENTILE))
    split_of: dict[str, str] = {}
    taken: set[Key] = set()
    held_targets: set[str] = set()
    report: dict = {"cost_cap": cap, "splits": {}}

    def eligible(candidate: Candidate) -> bool:
        return (
            candidate.task_id not in split_of
            and cost[candidate.task_id] <= cap
            and not keys[candidate.task_id] & taken
            and not near[candidate.task_id] & held_targets
        )

    for split in HELD_OUT:
        pool = [c for c in candidates if is_hard(c)] if split == "test_hard" else list(candidates)
        pool.sort(key=lambda c: stable_order(split, c.target))
        quotas = _quotas(DEPTH_QUOTAS[split], size)
        strata: dict[int, list[Candidate]] = defaultdict(list)
        for candidate in pool:
            strata[candidate.min_depth].append(candidate)
        picked: Counter = Counter()

        def take(candidate: Candidate) -> None:
            split_of[candidate.task_id] = split
            taken.update(keys[candidate.task_id])
            held_targets.add(candidate.target)
            picked[candidate.min_depth] += 1

        for depth in sorted(quotas, key=lambda d: len(strata[d])):  # rare depths choose first
            for candidate in strata[depth]:
                if picked[depth] >= quotas[depth]:
                    break
                if eligible(candidate):
                    take(candidate)
        for candidate in pool:  # fill any shortfall in stable order
            if sum(picked.values()) >= size:
                break
            if eligible(candidate):
                take(candidate)
        report["splits"][split] = {
            "selected": sum(picked.values()),
            "by_min_depth": dict(sorted(picked.items())),
            "pool": len(pool),
        }

    held_out_molecules = set(held_targets)
    for candidate in candidates:
        if candidate.task_id in split_of:
            held_out_molecules |= route_molecules(candidate)
    blocked_near = near_duplicate_hits([target_of[c.task_id] for c in candidates], held_out_molecules)
    barred = Counter()
    for candidate in candidates:
        if candidate.task_id in split_of:
            continue
        if keys[candidate.task_id] & taken:
            barred["shares_a_held_out_key"] += 1
        elif near[candidate.task_id] & held_targets or candidate.target in blocked_near:
            barred["near_duplicate_of_held_out"] += 1
        else:
            split_of[candidate.task_id] = "train"
    report["splits"]["train"] = {"selected": sum(v == "train" for v in split_of.values())}
    report["barred_from_train"] = dict(barred)
    return Assignment(split_of, taken, held_out_molecules, report)


def extend_held_out(
    assignment: Assignment, candidates: list[Candidate], keys: dict[str, set[Key]], extra: set[Key], molecules: set[str]
) -> None:
    """Add keys that held-out variant witnesses introduced, and bar train targets that touch them."""
    assignment.held_out_keys |= extra
    new_molecules = molecules - assignment.held_out_molecules
    assignment.held_out_molecules |= molecules
    train = [c for c in candidates if assignment.split_of.get(c.task_id) == "train"]
    near = near_duplicate_hits([c.target for c in train], new_molecules) if new_molecules else set()
    barred = 0
    for candidate in train:
        if keys[candidate.task_id] & assignment.held_out_keys or candidate.target in near:
            del assignment.split_of[candidate.task_id]
            barred += 1
    assignment.report["barred_by_held_out_variants"] = barred
    assignment.report["splits"]["train"] = {"selected": sum(v == "train" for v in assignment.split_of.values())}


def _quotas(shares: dict[int, float], size: int) -> dict[int, int]:
    raw = {depth: share * size for depth, share in shares.items()}
    quotas = {depth: int(value) for depth, value in raw.items()}
    for depth in sorted(raw, key=lambda d: raw[d] - quotas[d], reverse=True)[: size - sum(quotas.values())]:
        quotas[depth] += 1
    return quotas


def near_duplicate_hits(smiles: list[str], queries: set[str], workers: int = WORKERS) -> set[str]:
    """Members of ``smiles`` within Tanimoto 0.90 of any query molecule."""
    found = similarity.above(similarity.packed(sorted(queries), workers), similarity.packed(smiles, workers), NEAR_DUPLICATE, workers)
    return {smiles[i] for i in found}
