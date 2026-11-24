"""Leakage keys shared by the split builder and the precedent tool.

Two tasks leak into each other when they share a patent, a route molecule
(target or intermediate), a reaction, a scaffold group, or are Morgan
near-duplicates. The builder keeps held-out tasks key-disjoint from train, and
``PrecedentIndex`` hides the same keys from a train task's own searches so that
training never rewards looking up an answer evaluation cannot offer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .chemistry import canonicalize_smiles, scaffold_group_key
from .models import RetroTask
from .reactions import free_key

Key = tuple[str, str]


@dataclass(frozen=True)
class LeakageRules:
    generic_scaffolds: frozenset[str] = frozenset()
    near_duplicate_threshold: float = 0.90

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any] | str | Path | None) -> "LeakageRules":
        if manifest is None:
            return cls()
        if not isinstance(manifest, dict):
            path = Path(manifest)
            if not path.exists():
                return cls()
            manifest = json.loads(path.read_text())
        leakage = manifest.get("leakage", {})
        return cls(
            generic_scaffolds=frozenset(leakage.get("generic_scaffolds", ())),
            near_duplicate_threshold=float(leakage.get("near_duplicate_threshold", 0.90)),
        )

    def scaffold(self, smiles: str) -> str:
        return scaffold_group_key(smiles, exact_single_ring=True, generic=self.generic_scaffolds)


def molecule_keys(molecules: Iterable[str], rules: LeakageRules) -> set[Key]:
    keys: set[Key] = set()
    for smiles in molecules:
        keys.add(("molecule", smiles))
        keys.add(("scaffold", rules.scaffold(smiles)))
    return keys


def reaction_key(product: str, reactants: Iterable[str]) -> Key:
    product_key, reactant_key = free_key(product, reactants)
    return ("reaction", f"{product_key}>>{'.'.join(reactant_key)}")


def task_molecules(task: RetroTask) -> set[str]:
    """The target and every known-route intermediate (leaves are building blocks, not keys)."""
    molecules = {canonicalize_smiles(task.target_smiles)}
    for route in task.reference_routes:
        molecules.update(canonicalize_smiles(step.product) for step in route.steps)
    return molecules


def task_keys(task: RetroTask, rules: LeakageRules) -> set[Key]:
    """Patents count only for the target's own patent routes; a witness route's reactions come
    from filings about other targets, whose chemistry the molecule and reaction keys already cover."""
    keys = molecule_keys(task_molecules(task), rules)
    for route in task.reference_routes:
        if route.kind == "patent":
            keys.update(("patent", str(source["patent"])) for source in route.source if source.get("patent"))
        keys.update(reaction_key(step.product, step.reactants) for step in route.steps)
    return keys
