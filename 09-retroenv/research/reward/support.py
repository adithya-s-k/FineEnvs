"""Step support under stricter template rules than the shipped library.

The shipped `ReactionLibrary` accepts a step as soon as one retro-template seen
>= min_count times turns the product into a subset of the proposed reactants.
That only constrains the atoms at the bond being made, so it accepts steps at
the wrong ring position or on the wrong functional group.

This module re-runs each matching template *forward* over the precursors it
proposed and records how many distinct products come back. A regioselective
step gives exactly one; a wrong-site step gives several. Templates are visited
in descending corpus frequency, so one pass answers every
(min_count, forward rule) configuration at once.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from rdchiral.main import rdchiralReactants
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from retroenv.chemistry import ChemistryError, canonicalize_components, canonicalize_smiles
from retroenv.reactions import ReactionLibrary, free_key, run_template

MAX_MATCHES_KEPT = 128
MAX_FORWARD_OUTCOMES = 128


@dataclass(frozen=True)
class TemplateMatch:
    count: int
    forward_product_count: int
    forward_hits_target: bool

    def accepts(self, min_count: int, rule: str) -> bool:
        if self.count < min_count:
            return False
        if rule == "any":
            return True
        if not self.forward_hits_target:
            return False
        return self.forward_product_count <= (1 if rule == "sole" else 2)


@dataclass(frozen=True)
class StepProfile:
    """Everything a reward variant needs to decide support for one step."""

    product: str
    reactants: tuple[str, ...]
    in_corpus: bool
    matches: tuple[TemplateMatch, ...]
    heavy_atoms: int = 0
    error: str | None = None

    def supported(self, min_count: int = 5, rule: str = "any") -> bool:
        return self.in_corpus or any(match.accepts(min_count, rule) for match in self.matches)

    def template_count(self, min_count: int = 5, rule: str = "any") -> int:
        return max((m.count for m in self.matches if m.accepts(min_count, rule)), default=0)

    def to_dict(self) -> dict:
        return {
            "product": self.product,
            "reactants": list(self.reactants),
            "in_corpus": self.in_corpus,
            "heavy_atoms": self.heavy_atoms,
            "matches": [[m.count, m.forward_product_count, m.forward_hits_target] for m in self.matches],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "StepProfile":
        return cls(
            product=value["product"],
            reactants=tuple(value["reactants"]),
            in_corpus=value["in_corpus"],
            matches=tuple(TemplateMatch(c, n, bool(h)) for c, n, h in value["matches"]),
            heavy_atoms=value.get("heavy_atoms", 0),
            error=value.get("error"),
        )


def forward_products(retro_smarts: str, precursors: Iterable[str]) -> frozenset[str]:
    """Canonical products of running a retro template in the forward direction."""
    product_side, reactant_side = retro_smarts.split(">>", 1)
    reaction = _forward_reaction(f"{reactant_side}>>{product_side}")
    if reaction is None:
        return frozenset()
    try:
        mols = tuple(Chem.MolFromSmiles(item) for item in precursors)
    except Exception:
        return frozenset()
    if any(mol is None for mol in mols) or reaction.GetNumReactantTemplates() != len(mols):
        return frozenset()
    found: set[str] = set()
    orderings = itertools.permutations(mols) if len(mols) <= 3 else (mols,)
    for ordered in orderings:
        try:
            outcomes = reaction.RunReactants(ordered, maxProducts=MAX_FORWARD_OUTCOMES)
        except Exception:
            continue
        for outcome in outcomes:
            for product in outcome:
                try:
                    found.add(canonicalize_smiles(Chem.MolToSmiles(product), isomeric=False))
                except Exception:
                    continue
            if len(found) > MAX_FORWARD_OUTCOMES:
                return frozenset(found)
    return frozenset(found)


def profile_step(library: ReactionLibrary, product: str, reactants: Iterable[str]) -> StepProfile:
    """Corpus membership plus every template that would accept this step."""
    reactant_list = [str(item) for item in reactants]
    try:
        key = free_key(product, reactant_list)
    except ChemistryError as exc:
        return StepProfile("", (), False, (), error=str(exc))
    product_key, reactant_key = key
    if not reactant_key or product_key in reactant_key:
        return StepProfile(product_key, reactant_key, False, (), error="degenerate step")
    heavy = _heavy_atoms(product_key)
    if library._in_corpus(key, frozenset()):
        return StepProfile(product_key, reactant_key, True, (), heavy_atoms=heavy)
    extras = frozenset(smiles for smiles in reactant_key if smiles in library.reagents)
    try:
        # ReactionLibrary._support runs the templates on the product as given, stereo
        # included; only the acceptance comparison is stereo-free. Passing the stereo-free
        # key instead lets templates match that the shipped library rejects.
        matches = _template_matches(library, product, product_key, reactant_key, extras)
    except Exception as exc:  # rdchiral and RunReactants both fail on exotic valences
        return StepProfile(product_key, reactant_key, False, (), heavy_atoms=heavy, error=str(exc)[:200])
    return StepProfile(product_key, reactant_key, False, matches, heavy_atoms=heavy)


def _template_matches(
    library: ReactionLibrary,
    raw_product: str,
    product_key: str,
    reactants: tuple[str, ...],
    extras: frozenset[str],
) -> tuple[TemplateMatch, ...]:
    index = library.index
    product = rdchiralReactants(canonicalize_smiles(raw_product))
    reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactants]
    proposed = set(reactants)
    present: dict[int, bool] = {}
    matches: list[TemplateMatch] = []
    for template_id in index.matching(product.reactants_achiral):
        if len(matches) >= MAX_MATCHES_KEPT:
            break
        template_id = int(template_id)
        if not index._fragments_present(template_id, reactant_mols, present):
            continue
        for outcome in run_template(index._reaction(template_id), product):
            try:
                produced = canonicalize_components(outcome, isomeric=False)
            except ChemistryError:
                continue
            if not produced or not set(produced) <= proposed or not (proposed - set(produced)) <= extras:
                continue
            products = forward_products(index.smarts[template_id], produced)
            matches.append(
                TemplateMatch(
                    count=int(index.counts[template_id]),
                    forward_product_count=len(products),
                    forward_hits_target=product_key in products,
                )
            )
            break
    return tuple(matches)


@lru_cache(maxsize=4096)
def _forward_reaction(smarts: str) -> rdChemReactions.ChemicalReaction | None:
    try:
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
        rdChemReactions.SanitizeRxn(reaction)
    except Exception:
        return None
    return reaction


@lru_cache(maxsize=100_000)
def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0
