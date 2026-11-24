"""Stage 2: annotate unique reactions and build the frozen reaction library.

Each stereo-aware reaction gets its stereo-free key, reaction class, rdchiral
radius-1 retro-template and two quality flags read from the atom map:
``maps_missing`` (product atoms with no mapped source) and ``stereo_lost`` (a
mapped stereocentre that loses its configuration). Flagged reactions stay in the
library corpus but never enter the route graph used to build tasks.
"""

from __future__ import annotations

import io
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from multiprocessing import Pool

from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem, RDLogger
from retroenv.chemistry import canonicalize_components
from retroenv.classes import classify_step
from retroenv.reactions import free_key

from .paroutes import WORKERS, Reaction

RDLogger.DisableLog("rdApp.*")
REAGENT_MIN_COUNT = 20


@dataclass
class Annotated:
    product: str
    reactants: tuple[str, ...]
    free_product: str
    free_reactants: tuple[str, ...]
    reaction_class: str
    template: str | None
    reagents: tuple[str, ...]
    maps_missing: bool
    stereo_lost: bool

    @property
    def clean(self) -> bool:
        return not (self.maps_missing or self.stereo_lost)


def _map_flags(rsmi: str) -> tuple[bool, bool]:
    left, _, right = rsmi.split(">")
    reactants = Chem.MolFromSmiles(left, sanitize=False)
    product = Chem.MolFromSmiles(right, sanitize=False)
    if reactants is None or product is None:
        return True, False
    source = {atom.GetAtomMapNum(): atom for atom in reactants.GetAtoms() if atom.GetAtomMapNum()}
    unspecified = Chem.ChiralType.CHI_UNSPECIFIED
    maps_missing = stereo_lost = False
    for atom in product.GetAtoms():
        origin = source.get(atom.GetAtomMapNum())
        if origin is None:
            maps_missing = True
        elif origin.GetChiralTag() != unspecified and atom.GetChiralTag() == unspecified:
            stereo_lost = True
    return maps_missing, stereo_lost


def _template(rsmi: str) -> str | None:
    reactants, _, product = rsmi.split(">")
    try:
        with redirect_stdout(io.StringIO()):
            result = extract_from_reaction({"_id": 0, "reactants": reactants, "products": product})
    except Exception:
        return None
    return (result or {}).get("reaction_smarts") or None


def _annotate(item: tuple[str, tuple[str, ...], str]) -> Annotated:
    product, reactants, rsmi = item
    free_product, free_reactants = free_key(product, reactants)
    try:
        reagents = canonicalize_components(rsmi.split(">")[1], isomeric=False)
    except Exception:
        reagents = ()
    try:
        maps_missing, stereo_lost = _map_flags(rsmi)
    except Exception:
        maps_missing, stereo_lost = True, False
    return Annotated(
        product=product,
        reactants=reactants,
        free_product=free_product,
        free_reactants=free_reactants,
        reaction_class=classify_step(product, reactants).name,
        template=_template(rsmi) if rsmi else None,
        reagents=reagents,
        maps_missing=maps_missing,
        stereo_lost=stereo_lost,
    )


def annotate(reactions: dict[tuple[str, tuple[str, ...]], Reaction], workers: int = WORKERS) -> dict:
    items = [(r.product, r.reactants, r.rsmi) for r in reactions.values()]
    with Pool(workers) as pool:
        annotated = list(pool.imap(_annotate, items, chunksize=500))
    return {(a.product, a.reactants): a for a in annotated}


def library_rows(reactions: dict, annotated: dict) -> tuple[list[dict], Counter, Counter]:
    """Stereo-free unique corpus rows, template counts, and reagent counts."""
    merged: dict[tuple[str, tuple[str, ...]], dict] = {}
    reagent_counts: Counter = Counter()
    for key, reaction in reactions.items():
        info = annotated[key]
        reagent_counts.update(set(info.reagents))
        free = (info.free_product, info.free_reactants)
        row = merged.get(free)
        if row is None:
            row = merged[free] = {
                "product": info.free_product,
                "reactants": list(info.free_reactants),
                "product_smiles": reaction.product,
                "reactant_smiles": list(reaction.reactants),
                "reaction_class": info.reaction_class,
                "template": info.template,
                "patents": set(),
                "routes": 0,
                "reagents": Counter(),
                "clean": info.clean,
            }
        row["patents"] |= reaction.patents
        row["routes"] += reaction.routes
        row["reagents"].update(info.reagents)
        row["template"] = row["template"] or info.template
        row["clean"] = row["clean"] or info.clean
    template_counts = Counter(row["template"] for row in merged.values() if row["template"])
    rows = []
    for row in sorted(merged.values(), key=lambda r: (r["product"], r["reactants"])):
        row["patents"] = sorted(row["patents"])
        row["reagents"] = [smiles for smiles, _ in row["reagents"].most_common(10)]
        rows.append(row)
    return rows, template_counts, reagent_counts


def reagent_whitelist(reagent_counts: Counter, min_count: int = REAGENT_MIN_COUNT) -> list[str]:
    return sorted(smiles for smiles, count in reagent_counts.items() if count >= min_count)
