"""Deterministic RDKit operations used by normalization and verification."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from typing import Any, Iterable

from rdkit import Chem
from rdkit.Chem import Descriptors, rdChemReactions, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


class ChemistryError(ValueError):
    """An input cannot be represented under the environment's RDKit policy."""


def stable_hash(text: str, *, prefix: str = "", length: int = 24) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


def parse_molecule(smiles: str) -> Chem.Mol:
    text = (smiles or "").strip()
    if not text:
        raise ChemistryError("empty SMILES")
    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        raise ChemistryError(f"cannot parse SMILES: {text!r}")
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:
        raise ChemistryError(f"SMILES does not sanitize: {text!r}: {exc}") from exc
    return mol


def canonicalize_smiles(smiles: str, *, keep_atom_maps: bool = False) -> str:
    mol = parse_molecule(smiles)
    if not keep_atom_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_components(
    value: str | Iterable[str], *, keep_atom_maps: bool = False
) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split(".") if value.strip() else []
    else:
        raw = list(value)
    components = [
        canonicalize_smiles(str(item).strip(), keep_atom_maps=keep_atom_maps)
        for item in raw
        if str(item).strip()
    ]
    return tuple(sorted(components))


def split_reaction_smiles(reaction_smiles: str) -> tuple[str, str, str]:
    parts = (reaction_smiles or "").strip().split(">")
    if len(parts) != 3:
        raise ChemistryError(
            "reaction SMILES must have reactants>reagents>products form"
        )
    if not parts[0].strip() or not parts[2].strip():
        raise ChemistryError("reaction SMILES needs non-empty reactants and products")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def canonicalize_reaction(
    reaction_smiles: str, *, keep_atom_maps: bool = False
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    reactants, reagents, products = split_reaction_smiles(reaction_smiles)
    r = canonicalize_components(reactants, keep_atom_maps=keep_atom_maps)
    a = canonicalize_components(reagents, keep_atom_maps=keep_atom_maps)
    p = canonicalize_components(products, keep_atom_maps=keep_atom_maps)
    return f"{'.'.join(r)}>{'.'.join(a)}>{'.'.join(p)}", r, a, p


def primary_product(products: Iterable[str]) -> str:
    candidates = list(products)
    if not candidates:
        raise ChemistryError("reaction has no products")
    # Stable tie break after preferring the molecule with the largest heavy-atom count.
    return max(candidates, key=lambda value: (parse_molecule(value).GetNumHeavyAtoms(), value))


def audit_atom_mapping(reaction_smiles: str) -> dict[str, Any]:
    """Audit map uniqueness and atom identity without pretending maps are required."""
    reactants, _, products = split_reaction_smiles(reaction_smiles)
    reactant_mols = [parse_molecule(item) for item in reactants.split(".") if item]
    product_mols = [parse_molecule(item) for item in products.split(".") if item]

    def collect(mols: list[Chem.Mol]) -> tuple[dict[int, tuple[int, int]], int, list[int]]:
        mapped: dict[int, tuple[int, int]] = {}
        duplicates: list[int] = []
        total = 0
        for mol in mols:
            for atom in mol.GetAtoms():
                total += 1
                number = atom.GetAtomMapNum()
                if not number:
                    continue
                if number in mapped:
                    duplicates.append(number)
                mapped[number] = (atom.GetAtomicNum(), atom.GetIsotope())
        return mapped, total, sorted(set(duplicates))

    left, left_total, left_duplicates = collect(reactant_mols)
    right, right_total, right_duplicates = collect(product_mols)
    mapped_atoms = len(left) + len(right)
    if mapped_atoms == 0:
        return {
            "status": "unmapped",
            "reactant_coverage": 0.0,
            "product_coverage": 0.0,
            "errors": [],
        }

    errors: list[str] = []
    if left_duplicates:
        errors.append(f"duplicate reactant atom maps: {left_duplicates}")
    if right_duplicates:
        errors.append(f"duplicate product atom maps: {right_duplicates}")
    missing = sorted(set(right) - set(left))
    if missing:
        errors.append(f"product atom maps absent from reactants: {missing}")
    mismatched = sorted(
        number for number in set(left) & set(right) if left[number] != right[number]
    )
    if mismatched:
        errors.append(f"mapped atom element/isotope changed: {mismatched}")

    left_coverage = len(left) / left_total if left_total else 0.0
    right_coverage = len(right) / right_total if right_total else 0.0
    if errors:
        status = "invalid"
    elif left_coverage == 1.0 and right_coverage == 1.0:
        status = "complete"
    else:
        status = "partial"
    return {
        "status": status,
        "reactant_coverage": round(left_coverage, 6),
        "product_coverage": round(right_coverage, 6),
        "errors": errors,
    }


def canonical_step_key(product: str, reactants: Iterable[str]) -> str:
    canonical_product = canonicalize_smiles(product)
    canonical_reactants = canonicalize_components(reactants)
    return f"{canonical_product}<<{'.'.join(canonical_reactants)}"


def inspect_molecule(smiles: str) -> dict[str, Any]:
    mol = parse_molecule(smiles)
    canonical = canonicalize_smiles(smiles)
    chiral_centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return {
        "valid": True,
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "chiral_centres": len(chiral_centres),
        "unassigned_chiral_centres": sum(
            1 for _, assignment in chiral_centres if assignment == "?"
        ),
        "murcko_scaffold": scaffold_smiles(canonical),
    }


def scaffold_smiles(smiles: str) -> str:
    mol = parse_molecule(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        # All acyclic molecules otherwise share the empty scaffold.
        return f"acyclic:{canonicalize_smiles(smiles)}"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def template_produces(
    reaction_smarts: str,
    reactants: Iterable[str],
    expected_product: str,
    *,
    max_permutations: int = 24,
    max_outcomes: int = 256,
) -> bool:
    """Return whether a trusted SMARTS produces the expected canonical product."""
    if not reaction_smarts:
        return False
    try:
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        rdChemReactions.SanitizeRxn(reaction)
    except Exception:
        return False
    required = reaction.GetNumReactantTemplates()
    raw = list(reactants)
    if required != len(raw) or required > 5:
        return False
    try:
        mols = tuple(parse_molecule(item) for item in raw)
        expected = canonicalize_smiles(expected_product)
    except ChemistryError:
        return False

    permutations: Iterable[tuple[Chem.Mol, ...]]
    count = _factorial_bounded(len(mols), max_permutations + 1)
    if count <= max_permutations:
        permutations = itertools.permutations(mols)
    else:
        permutations = (mols,)

    seen = 0
    for ordered in permutations:
        try:
            outcomes = reaction.RunReactants(ordered, maxProducts=max_outcomes)
        except Exception:
            continue
        for outcome in outcomes:
            seen += 1
            if seen > max_outcomes:
                return False
            for product in outcome:
                try:
                    Chem.SanitizeMol(product)
                    value = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                    value = canonicalize_smiles(value)
                except Exception:
                    continue
                if value == expected:
                    return True
    return False


def formula_counter(smiles_values: Iterable[str]) -> Counter[str]:
    """Debug helper; formula equality is not used as a reaction-feasibility gate."""
    counter: Counter[str] = Counter()
    for value in smiles_values:
        for atom in parse_molecule(value).GetAtoms():
            counter[atom.GetSymbol()] += 1
    return counter


def _factorial_bounded(value: int, stop: int) -> int:
    result = 1
    for number in range(2, value + 1):
        result *= number
        if result >= stop:
            return result
    return result

