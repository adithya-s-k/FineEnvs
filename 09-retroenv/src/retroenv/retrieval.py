"""Bounded, deterministic retrieval indexes used by the agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .chemistry import canonical_step_key, canonicalize_smiles
from .models import RetroTask


CLASS_SMARTS = {
    "alcohol": "[OX2H][CX4]",
    "amine": "[NX3;H1,H2;!$(NC=O)]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aryl_halide": "[c][F,Cl,Br,I]",
    "boronic_acid": "B(O)O",
    "ester": "C(=O)O[#6]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
}


@dataclass(frozen=True)
class Precedent:
    task_id: str
    product: str
    reactants: tuple[str, ...]
    reaction_class: str | None
    reaction_id: str | None
    conditions: tuple[dict[str, Any], ...]
    literature: tuple[dict[str, Any], ...]
    source: tuple[dict[str, Any], ...]


class PrecedentIndex:
    """Search training-visible reaction steps without exposing eval references."""

    def __init__(self, tasks: Iterable[RetroTask]):
        records: list[Precedent] = []
        seen: set[str] = set()
        for task in tasks:
            for route in task.reference_routes:
                for step in route.steps:
                    key = canonical_step_key(step.product, step.reactants)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        Precedent(
                            task_id=task.task_id,
                            product=canonicalize_smiles(step.product),
                            reactants=tuple(step.reactants),
                            reaction_class=step.reaction_class,
                            reaction_id=step.reaction_id,
                            conditions=step.conditions,
                            literature=step.literature,
                            source=route.source,
                        )
                    )
        self.records = tuple(records)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = tuple(
            generator.GetFingerprint(Chem.MolFromSmiles(record.product))
            for record in self.records
        )
        self._generator = generator

    def search(
        self,
        *,
        task_id: str,
        product_smiles: str | None = None,
        reaction_class: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        query_fp = None
        canonical_product = None
        if product_smiles:
            canonical_product = canonicalize_smiles(product_smiles)
            query_fp = self._generator.GetFingerprint(Chem.MolFromSmiles(canonical_product))
        normalized_class = _normalize_class(reaction_class)
        ranked: list[tuple[float, str, Precedent]] = []
        for record, fingerprint in zip(self.records, self._fingerprints):
            if record.task_id == task_id:
                continue
            if normalized_class and _normalize_class(record.reaction_class) != normalized_class:
                continue
            similarity = DataStructs.TanimotoSimilarity(query_fp, fingerprint) if query_fp else 0.0
            ranked.append((similarity, record.reaction_id or "", record))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2].product))
        results = []
        for similarity, _, record in ranked[:limit]:
            results.append(
                {
                    "product_smiles": record.product,
                    "reactants": list(record.reactants),
                    "reaction_class": record.reaction_class,
                    "similarity": round(similarity, 6),
                    "conditions": list(record.conditions)[:3],
                    "literature": list(record.literature)[:3],
                    "source": [
                        {
                            key: value
                            for key, value in source.items()
                            if key in {"name", "url", "publication_year"}
                        }
                        for source in record.source[:3]
                    ],
                }
            )
        return {
            "query": {
                "product_smiles": canonical_product,
                "reaction_class": reaction_class,
            },
            "results": results,
            "returned": len(results),
            "truncated": len(ranked) > limit,
            "visibility": "training-split precedents only",
        }


class StockIndex:
    """The only stock surface: exact, InChIKey, class, SMARTS, or similarity."""

    def __init__(self, stock: Iterable[str]):
        self.smiles = tuple(sorted({canonicalize_smiles(item) for item in stock}))
        self.molecules = tuple(Chem.MolFromSmiles(item) for item in self.smiles)
        self.inchikeys = tuple(Chem.MolToInchiKey(molecule) for molecule in self.molecules)
        self._by_inchikey = dict(zip(self.inchikeys, self.smiles))
        self._by_smiles = dict(zip(self.smiles, self.inchikeys))
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = tuple(
            self._generator.GetFingerprint(molecule) for molecule in self.molecules
        )

    def retrieve(self, query: str, *, mode: str = "auto", limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        query = str(query or "").strip()
        if not query:
            return {"mode": mode, "query": query, "results": [], "error": "query is empty"}
        if mode == "auto" and ":" in query:
            prefix, value = query.split(":", 1)
            if prefix in {"exact", "inchikey", "class", "substructure", "similarity"}:
                mode, query = prefix, value.strip()
        if mode == "auto":
            mode = "inchikey" if len(query) == 27 and query.count("-") == 2 else "exact"

        matches: list[tuple[float, str, str]] = []
        try:
            if mode == "exact":
                canonical = canonicalize_smiles(query)
                inchikey = self._by_smiles.get(canonical)
                if inchikey:
                    matches = [(1.0, canonical, inchikey)]
            elif mode == "inchikey":
                smiles = self._by_inchikey.get(query.upper())
                if smiles:
                    matches = [(1.0, smiles, query.upper())]
            elif mode in {"class", "substructure"}:
                smarts = CLASS_SMARTS.get(query.lower().replace(" ", "_")) if mode == "class" else query
                if not smarts:
                    raise ValueError(
                        f"unknown class {query!r}; available: {sorted(CLASS_SMARTS)}"
                    )
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    raise ValueError("invalid SMARTS query")
                matches = [
                    (1.0, smiles, inchikey)
                    for smiles, inchikey, molecule in zip(
                        self.smiles, self.inchikeys, self.molecules
                    )
                    if molecule.HasSubstructMatch(pattern)
                ]
            elif mode == "similarity":
                molecule = Chem.MolFromSmiles(canonicalize_smiles(query))
                query_fp = self._generator.GetFingerprint(molecule)
                matches = [
                    (DataStructs.TanimotoSimilarity(query_fp, fingerprint), smiles, inchikey)
                    for smiles, inchikey, fingerprint in zip(
                        self.smiles, self.inchikeys, self._fingerprints
                    )
                ]
                matches.sort(key=lambda item: (-item[0], item[1]))
            else:
                raise ValueError("mode must be auto, exact, inchikey, class, substructure, or similarity")
        except Exception as exc:
            return {"mode": mode, "query": query, "results": [], "error": str(exc)}
        rendered = [
            {"smiles": smiles, "inchikey": inchikey, "similarity": round(score, 6)}
            for score, smiles, inchikey in matches[:limit]
        ]
        return {
            "mode": mode,
            "query": query,
            "results": rendered,
            "returned": len(rendered),
            "truncated": len(matches) > limit,
        }


@lru_cache(maxsize=8)
def cached_stock_index(stock: frozenset[str]) -> StockIndex:
    """Build fingerprints once for an immutable serving stock snapshot."""
    return StockIndex(stock)


def molecule_lookup(
    query: str, cache: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Resolve a SMILES locally; names/CAS require the pre-hydrated PubChem cache."""

    normalized_query = str(query or "").strip()
    cached = (cache or {}).get(normalized_query.casefold())
    if cached:
        try:
            smiles = canonicalize_smiles(str(cached["canonical_smiles"]))
        except Exception as exc:
            return {"found": False, "query": query, "error": f"bad cache row: {exc}"}
        return {
            "found": True,
            "query": query,
            **cached,
            "canonical_smiles": smiles,
            "source": "frozen_pubchem_cache",
        }
    try:
        smiles = canonicalize_smiles(normalized_query)
    except Exception:
        return {
            "found": False,
            "query": query,
            "error": "name/CAS not in the local PubChem cache; prefetch it before rollouts",
        }
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "found": True,
        "query": query,
        "canonical_smiles": smiles,
        "inchikey": Chem.MolToInchiKey(molecule),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "molecular_weight": round(Descriptors.MolWt(molecule), 6),
        "source": "local_rdkit_canonicalization",
    }


def _normalize_class(value: str | None) -> str:
    return "".join(character for character in (value or "").lower() if character.isalnum())
