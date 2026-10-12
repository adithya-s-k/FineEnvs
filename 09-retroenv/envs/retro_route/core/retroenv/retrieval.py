"""Bounded, deterministic retrieval indexes used by the agent tools."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .chemistry import canonicalize_smiles
from .leakage import Key, LeakageRules, molecule_keys, reaction_key, task_keys, task_molecules
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
    product: str
    reactants: tuple[str, ...]
    reaction_class: str
    reagents: tuple[str, ...]
    patents: tuple[str, ...]
    keys: frozenset[Key]


class PrecedentIndex:
    """Search train-visible corpus reactions without exposing held-out answers.

    The builder marks a library reaction visible only when it shares no leakage
    key with any held-out task, so held-out tasks see the whole visible corpus.
    Given a train ``task``, ``search`` also hides that task's own keys and
    near-duplicates, so a policy cannot learn to look up answers that held-out
    evaluation never offers.
    """

    def __init__(self, reactions: Iterable[dict[str, Any]], rules: LeakageRules | None = None):
        self.rules = rules or LeakageRules()
        records = []
        for row in reactions:
            if not row.get("visible", True):
                continue
            product = row["product_smiles"]
            reactants = tuple(row["reactant_smiles"])
            keys = molecule_keys([product], self.rules) | {reaction_key(product, reactants)}
            keys |= {("patent", patent) for patent in row.get("patents", ())}
            records.append(
                Precedent(
                    product=product,
                    reactants=reactants,
                    reaction_class=row.get("reaction_class", "other"),
                    reagents=tuple(row.get("reagents", ())),
                    patents=tuple(row.get("patents", ())),
                    keys=frozenset(keys),
                )
            )
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._finish(tuple(records), [generator.GetFingerprint(Chem.MolFromSmiles(r.product)) for r in records])

    @classmethod
    def from_records(
        cls, records: Iterable[Precedent], fingerprints: Iterable[Any], rules: LeakageRules | None = None
    ) -> "PrecedentIndex":
        """Rebuild from a serving index: the records and fingerprints ``__init__`` would compute."""
        index = cls.__new__(cls)
        index.rules = rules or LeakageRules()
        index._finish(tuple(records), list(fingerprints))
        return index

    def _finish(self, records: tuple[Precedent, ...], fingerprints: list[Any]) -> None:
        if len(records) != len(fingerprints):
            raise ValueError(f"{len(records)} precedents but {len(fingerprints)} fingerprints")
        self.records = records
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = fingerprints
        self._classes = np.array([_normalize_class(r.reaction_class) for r in self.records])
        self._key_index: dict[Key, list[int]] = {}
        for index, record in enumerate(self.records):
            for key in record.keys:
                self._key_index.setdefault(key, []).append(index)
        self._hidden: dict[tuple[str, str], np.ndarray] = {}

    def fingerprints(self) -> list[Any]:
        """Product fingerprints in record order, for writing a serving index."""
        return list(self._fingerprints)

    def hidden(self, task: RetroTask) -> np.ndarray:
        """Boolean mask of records sharing a leakage key with ``task`` or near-duplicating one of its molecules."""
        cached = self._hidden.get((task.task_id, task.split))
        if cached is not None:
            return cached
        mask = np.zeros(len(self.records), dtype=bool)
        if task.split == "train" and self.records:
            for key in task_keys(task, self.rules):
                mask[self._key_index.get(key, [])] = True
            for smiles in task_molecules(task):
                fingerprint = self._generator.GetFingerprint(Chem.MolFromSmiles(smiles))
                similarity = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint, self._fingerprints))
                mask |= similarity >= self.rules.near_duplicate_threshold
        self._hidden[(task.task_id, task.split)] = mask
        return mask

    def search(
        self,
        *,
        task: RetroTask | None = None,
        product_smiles: str | None = None,
        reaction_class: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        canonical_product = canonicalize_smiles(product_smiles) if product_smiles else None
        if not self.records:
            similarity = np.zeros(0)
        elif canonical_product:
            fingerprint = self._generator.GetFingerprint(Chem.MolFromSmiles(canonical_product))
            similarity = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint, self._fingerprints))
        else:
            similarity = np.zeros(len(self.records))
        allowed = ~self.hidden(task) if task is not None else np.ones(len(self.records), dtype=bool)
        normalized_class = _normalize_class(reaction_class)
        if normalized_class:
            allowed &= self._classes == normalized_class
        candidates = np.flatnonzero(allowed)
        order = candidates[np.lexsort((candidates, -similarity[candidates]))]
        results = [
            {
                "product_smiles": self.records[i].product,
                "reactants": list(self.records[i].reactants),
                "reaction_class": self.records[i].reaction_class,
                "similarity": round(float(similarity[i]), 6),
                "reagents": list(self.records[i].reagents[:8]),
                "patents": list(self.records[i].patents[:3]),
            }
            for i in order[:limit]
        ]
        return {
            "query": {"product_smiles": canonical_product, "reaction_class": reaction_class},
            "results": results,
            "returned": len(results),
            "truncated": len(order) > limit,
            "visibility": "train-visible corpus reactions only",
        }


class StockIndex:
    """The only stock surface: exact, InChIKey, class, SMARTS, or similarity."""

    def __init__(self, stock: Iterable[str]):
        smiles = tuple(sorted({canonicalize_smiles(item) for item in stock}))
        molecules = tuple(Chem.MolFromSmiles(item) for item in smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._finish(smiles, tuple(generator.GetFingerprint(molecule) for molecule in molecules), molecules)

    @classmethod
    def from_canonical(cls, smiles: Iterable[str], fingerprints: Iterable[Any]) -> "StockIndex":
        """Rebuild from a serving index: sorted canonical SMILES and their fingerprints.

        Molecules are parsed only when a class or substructure query first needs them;
        exact, InChIKey and similarity search never do, and parsing them all costs about a
        gigabyte for this stock.
        """
        index = cls.__new__(cls)
        index._finish(tuple(smiles), tuple(fingerprints), None)
        if list(index.smiles) != sorted(index.smiles):
            raise ValueError("stock SMILES must be sorted, as StockIndex orders them")
        return index

    def _finish(self, smiles: tuple[str, ...], fingerprints: tuple[Any, ...], molecules: tuple | None) -> None:
        if len(smiles) != len(fingerprints):
            raise ValueError(f"{len(smiles)} stock molecules but {len(fingerprints)} fingerprints")
        self.smiles = smiles
        self._molecules = molecules
        self._molecules_lock = threading.Lock()
        self._known = frozenset(self.smiles)
        self._by_inchikey: dict[str, str] | None = None  # built on the first InChIKey query; InChI is slow
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = fingerprints

    @property
    def known(self) -> frozenset[str]:
        """The stock as one shared set; sessions reuse it instead of rebuilding it per episode."""
        return self._known

    @property
    def molecules(self) -> tuple:
        if self._molecules is None:
            with self._molecules_lock:
                if self._molecules is None:
                    self._molecules = tuple(Chem.MolFromSmiles(item) for item in self.smiles)
        return self._molecules

    def fingerprints(self) -> tuple[Any, ...]:
        """Fingerprints in SMILES order, for writing a serving index."""
        return self._fingerprints

    @staticmethod
    @lru_cache(maxsize=100_000)
    def inchikey(smiles: str) -> str:
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))

    def retrieve(
        self, query: str, *, mode: str = "auto", limit: int = 10, excluded: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """``excluded`` molecules are treated as absent from the stock (restricted-stock tasks)."""
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

        matches: list[tuple[float, str]] = []
        try:
            if mode == "exact":
                canonical = canonicalize_smiles(query)
                if canonical in self._known:
                    matches = [(1.0, canonical)]
            elif mode == "inchikey":
                if self._by_inchikey is None:
                    self._by_inchikey = {self.inchikey(smiles): smiles for smiles in self.smiles}
                smiles = self._by_inchikey.get(query.upper())
                if smiles:
                    matches = [(1.0, smiles)]
            elif mode in {"class", "substructure"}:
                smarts = CLASS_SMARTS.get(query.lower().replace(" ", "_")) if mode == "class" else query
                if not smarts:
                    raise ValueError(f"unknown class {query!r}; available: {sorted(CLASS_SMARTS)}")
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    raise ValueError("invalid SMARTS query")
                matches = [
                    (1.0, s) for s, molecule in zip(self.smiles, self.molecules) if molecule.HasSubstructMatch(pattern)
                ]
            elif mode == "similarity":
                query_fp = self._generator.GetFingerprint(Chem.MolFromSmiles(canonicalize_smiles(query)))
                values = DataStructs.BulkTanimotoSimilarity(query_fp, self._fingerprints)
                matches = sorted(zip(values, self.smiles), key=lambda item: (-item[0], item[1]))
            else:
                raise ValueError("mode must be auto, exact, inchikey, class, substructure, or similarity")
        except Exception as exc:
            return {"mode": mode, "query": query, "results": [], "error": str(exc)}
        if excluded:
            matches = [match for match in matches if match[1] not in excluded]
        rendered = [
            {"smiles": smiles, "inchikey": self.inchikey(smiles), "similarity": round(score, 6)}
            for score, smiles in matches[:limit]
        ]
        return {
            "mode": mode,
            "query": query,
            "results": rendered,
            "returned": len(rendered),
            "truncated": len(matches) > limit,
        }


_PRELOADED_STOCKS: dict[frozenset[str], StockIndex] = {}


def register_stock_index(index: StockIndex) -> frozenset[str]:
    """Make a prebuilt index (from a serving index) the one ``cached_stock_index`` returns for its stock."""
    _PRELOADED_STOCKS[index.known] = index
    return index.known


@lru_cache(maxsize=8)
def cached_stock_index(stock: frozenset[str]) -> StockIndex:
    """Build fingerprints once for an immutable serving stock snapshot."""
    return _PRELOADED_STOCKS.get(stock) or StockIndex(stock)


def molecule_lookup(query: str, cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
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
