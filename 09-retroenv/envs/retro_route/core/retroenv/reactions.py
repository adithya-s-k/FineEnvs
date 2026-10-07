"""Reference-free single-step support: a frozen reaction corpus plus rdchiral retro-templates.

A step ``product <- reactants`` is supported when it is a known corpus reaction
(stereo-free canonical key, optionally padded with common reagents) or when a
radius-1 retro-template seen at least ``min_count`` times in the corpus turns the
product into a subset of the proposed reactants whose extras are common reagents.
Comparison is stereo-free; whether stereo also agrees is reported separately.
"""

from __future__ import annotations

import gzip
import io
import itertools
import json
import threading
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from rdchiral.main import rdchiralReactants, rdchiralReaction, rdchiralRun
from rdkit import Chem, DataStructs, RDLogger

from .chemistry import ChemistryError, canonicalize_components, canonicalize_smiles

RDLogger.DisableLog("rdApp.*")

DEFAULT_MIN_COUNT = 5
FP_SIZE = 2048
CACHE_SIZE = 50_000

StepKey = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class StepSupport:
    """``kind`` is ``corpus``, ``template``, ``none`` or ``invalid``."""

    kind: str
    product: str
    reactants: tuple[str, ...]
    template_count: int = 0
    stereo_match: bool | None = None
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.kind in {"corpus", "template"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def free_key(product: str, reactants: Iterable[str]) -> StepKey:
    """Stereo-free canonical (product, sorted reactant components) key."""
    return canonicalize_smiles(product, isomeric=False), canonicalize_components(reactants, isomeric=False)


def _query(smarts: str) -> Chem.Mol:
    mol = Chem.MolFromSmarts(smarts)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _pattern_fp(mol: Chem.Mol) -> np.ndarray:
    bits = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(Chem.PatternFingerprint(mol, fpSize=FP_SIZE), bits)
    return np.packbits(bits).view(np.uint64)


def run_template(reaction: rdchiralReaction, product: rdchiralReactants) -> list[str]:
    try:
        with redirect_stdout(io.StringIO()):
            return rdchiralRun(reaction, product)
    except Exception:
        return []


class TemplateIndex:
    """Retro-templates screened by product-side pattern fingerprints, then substructure."""

    def __init__(self, template_counts: dict[str, int]):
        ordered = sorted(template_counts.items(), key=lambda item: (-item[1], item[0]))
        self.smarts = [smarts for smarts, _ in ordered]
        self.counts = np.array([count for _, count in ordered], dtype=np.int64)
        product_sides = [smarts.split(">>")[0] for smarts in self.smarts]
        unique_sides = sorted(set(product_sides))
        side_id = {side: i for i, side in enumerate(unique_sides)}
        self._pattern_of = np.array([side_id[side] for side in product_sides], dtype=np.int64)
        self._patterns = [_query(side) for side in unique_sides]
        # Pattern-fingerprint screening is only safe for product-side queries.
        self._pattern_fps = (
            np.stack([_pattern_fp(pattern) for pattern in self._patterns])
            if self._patterns
            else np.zeros((0, FP_SIZE // 64), dtype=np.uint64)
        )
        fragment_lists = [smarts.split(">>")[1].split(".") for smarts in self.smarts]
        unique_fragments = sorted({fragment for fragments in fragment_lists for fragment in fragments})
        fragment_id = {fragment: i for i, fragment in enumerate(unique_fragments)}
        self._fragments = [_query(fragment) for fragment in unique_fragments]
        self._template_fragments = [tuple(fragment_id[f] for f in fragments) for fragments in fragment_lists]
        self._reactions: dict[int, rdchiralReaction] = {}

    def __len__(self) -> int:
        return len(self.smarts)

    def _reaction(self, template_id: int) -> rdchiralReaction:
        if template_id not in self._reactions:
            self._reactions[template_id] = rdchiralReaction(self.smarts[template_id])
        return self._reactions[template_id]

    def matching(self, product: Chem.Mol) -> np.ndarray:
        if not len(self._patterns):
            return np.zeros(0, dtype=np.int64)
        fp = _pattern_fp(product)
        screened = np.flatnonzero(~np.any(self._pattern_fps & ~fp, axis=1))
        matched = np.zeros(len(self._patterns), dtype=bool)
        matched[[p for p in screened if product.HasSubstructMatch(self._patterns[p])]] = True
        return np.flatnonzero(matched[self._pattern_of])

    def proposals(self, product_smiles: str, limit: int = 50) -> list[tuple[tuple[str, ...], int]]:
        """Stereo-free reactant sets the templates propose for ``product``, most frequent template first."""
        product = rdchiralReactants(canonicalize_smiles(product_smiles))
        seen: dict[tuple[str, ...], int] = {}
        for template_id in self.matching(product.reactants_achiral):
            for outcome in run_template(self._reaction(int(template_id)), product):
                try:
                    reactants = canonicalize_components(outcome, isomeric=False)
                except ChemistryError:
                    continue
                seen.setdefault(reactants, int(self.counts[template_id]))
            if len(seen) >= limit:
                break
        return list(seen.items())[:limit]

    def first_support(
        self, product_smiles: str, reactants: tuple[str, ...], extras: frozenset[str]
    ) -> tuple[int, str] | None:
        product = rdchiralReactants(canonicalize_smiles(product_smiles))
        reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactants]
        proposed = set(reactants)
        present: dict[int, bool] = {}
        for template_id in self.matching(product.reactants_achiral):
            if not self._fragments_present(int(template_id), reactant_mols, present):
                continue
            for outcome in run_template(self._reaction(int(template_id)), product):
                try:
                    produced = set(canonicalize_components(outcome, isomeric=False))
                except ChemistryError:
                    continue
                if produced and produced <= proposed and proposed - produced <= extras:
                    return int(template_id), outcome
        return None

    def _fragments_present(self, template_id: int, reactant_mols: list[Chem.Mol], present: dict[int, bool]) -> bool:
        for fragment in self._template_fragments[template_id]:
            if fragment not in present:
                pattern = self._fragments[fragment]
                present[fragment] = any(mol is not None and mol.HasSubstructMatch(pattern) for mol in reactant_mols)
            if not present[fragment]:
                return False
        return True


class ReactionLibrary:
    """Judge arbitrary single steps without consulting any task's hidden route."""

    def __init__(
        self,
        corpus: Iterable[StepKey],
        templates: dict[str, int],
        reagents: Iterable[str],
        *,
        min_count: int = DEFAULT_MIN_COUNT,
    ):
        self.min_count = min_count
        self.corpus = frozenset(corpus)
        self.reagents = frozenset(reagents)
        self.index = TemplateIndex({smarts: n for smarts, n in templates.items() if n >= min_count})
        self._cache: OrderedDict[tuple[StepKey, frozenset[StepKey]], StepSupport] = OrderedDict()
        self._lock = threading.Lock()  # servers call tools from worker threads

    def support(
        self,
        product: str,
        reactants: Iterable[str],
        *,
        exclude: frozenset[StepKey] = frozenset(),
    ) -> StepSupport:
        """Support for ``product <- reactants``; ``exclude`` hides corpus keys (never templates)."""
        reactant_list = [str(item) for item in reactants]
        try:
            key = free_key(product, reactant_list)
            stereo_reactants = canonicalize_components(reactant_list)
        except ChemistryError as exc:
            return StepSupport("invalid", "", (), error=str(exc))
        product_key, reactant_key = key
        if not reactant_key:
            return StepSupport("invalid", product_key, (), error="a step needs at least one reactant")
        if product_key in reactant_key:
            return StepSupport("invalid", product_key, reactant_key, error="product appears unchanged among reactants")
        cache_key = (key, exclude)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached
        result = self._support(product, key, stereo_reactants, exclude)
        with self._lock:
            self._cache[cache_key] = result
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    def _support(
        self, product: str, key: StepKey, stereo_reactants: tuple[str, ...], exclude: frozenset[StepKey]
    ) -> StepSupport:
        product_key, reactant_key = key
        if self._in_corpus(key, exclude):
            return StepSupport("corpus", product_key, reactant_key)
        extras = frozenset(smiles for smiles in reactant_key if smiles in self.reagents)
        try:
            found = self.index.first_support(product, reactant_key, extras)
        except Exception as exc:  # rdchiral can fail on exotic valences
            return StepSupport("none", product_key, reactant_key, error=f"template engine: {exc}")
        if found is None:
            return StepSupport("none", product_key, reactant_key)
        template_id, outcome = found
        try:
            stereo_match = set(canonicalize_components(outcome)) <= set(stereo_reactants)
        except ChemistryError:
            stereo_match = None
        return StepSupport(
            "template", product_key, reactant_key, int(self.index.counts[template_id]), stereo_match
        )

    def _in_corpus(self, key: StepKey, exclude: frozenset[StepKey]) -> bool:
        product, reactants = key
        padding = [smiles for smiles in reactants if smiles in self.reagents]
        for size in range(len(padding) + 1):
            for dropped in itertools.combinations(padding, size):
                core = tuple(smiles for smiles in reactants if smiles not in dropped)
                if core and (product, core) in self.corpus and (product, core) not in exclude:
                    return True
        return False

    @classmethod
    def load(cls, directory: str | Path, *, templates: str = "templates.json", visible_only: bool = False,
             min_count: int = DEFAULT_MIN_COUNT) -> "ReactionLibrary":
        root = Path(directory)
        corpus = (
            (record["product"], tuple(record["reactants"]))
            for record in iter_reactions(root / "reactions.jsonl.gz")
            if record["visible"] or not visible_only
        )
        counts = json.loads((root / templates).read_text())
        reagents = json.loads((root / "reagents.json").read_text())
        return cls(corpus, counts, reagents, min_count=min_count)


def iter_reactions(path: str | Path) -> Iterator[dict[str, Any]]:
    """Rows of the library's reaction table (stereo-free keys plus provenance)."""
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
