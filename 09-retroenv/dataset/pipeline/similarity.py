"""Morgan (radius 2, 2048 bit) Tanimoto on packed numpy fingerprints, parallel and copy-on-write safe.

Forked workers read one shared uint64 array instead of touching the reference
counts of millions of RDKit bit-vector objects, which would copy every page.
Bits are identical to ``rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)``.
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .paroutes import WORKERS

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_A: np.ndarray = np.zeros((0, 32), dtype=np.uint64)
_B: np.ndarray = np.zeros((0, 32), dtype=np.uint64)
_B_COUNTS: np.ndarray = np.zeros(0)


def _packed(smiles: str) -> np.ndarray:
    bits = _GENERATOR.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.uint8)
    return np.packbits(bits).view(np.uint64)


def packed(smiles: list[str], workers: int = WORKERS) -> np.ndarray:
    with Pool(workers) as pool:
        rows = pool.map(_packed, smiles, chunksize=2000)
    return np.stack(rows) if rows else np.zeros((0, 32), dtype=np.uint64)


def _similarity(query: np.ndarray) -> np.ndarray:
    inter = np.bitwise_count(_B & query).sum(axis=1)
    union = _B_COUNTS + np.bitwise_count(query).sum() - inter
    return inter / np.maximum(union, 1)


def _top(bounds: tuple[int, int, int, bool]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    start, stop, k, same = bounds
    out = []
    for i in range(start, stop):
        similarity = _similarity(_A[i])
        if same:
            similarity[i] = -1.0
        top = np.argpartition(-similarity, min(k, len(similarity) - 1))[:k]
        top = top[np.argsort(-similarity[top], kind="stable")]
        out.append((i, top, similarity[top]))
    return out


def _above(bounds: tuple[int, int, float]) -> set[int]:
    start, stop, threshold = bounds
    hits: set[int] = set()
    for i in range(start, stop):
        hits.update(np.flatnonzero(_similarity(_A[i]) >= threshold).tolist())
    return hits


def _chunks(n: int, workers: int):
    size = max(1, n // (workers * 4))
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def _share(queries: np.ndarray, targets: np.ndarray) -> None:
    global _A, _B, _B_COUNTS
    _A, _B = queries, targets
    _B_COUNTS = np.bitwise_count(targets).sum(axis=1)


def nearest(fps: np.ndarray, k: int, workers: int = WORKERS):
    """For every row, its k most similar other rows: yields (row, indexes, similarities)."""
    _share(fps, fps)
    with Pool(workers) as pool:
        for block in pool.imap_unordered(_top, [(a, b, k, True) for a, b in _chunks(len(fps), workers)]):
            yield from block


def above(queries: np.ndarray, targets: np.ndarray, threshold: float, workers: int = WORKERS) -> set[int]:
    """Indexes of ``targets`` within ``threshold`` of any query."""
    if not len(queries) or not len(targets):
        return set()
    _share(queries, targets)
    found: set[int] = set()
    with Pool(workers) as pool:
        for block in pool.imap_unordered(_above, [(a, b, threshold) for a, b in _chunks(len(queries), workers)]):
            found |= block
    return found
