"""Synthetic submissions that a reward must order correctly.

Two groups. The *staircase* truncates a known route at every depth: a reward
with training signal must rank these strictly increasing, so a policy always
gains by pushing one level further toward stock. The *hacks* are degenerate
submissions that cost nothing to produce; none may out-score a real solve.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Iterable

from retroenv.graph import GRAPH_SCHEMA_VERSION
from retroenv.models import RetroTask
from retroenv.verifier import _canonical_stock, known_routes_submission

JUNK_LEAF = "C1CC2CCC1C2"  # a valid molecule that no release stock contains


def tree_depth(node: dict[str, Any]) -> int:
    reaction = (node.get("children") or [None])[0]
    if not reaction:
        return 0
    return 1 + max((tree_depth(child) for child in reaction["children"]), default=0)


def truncate(node: dict[str, Any], levels: int, stock: frozenset[str]) -> dict[str, Any]:
    """Keep the top `levels` reactions; everything below becomes an honest leaf."""
    node = copy.deepcopy(node)
    if levels <= 0 or not node.get("children"):
        return {"type": "mol", "smiles": node["smiles"], "in_stock": node["smiles"] in stock, "children": []}
    reaction = node["children"][0]
    reaction["children"] = [truncate(child, levels - 1, stock) for child in reaction["children"]]
    return node


def _leaves(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if not node.get("children"):
        yield node
        return
    for child in node["children"][0]["children"]:
        yield from _leaves(child)


def _reaction(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "reaction",
        "is_reaction": True,
        "metadata": {
            "explanation": "probe",
            "reaction_class": "unknown",
            "confidence": 0.9,
            "literature": [],
            "precursor_roles": {child["smiles"]: "precursor" for child in children},
        },
        "children": children,
    }


def _mol(smiles: str, in_stock: bool) -> dict[str, Any]:
    return {"type": "mol", "smiles": smiles, "in_stock": in_stock, "children": []}


def build(task: RetroTask, stock: Iterable[str], *, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Named probe submissions for one task; empty when the task has no usable known route."""
    effective = _canonical_stock(stock) - _canonical_stock(task.constraints.excluded_stock)
    oracle = known_routes_submission(task, stock)
    if not oracle["routes"]:
        return {}
    first = oracle["routes"][0]
    one = {"schema_version": GRAPH_SCHEMA_VERSION, "routes": [first]}
    rng = random.Random(seed)
    buyable = rng.sample(sorted(effective), 2) if len(effective) >= 2 else sorted(effective) * 2

    probes: dict[str, dict[str, Any]] = {
        "solve-oracle": oracle,
        "solve-one-route": one,
        "hack-empty": {"routes": []},
        # Padded solves really do solve the task; they must simply never pay as well
        # as the clean submission, so they are scored apart from the true hacks.
        "pad-duplicate-x3": {"routes": [copy.deepcopy(first) for _ in range(3)]},
        # All leaves really are in stock, but no step is a reaction the library knows.
        "hack-stock-stub": {
            "routes": [
                {
                    "type": "mol",
                    "smiles": first["smiles"],
                    "in_stock": False,
                    "children": [_reaction([_mol(smiles, True) for smiles in buyable])],
                }
            ]
        },
        # A solve diluted with cheap junk, up to the route budget.
        "pad-junk": {
            "routes": [first]
            + [
                {
                    "type": "mol",
                    "smiles": first["smiles"],
                    "in_stock": False,
                    "children": [_reaction([_mol(JUNK_LEAF, False), _mol(buyable[i % 2], True)])],
                }
                for i in range(task.max_routes - 1)
            ]
        },
    }

    lying = copy.deepcopy(one)
    for leaf in _leaves(lying["routes"][0]):
        leaf["smiles"], leaf["in_stock"] = JUNK_LEAF, True
        break
    probes["hack-false-stock"] = lying

    depth = tree_depth(first)
    for level in range(1, depth):
        probes[f"stair-{level}of{depth}"] = {"routes": [truncate(first, level, effective)]}
    # The top of the ladder is the full route, so it counts as a solve, not a hack.
    probes[f"solve-stair-{depth}of{depth}"] = one
    return probes
