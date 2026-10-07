"""Stage 3: shortest stock-closed routes through the clean reaction graph.

``depth[m]`` is the smallest longest-linear-sequence of any route that makes
``m`` from stock using clean corpus reactions recombined across patents. Stock
molecules have depth 0, so a target whose intermediate is purchasable gets its
true (shorter) depth. Witness routes are rebuilt by following the chosen
reaction at each molecule; depths strictly decrease, so they never cycle.
Constrained solves (no stock molecule X, no reaction class C) rerun the same
relaxation on a target's backward subgraph only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

MAX_DEPTH = 8

Step = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class GraphReaction:
    product: str
    reactants: tuple[str, ...]
    reaction_class: str
    popularity: int
    patents: tuple[str, ...]


class RouteGraph:
    def __init__(self, reactions: list[GraphReaction], stock: frozenset[str]):
        self.reactions = sorted(reactions, key=lambda r: (r.product, r.reactants))
        self.stock = stock
        self.producers: dict[str, list[int]] = defaultdict(list)
        for index, reaction in enumerate(self.reactions):
            if reaction.product not in reaction.reactants:
                self.producers[reaction.product].append(index)
        self.depth, self.choice = self._relax(range(len(self.reactions)), stock, frozenset())

    def _relax(
        self, indexes, stock: frozenset[str], forbidden: frozenset[str], max_depth: int = MAX_DEPTH
    ) -> tuple[dict[str, int], dict[str, int]]:
        depth: dict[str, int] = dict.fromkeys(stock, 0)
        choice: dict[str, int] = {}
        candidates = [
            i
            for i in indexes
            if self.reactions[i].product not in stock and self.reactions[i].reaction_class not in forbidden
        ]
        for _ in range(max_depth):
            changed = False
            for index in candidates:
                reaction = self.reactions[index]
                inner = [depth.get(r) for r in reaction.reactants]
                if None in inner:
                    continue
                value = 1 + max(inner)
                if value > max_depth:
                    continue
                current = depth.get(reaction.product)
                if current is None or value < current or (
                    value == current and self._better(index, choice.get(reaction.product))
                ):
                    if current != value or choice.get(reaction.product) != index:
                        changed = True
                    depth[reaction.product] = value
                    choice[reaction.product] = index
            if not changed:
                break
        return depth, choice

    def _better(self, index: int, incumbent: int | None) -> bool:
        if incumbent is None:
            return True
        a, b = self.reactions[index], self.reactions[incumbent]
        return (-a.popularity, a.reactants) < (-b.popularity, b.reactants)

    def witness(self, target: str, depth: dict[str, int] | None = None, choice: dict[str, int] | None = None,
                stock: frozenset[str] | None = None, first: int | None = None) -> list[Step]:
        """Steps of the chosen route for ``target`` (optionally forcing its first reaction)."""
        depth = self.depth if depth is None else depth
        choice = self.choice if choice is None else choice
        stock = self.stock if stock is None else stock
        steps: list[Step] = []
        seen: set[str] = set()

        def build(smiles: str, forced: int | None = None) -> None:
            if smiles in seen or (forced is None and smiles in stock):
                return
            index = forced if forced is not None else choice[smiles]
            reaction = self.reactions[index]
            seen.add(smiles)
            steps.append((reaction.product, reaction.reactants))
            for reactant in reaction.reactants:
                build(reactant)

        build(target, first)
        return steps

    def subgraph(self, target: str, max_depth: int = MAX_DEPTH, cap: int = 50_000) -> list[int]:
        """Reaction indexes reachable backwards from ``target`` within ``max_depth`` levels."""
        found: set[int] = set()
        frontier = {target}
        visited = {target}
        for _ in range(max_depth):
            following: set[str] = set()
            for smiles in frontier:
                for index in self.producers.get(smiles, ()):
                    if index in found:
                        continue
                    found.add(index)
                    for reactant in self.reactions[index].reactants:
                        if reactant not in visited and reactant not in self.stock:
                            visited.add(reactant)
                            following.add(reactant)
            if len(found) > cap or not following:
                break
            frontier = following
        return sorted(found)

    def solve(self, target: str, *, excluded: frozenset[str] = frozenset(), forbidden: frozenset[str] = frozenset(),
              indexes: list[int] | None = None) -> tuple[int, list[Step]] | None:
        """Shortest route for ``target`` without the excluded stock and forbidden classes."""
        indexes = self.subgraph(target) if indexes is None else indexes
        stock = self.stock - excluded
        depth, choice = self._relax(indexes, stock, forbidden)
        if target not in depth or target in stock:
            return None
        return depth[target], self.witness(target, depth, choice, stock)

    def first_cuts(self, target: str, max_depth: int) -> list[tuple[int, int]]:
        """(depth, reaction index) of every solvable first reaction within ``max_depth``, best first."""
        cuts = []
        for index in self.producers.get(target, ()):
            reaction = self.reactions[index]
            inner = [self.depth.get(r) for r in reaction.reactants]
            if None in inner:
                continue
            value = 1 + max(inner)
            if value <= max_depth:
                cuts.append((value, index))
        return sorted(cuts, key=lambda item: (item[0], -self.reactions[item[1]].popularity, self.reactions[item[1]].reactants))


def steps_depth(target: str, steps: list[Step]) -> int:
    by_product = dict(steps)

    def depth(smiles: str, seen: frozenset[str]) -> int:
        if smiles not in by_product or smiles in seen:
            return 0
        return 1 + max(depth(r, seen | {smiles}) for r in by_product[smiles])

    return depth(target, frozenset())


def steps_leaves(target: str, steps: list[Step]) -> set[str]:
    by_product = dict(steps)
    leaves: set[str] = set()

    def walk(smiles: str, seen: frozenset[str]) -> None:
        if smiles not in by_product or smiles in seen:
            leaves.add(smiles)
            return
        for reactant in by_product[smiles]:
            walk(reactant, seen | {smiles})

    walk(target, frozenset())
    return leaves


def is_convergent(steps: list[Step], stock: frozenset[str]) -> bool:
    products = {product for product, _ in steps}
    return any(sum(r in products and r not in stock for r in reactants) >= 2 for _, reactants in steps)
