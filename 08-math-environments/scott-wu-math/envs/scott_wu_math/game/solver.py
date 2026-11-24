"""Find one way to make the target from all the cards.

Every expression tree splits the cards at its root into two groups, and the smaller group has at
most half the cards. So the search enumerates every value a small group (up to four cards) can
make, and asks the rest of the cards for the one number that would complete the target:
``target = v + rest`` needs ``rest = target - v``, and so on for the six ways to put v on either
side of the four operators. Recursing until four cards remain, where the answer is a lookup, makes
the search complete for up to nine cards, and in practice it finds an answer in milliseconds
because big targets have many.

A first pass keeps every intermediate value a whole number (the way people play, as in the
video); a second pass allows fractions. Both stop at a deadline, which is checked at every step of
the search and while building the values of any group bigger than the cached ones.
"""

from __future__ import annotations

import time
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from .expr import Tree, evaluate, leaves

LOOKUP_SIZE = 4
CACHE_SIZE = 8_192  # every group of up to LOOKUP_SIZE values 1-13, for both passes, is 4,758 entries


class Timeout(Exception):
    pass


def _reach(values: tuple[int, ...], integers: bool, deadline: float | None = None) -> dict[Fraction, Tree]:
    """Every value a sorted group of cards can make, with one tree for each.

    Groups of up to LOOKUP_SIZE cards are cached: with values 1-13 there are 2,379 of them, so the
    cache holds every one for both passes. A bigger group, which only a hand of more than nine cards
    needs, is built fresh and stops at the deadline.
    """
    if len(values) <= LOOKUP_SIZE:
        return _reach_small(values, integers)
    return _combine(values, integers, deadline)


@lru_cache(maxsize=CACHE_SIZE)
def _reach_small(values: tuple[int, ...], integers: bool) -> dict[Fraction, Tree]:
    return _combine(values, integers, None)


def _combine(values: tuple[int, ...], integers: bool, deadline: float | None) -> dict[Fraction, Tree]:
    if len(values) == 1:
        return {Fraction(values[0]): values[0]}
    out: dict[Fraction, Tree] = {}
    n = len(values)
    seen: set[tuple[int, ...]] = set()
    for k in range(1, n // 2 + 1):
        for idx in combinations(range(n), k):
            left = tuple(values[i] for i in idx)
            if left in seen:
                continue
            seen.add(left)
            right = tuple(v for i, v in enumerate(values) if i not in idx)
            if k * 2 == n and right in seen and right != left:
                continue
            lr, rr = _reach(left, integers, deadline), _reach(right, integers, deadline)
            for a, ta in lr.items():
                if deadline is not None and time.monotonic() > deadline:
                    raise Timeout
                for b, tb in rr.items():
                    _add(out, a + b, ("+", ta, tb), integers)
                    _add(out, a * b, ("*", ta, tb), integers)
                    _add(out, a - b, ("-", ta, tb), integers)
                    _add(out, b - a, ("-", tb, ta), integers)
                    if b:
                        _add(out, a / b, ("/", ta, tb), integers)
                    if a:
                        _add(out, b / a, ("/", tb, ta), integers)
    return out


def _add(out: dict, value: Fraction, tree: Tree, integers: bool) -> None:
    if integers and (value.denominator != 1 or value < 0):
        return
    out.setdefault(value, tree)


class _Search:
    def __init__(self, integers: bool, deadline: float):
        self.integers = integers
        self.deadline = deadline
        self.failed: set[tuple[Fraction, tuple[int, ...]]] = set()

    def solve(self, target: Fraction, values: tuple[int, ...]) -> Tree | None:
        if len(values) <= LOOKUP_SIZE:
            return _reach(values, self.integers).get(target)
        key = (target, values)
        if key in self.failed:
            return None
        if time.monotonic() > self.deadline:
            raise Timeout
        n = len(values)
        seen: set[tuple[int, ...]] = set()
        for k in range(1, n // 2 + 1):
            for idx in combinations(range(n), k):
                group = tuple(values[i] for i in idx)
                if group in seen:
                    continue
                seen.add(group)
                rest = tuple(v for i, v in enumerate(values) if i not in idx)
                for v, tv in _reach(group, self.integers, self.deadline).items():
                    for need, build in self._needs(target, v, tv):
                        if self.integers and (need.denominator != 1 or need < 0):
                            continue
                        found = self.solve(need, rest)
                        if found is not None:
                            return build(found)
        self.failed.add(key)
        return None

    @staticmethod
    def _needs(target: Fraction, v: Fraction, tv: Tree):
        yield target - v, lambda r: ("+", tv, r)
        yield v - target, lambda r: ("-", tv, r)
        yield target + v, lambda r: ("-", r, tv)
        if v:
            yield target / v, lambda r: ("*", tv, r)
            yield target * v, lambda r: ("/", r, tv)
        if target and v:  # rest = v / target; with v = 0 the rest would have to be 0, a division by zero
            yield v / target, lambda r: ("/", tv, r)


def solve(values: list[int], target: int, time_limit: float = 2.0) -> Tree | None:
    """Return a tree using every value once that equals target, or None if none was found in time."""
    ordered = tuple(sorted(values))
    goal = Fraction(target)
    deadline = time.monotonic() + time_limit
    for integers in (True, False):
        try:
            tree = _Search(integers, deadline).solve(goal, ordered)
        except Timeout:
            return None
        if tree is not None:
            assert evaluate(tree) == goal and sorted(leaves(tree)) == list(ordered)
            return tree
    return None
