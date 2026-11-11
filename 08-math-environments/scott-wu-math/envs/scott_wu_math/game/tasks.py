"""Levels and task generation.

The levels follow the video: the 24 game with four cards, then "once that got too easy, you take
eight cards and you make the current year", then nine cards for any four-digit number someone
names. Three-digit with six cards sits between the two as a step up from 24.

Every generated task is solvable: the deal is checked with the solver and redealt if it fails.
"""

from __future__ import annotations

import datetime as _dt
import random
import time
from dataclasses import dataclass

from .cards import Card, cards_from_ranks, deal
from .expr import Tree, steps, to_text
from .solver import solve

CURRENT_YEAR = _dt.date.today().year


@dataclass(frozen=True)
class Level:
    name: str
    title: str
    n_cards: int
    low: int
    high: int


LEVELS = {
    "24": Level("24", "24", 4, 24, 24),
    "three-digit": Level("three-digit", "Three-digit", 6, 100, 999),
    "year": Level("year", "Year", 8, 1900, 2100),
    "four-digit": Level("four-digit", "Four-digit", 9, 1000, 9999),
}
DEFAULT_LEVEL = "year"

# A deal is solved while the caller waits, so custom deals stay where the solver answers in well under
# a second: up to nine cards (where it is complete) and four-digit targets, as in the video.
MIN_CARDS, MAX_CARDS = 2, 9
MAX_TARGET = 9_999


def cards_for_target(target: int) -> int:
    """How many cards a custom target gets: 4 for up to 99, 6 for three digits, 8 for 1000-2999, then 9."""
    if target < 100:
        return 4
    if target < 1000:
        return 6
    if target < 3000:
        return 8
    return 9


def make_task(
    rng: random.Random,
    level: str | None = None,
    target: int | None = None,
    n_cards: int | None = None,
    ranks: list[int] | None = None,
    task_id: str | None = None,
    time_limit: float = 2.0,
    max_deals: int = 50,
    budget: float | None = None,
) -> dict:
    """Deal a solvable task.

    With no `target`, the level picks one from its range. A given `target` keeps the level's card
    count, or without a level picks one by size (`cards_for_target`). Given `ranks` are dealt as
    they are and may have no solution (then `solution` is None).

    The solver gets `time_limit` seconds per deal and at most `max_deals` deals; `budget`, if given,
    caps the seconds spent on all of them together.
    """
    if level is not None and level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose one of {', '.join(LEVELS)}")
    if target is None:
        lvl = LEVELS[level or DEFAULT_LEVEL]
        level, target = lvl.name, rng.randint(lvl.low, lvl.high)
    target = int(target)
    if not 0 <= target <= MAX_TARGET:
        raise ValueError(f"target must be a whole number from 0 to {MAX_TARGET:,}")
    started = time.monotonic()
    if ranks is not None:
        if not MIN_CARDS <= len(ranks) <= MAX_CARDS:
            raise ValueError(f"play between {MIN_CARDS} and {MAX_CARDS} cards, not {len(ranks)}")
        cards = cards_from_ranks(list(ranks), rng)
        limit = time_limit if budget is None else min(time_limit, budget)
        tree = solve([c.rank for c in cards], target, limit)
        return _task(cards, target, tree, level or "custom", task_id)
    n = n_cards or (LEVELS[level].n_cards if level else cards_for_target(target))
    if not MIN_CARDS <= n <= MAX_CARDS:
        raise ValueError(f"deal between {MIN_CARDS} and {MAX_CARDS} cards, not {n}")
    for _ in range(max_deals):
        limit = time_limit
        if budget is not None:
            limit = min(limit, budget - (time.monotonic() - started))
            if limit <= 0:
                break
        cards = deal(n, rng)
        tree = solve([c.rank for c in cards], target, limit)
        if tree is not None:
            return _task(cards, target, tree, level or "custom", task_id)
    limits = f"{max_deals} deals" + ("" if budget is None else f" and {budget:g} s")
    raise ValueError(
        f"no solvable deal of {n} cards for {target} found within {limits}; try another target or card count"
    )


def _task(cards: list[Card], target: int, tree: Tree | None, level: str, task_id: str | None) -> dict:
    return {
        "id": task_id or f"{level}-{target}-" + "".join(c.label for c in cards),
        "level": level,
        "target": target,
        "cards": [c.to_dict() for c in cards],
        "solution": None if tree is None else to_text(tree),
        "solution_steps": [] if tree is None else steps(tree),
    }


def public(task: dict) -> dict:
    """The task as an agent may see it: no reference solution."""
    return {k: v for k, v in task.items() if k not in ("solution", "solution_steps")}
