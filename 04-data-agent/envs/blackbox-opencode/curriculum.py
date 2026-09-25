# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordering the prompts by difficulty, instead of shuffling them.

A PERMUTATION, NOT A FILTER (except where it says otherwise). Every prompt appears once, so "one
epoch" keeps its meaning and two runs under different curricula see the same work in a different
order -- which is what makes their reward curves comparable at all.

Two schedules, named rather than expressed as a mini-language, because a curriculum you cannot read
off the name is a curriculum nobody will check:

  `sprinkle`            medium throughout, hard evenly interspersed at the rate the pool implies.
                        Difficulty stays roughly stationary -- the right default when you want a
                        clean reward curve rather than a schedule.

  `warmup:<n>`          n EASY prompts first, then `sprinkle` over medium and hard. The warmup is
                        there because a group whose `num_generations` rollouts ALL score zero
                        contributes exactly zero gradient, and at the start of training that is the
                        likely outcome on the harder tiers. This one SUBSETS the pool: only n easy
                        prompts are used and the rest are dropped, so one epoch is
                        (n easy + all medium + all hard).

Deterministic given `seed`.
"""

from __future__ import annotations

import logging
import random
from typing import Any


logger = logging.getLogger(__name__)

TIERS = ("easy", "medium", "hard")


def apply_curriculum(rows: list[Any], tier_of: dict[int, str | None], spec: str, seed: int) -> list[Any]:
    """Reorder `rows` according to `spec`.

    Args:
        rows (`list`):
            Prompt rows, in any order.
        tier_of (`dict[int, str]`):
            `id(row) -> difficulty tier`. Passed in rather than derived here so this module never has
            to know how a row stores its instruction -- and so a lookup that silently returns `None`
            for every row, degrading the whole thing to a plain shuffle, is the caller's bug to make
            and the caller's to test.
        spec (`str`):
            `"sprinkle"` or `"warmup:<n>"`.
        seed (`int`):
            Makes the result reproducible.

    Returns:
        `list`: the reordered rows.
    """
    med = [r for r in rows if tier_of.get(id(r)) == "medium"]
    hard = [r for r in rows if tier_of.get(id(r)) == "hard"]
    other = [r for r in rows if tier_of.get(id(r)) not in ("medium", "hard")]
    rng = random.Random(seed)
    rng.shuffle(med)
    rng.shuffle(hard)
    rng.shuffle(other)

    if spec.startswith("warmup:"):
        n_easy = int(spec.split(":", 1)[1].split(",")[0])
        easy = [r for r in rows if tier_of.get(id(r)) == "easy"]
        rng.shuffle(easy)
        head = easy[:n_easy]
        rest = apply_curriculum(
            med + hard + [r for r in other if tier_of.get(id(r)) != "easy"], tier_of, "sprinkle", seed
        )
        dropped = len(easy) - len(head)
        if dropped:
            logger.warning(
                "curriculum %s: using %d easy prompts as warmup and DROPPING %d unused easy prompts; "
                "one epoch is %d prompts, not %d",
                spec,
                len(head),
                dropped,
                len(head) + len(rest),
                len(rows),
            )
        ordered = head + rest
    elif spec == "sprinkle":
        ordered = med + other
        if hard:
            every = max(1, len(ordered) // len(hard))
            for i, h in enumerate(hard):
                pos = min(len(ordered), (i + 1) * every + i)
                ordered.insert(pos, h)
    else:
        raise ValueError(f"unknown curriculum {spec!r}; use 'sprinkle' or 'warmup:<n>'")

    # A curriculum that grows or duplicates the pool trains some prompts twice per "epoch" and
    # reports nothing. Cheap to assert, expensive to discover from a reward curve.
    assert len(ordered) <= len(rows), f"curriculum GREW the pool {len(rows)} -> {len(ordered)}"
    assert len(set(map(id, ordered))) == len(ordered), "curriculum duplicated a prompt"
    return ordered
