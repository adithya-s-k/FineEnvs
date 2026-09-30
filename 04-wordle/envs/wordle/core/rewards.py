"""Reward shapes for a Wordle episode.

`sparse` is win/loss with a small speed bonus — the shape that produces
all-zero GRPO groups early in training, when almost nobody solves.

`greens` is the demo game's partial credit: 0.1 per green on the best guess.
It still ties together every episode that never found a green.

`process` pays realised information gain, scaled by log2(|answers|) so a
perfect collapse of the set is 1.0, plus a solve bonus. Two failed episodes
that reduced the set differently no longer share a reward.
"""

from __future__ import annotations

import math

from .information import episode_information
from .words import ANSWERS

SOLVE_BONUS = 0.5
SPEED_BONUS = 0.5


def sparse_reward(
    *,
    won: bool,
    n_guesses: int,
    max_guesses: int = 6,
    invalid_count: int = 0,
) -> float:
    """1.0 plus leftover-guess bonus on a solve; 0 otherwise, minus invalids."""
    invalid_penalty = 0.05 * invalid_count
    if not won:
        return max(0.0, 0.0 - invalid_penalty)
    speed = SPEED_BONUS * max(0.0, 1.0 - n_guesses / max_guesses)
    return max(0.0, 1.0 + speed - invalid_penalty)


def terminal_reward(
    *,
    won: bool,
    n_guesses: int,
    patterns: list[str],
    max_guesses: int = 6,
    invalid_count: int = 0,
) -> float:
    """The 00-environments-101 shape: solve bonus, else 0.1 per best green."""
    invalid_penalty = 0.05 * invalid_count
    if won:
        speed = SPEED_BONUS * max(0.0, 1.0 - n_guesses / max_guesses)
        return max(0.0, 1.0 + speed - invalid_penalty)
    best = 0
    for pattern in patterns:
        best = max(best, pattern.count("🟩"))
    return max(0.0, 0.1 * best - invalid_penalty)


def process_reward(
    *,
    won: bool,
    n_guesses: int,
    guesses: list[str],
    patterns: list[str],
    max_guesses: int = 6,
    invalid_count: int = 0,
    pool: tuple[str, ...] | None = None,
) -> float:
    """Information gain in [0, 1], plus a solve bonus that still prefers speed.

    Scaled by log2 of the answer list, so the units are 'fraction of the
    prior entropy removed', not raw bits. A policy that never solves but
    consistently halves the set still outscores one that guesses noise.
    """
    answers = pool if pool is not None else ANSWERS
    prior = math.log2(len(answers)) if len(answers) > 1 else 1.0
    ig = episode_information(guesses, patterns, pool=answers) / prior
    ig = min(1.0, max(0.0, ig))
    invalid_penalty = 0.05 * invalid_count
    if won:
        speed = SPEED_BONUS * max(0.0, 1.0 - n_guesses / max_guesses)
        return max(0.0, SOLVE_BONUS + speed + 0.5 * ig - invalid_penalty)
    return max(0.0, ig - invalid_penalty)


def reward_for(
    shape: str,
    *,
    won: bool,
    n_guesses: int,
    guesses: list[str],
    patterns: list[str],
    max_guesses: int = 6,
    invalid_count: int = 0,
    pool: tuple[str, ...] | None = None,
) -> float:
    """Dispatch on `sparse` / `greens` / `process`."""
    if shape == "sparse":
        return sparse_reward(
            won=won,
            n_guesses=n_guesses,
            max_guesses=max_guesses,
            invalid_count=invalid_count,
        )
    if shape == "greens":
        return terminal_reward(
            won=won,
            n_guesses=n_guesses,
            patterns=patterns,
            max_guesses=max_guesses,
            invalid_count=invalid_count,
        )
    if shape == "process":
        return process_reward(
            won=won,
            n_guesses=n_guesses,
            guesses=guesses,
            patterns=patterns,
            max_guesses=max_guesses,
            invalid_count=invalid_count,
            pool=pool,
        )
    raise ValueError(f"unknown reward shape {shape!r}")
