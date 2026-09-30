"""Information gain of a Wordle guess, in bits.

The remaining set is the answer list consistent with the feedback so far.
A guess that splits that set in half is worth 1 bit; a guess that names the
answer is worth log2 of whatever was left. The policy never sees this set —
only the coloured history — so using it as a *reward* is not leakage. It is
the same trick as scoring GeoGuesser on kilometres the model is not told.

Patterns are scored in numpy against the whole pool at once. A training step
does this thousands of times; the Python loop over 2,309 words was the
thing that made a CPU run look like a GPU one.
"""

from __future__ import annotations

import math

import numpy as np

from .game import feedback_pattern
from .words import ANSWERS

# 0 grey, 1 yellow, 2 green. Packed as a base-3 int so a (N,) compare is one op.
_COLOUR_FROM_CHAR = {"⬛": 0, "🟨": 1, "🟩": 2}


def _pack_pattern(pattern: str) -> int:
    n = 0
    for ch in pattern:
        n = 3 * n + _COLOUR_FROM_CHAR[ch]
    return n


def _codes(words: tuple[str, ...]) -> np.ndarray:
    return np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8).reshape(len(words), 5)


_ANSWER_CODES = _codes(ANSWERS)


def _pattern_codes(guess: str, answers: np.ndarray) -> np.ndarray:
    """Packed colouring of `guess` against every row of `answers`."""
    g = np.frombuffer(guess.encode("ascii"), dtype=np.uint8)
    greens = answers == g
    leftover = answers.copy()
    leftover[greens] = 0
    yellows = np.zeros(answers.shape, dtype=bool)
    for i in range(5):
        gi = g[i]
        # Positions already green on this guess cannot also be yellow.
        eligible = ~greens[:, i]
        hits = (leftover == gi) & eligible[:, None]
        # First leftover occurrence of gi in each word, if any.
        any_hit = hits.any(axis=1) & eligible
        if not any_hit.any():
            continue
        # Consume one copy of gi from leftover for those words.
        first = hits.argmax(axis=1)
        rows = np.nonzero(any_hit)[0]
        leftover[rows, first[rows]] = 0
        yellows[rows, i] = True
    packed = greens.astype(np.int64) * 2 + yellows.astype(np.int64)
    # base-3 pack along the 5 positions
    return packed[:, 0] * 81 + packed[:, 1] * 27 + packed[:, 2] * 9 + packed[:, 3] * 3 + packed[:, 4]


def remaining_mask(
    guesses: list[str],
    patterns: list[str],
    pool: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Boolean mask over `pool` (or the answer list) still consistent with history."""
    if pool is None:
        answers = _ANSWER_CODES
        n = len(ANSWERS)
    else:
        answers = _codes(pool)
        n = len(pool)
    mask = np.ones(n, dtype=bool)
    for guess, pattern in zip(guesses, patterns):
        if len(guess) != 5 or not guess.isalpha():
            continue
        want = _pack_pattern(pattern)
        mask &= _pattern_codes(guess, answers) == want
    return mask


def remaining_words(
    guesses: list[str],
    patterns: list[str],
    pool: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Answers still consistent with the observed colouring."""
    words = pool if pool is not None else ANSWERS
    mask = remaining_mask(guesses, patterns, pool=pool)
    return tuple(w for w, keep in zip(words, mask) if keep)


def entropy_bits(n: int) -> float:
    """log2(n) with 0 for an empty set."""
    if n <= 1:
        return 0.0
    return math.log2(n)


def information_gain(
    guess: str,
    pattern: str,
    before: tuple[str, ...],
) -> float:
    """Realised bits: H(before) - H(after this guess/pattern).

    Realised, not expected. A lucky split pays more than a theoretically
    good guess that happened to land in a large bucket. Expected IG is the
    right *action selection* criterion for a solver; realised IG is the
    right *credit* for a sampled trajectory.
    """
    if len(guess) != 5 or not guess.isalpha() or not before:
        return 0.0
    n_before = len(before)
    n_after = int(remaining_mask([guess], [pattern], pool=before).sum())
    return max(0.0, entropy_bits(n_before) - entropy_bits(n_after))


def episode_information(
    guesses: list[str],
    patterns: list[str],
    pool: tuple[str, ...] | None = None,
) -> float:
    """Sum of realised IG over the episode, in bits."""
    words = pool if pool is not None else ANSWERS
    codes = _ANSWER_CODES if pool is None else _codes(words)
    mask = np.ones(len(words), dtype=bool)
    total = 0.0
    for guess, pattern in zip(guesses, patterns):
        if len(guess) != 5 or not guess.isalpha():
            continue
        n_before = int(mask.sum())
        want = _pack_pattern(pattern)
        hit = _pattern_codes(guess, codes) == want
        mask &= hit
        n_after = int(mask.sum())
        total += max(0.0, entropy_bits(n_before) - entropy_bits(n_after))
    return total
