"""Stateful Wordle episode.

Two departures from the demo game in `00-environments-101`:

- The answer is not written into the observation. GeoGuesser run 1 had to
  override the environment's reveal for the same reason: once the truth is in
  the trajectory, a completion that never guessed still contains the answer.
- Any 5-letter alphabetic string is a legal guess. Restricting to the answer
  list would turn the policy into a pointer over a set it should not see.
"""

from __future__ import annotations

import random

from .words import ANSWERS


def feedback_pattern(guess: str, answer: str) -> str:
    """Wordle colouring: greens first, then yellows from the leftover letters."""
    guess = guess.lower()
    answer = answer.lower()
    result = ["⬛"] * 5
    leftover = list(answer)
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = "🟩"
            leftover[i] = None
    for i, g in enumerate(guess):
        if result[i] == "🟩":
            continue
        if g in leftover:
            result[i] = "🟨"
            leftover[leftover.index(g)] = None
    return "".join(result)


class WordleGame:
    """One episode. `observe()` never names the answer."""

    def __init__(self, answer: str = "", max_guesses: int = 6, *, rng: random.Random | None = None):
        pool = ANSWERS
        if answer:
            self.answer = answer.lower()
        else:
            self.answer = (rng or random).choice(pool)
        self.max_guesses = max_guesses
        self.guesses: list[str] = []
        self.patterns: list[str] = []
        self.won = False
        self.done = False
        self.invalid_count = 0

    def guess(self, word: str) -> str:
        """Submit a guess. Invalid input does not consume a turn."""
        word = word.lower().strip()
        if self.done:
            return self.observe()
        if len(word) != 5 or not word.isalpha():
            self.invalid_count += 1
            remaining = self.max_guesses - len(self.guesses)
            return (
                f"Invalid guess {word!r} — need exactly 5 letters. "
                f"{remaining} guesses remaining."
            )
        pattern = feedback_pattern(word, self.answer)
        self.guesses.append(word)
        self.patterns.append(pattern)
        if word == self.answer:
            self.won = True
            self.done = True
        elif len(self.guesses) >= self.max_guesses:
            self.done = True
        return self.observe()

    def observe(self) -> str:
        """History the policy is allowed to see."""
        if not self.guesses:
            remaining = self.max_guesses
            return (
                "Guess the 5-letter word. "
                f"You have {remaining} guesses. "
                "Reply with guess <word>."
            )
        lines = [
            f"{i}. {g} {p}"
            for i, (g, p) in enumerate(zip(self.guesses, self.patterns), 1)
        ]
        if self.won:
            lines.append(f"Correct in {len(self.guesses)}.")
        elif self.done:
            lines.append("No guesses left.")
        else:
            left = self.max_guesses - len(self.guesses)
            lines.append(f"{left} guesses remaining.")
        return "\n".join(lines)
