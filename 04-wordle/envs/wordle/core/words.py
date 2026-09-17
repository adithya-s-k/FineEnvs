"""Original Wordle answer list.

Source: the 2,309-word possible-answers file used by 3Blue1Brown's Wordle
notes, which is the original game's answer list minus a handful of later
removals. The words themselves are the game's; this file is a convenience
copy so a training run does not need the network.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ANSWERS_PATH = Path(__file__).with_name("answers.txt")


@lru_cache(maxsize=1)
def load_answers() -> tuple[str, ...]:
    """Load the answer list, lowercased, in file order."""
    text = _ANSWERS_PATH.read_text(encoding="utf-8")
    words = tuple(
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    )
    if not words:
        raise FileNotFoundError(f"no answers in {_ANSWERS_PATH}")
    for word in words:
        if len(word) != 5 or not word.isalpha():
            raise ValueError(f"invalid answer {word!r} in {_ANSWERS_PATH}")
    return words


ANSWERS: tuple[str, ...] = load_answers()
