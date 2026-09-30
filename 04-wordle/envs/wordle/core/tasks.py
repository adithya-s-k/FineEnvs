"""Train / eval split of the answer list.

The cut is deterministic: last 200 words in file order are eval, the rest
are train. File order is alphabetical, so this is not a random 200 — it is
the tail of the alphabet (`w`–`z` heavy). That is intentional. A random
split would put `crane`-like openers on both sides; an alphabetical tail
makes eval a slightly different letter distribution, which is the cheaper
cousin of GeoGuesser's country-capped holdout.
"""

from __future__ import annotations

from .words import ANSWERS, load_answers

EVAL_SIZE = 200


def train_tasks() -> tuple[str, ...]:
    answers = load_answers()
    if len(answers) <= EVAL_SIZE:
        raise ValueError("answer list shorter than the eval holdout")
    return answers[:-EVAL_SIZE]


def eval_tasks() -> tuple[str, ...]:
    answers = load_answers()
    return answers[-EVAL_SIZE:]
