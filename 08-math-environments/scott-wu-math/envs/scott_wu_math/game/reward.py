"""The grader: one answer in, one reward out.

- 1.0 for an expression that uses every card exactly once and equals the target.
- Shaped mode (default) adds partial credit up to 0.3 for an expression that uses every card
  exactly once but misses: ``0.3 * c**2`` with ``c = max(0, 1 - |value - target| / (0.1 * target))``,
  so only answers within 10% of the target earn anything. A group of failed attempts still
  differs, which GRPO needs, and an exact answer is always worth far more than a near one.
- 0 for anything else: no answer, a parse error, a division by zero, a missing or extra card.

Binary mode gives 1 or 0.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .expr import check

PARTIAL_MAX = 0.3
TOLERANCE = 0.1
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def closeness(value: Fraction, target: int) -> float:
    scale = max(1.0, TOLERANCE * abs(target))
    return max(0.0, 1.0 - float(abs(value - target)) / scale)


def grade(expression: str | None, card_values: list[int], target: int, shaped: bool = True) -> dict:
    if expression is None or not expression.strip():
        return {"reward": 0.0, "exact": False, "valid": False, "error": "no answer", "check": None}
    result = check(expression, card_values, target)
    valid = result.uses_all_cards and result.value is not None
    if result.exact:
        reward = 1.0
    elif valid and shaped:
        reward = round(PARTIAL_MAX * closeness(result.value, target) ** 2, 6)
    else:
        reward = 0.0
    return {
        "reward": reward,
        "exact": result.exact,
        "valid": valid,
        "error": result.error,
        "check": result.to_dict(),
    }


def extract_answer(completion: str) -> str | None:
    """The last <answer>...</answer> outside any <think> block, or None."""
    text = _THINK.sub("", completion or "")
    found = _ANSWER.findall(text)
    return found[-1].strip() if found else None
