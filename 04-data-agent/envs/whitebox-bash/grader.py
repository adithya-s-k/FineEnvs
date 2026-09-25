# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoring, server-side.

Three rules carried over from the black-box data-agent env, each of which was learned the expensive
way:

  * AN UNGRADED ROLLOUT IS `None`, NEVER 0.0. A sandbox that died and a policy that answered wrongly
    are different events, and collapsing them teaches the model that the dead sandbox was its fault.
    `None` drops the rollout from its group baseline instead.
  * THE EFFICIENCY BONUS IS GATED ON A SOLVE AND IS NEVER A PENALTY. Ungated, "make no tool calls at
    all" becomes the highest-scoring move for a policy that cannot solve the task.
  * A SUBMITTED ANSWER THAT IS REALLY A COMMAND EARNS NOTHING. On the data-agent env 42% of partial
    credit once went to strings like `echo -n "2.14" > answer.txt` -- paying the model for narrating
    the submission rather than making it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A solve is worth this; everything else is a small addition on top, so the ordering of outcomes can
# never be inverted by the shaping terms.
CORRECT_CREDIT = 1.0
# Paid only when the task ships a `check` and it passes. Independent evidence that the side effect
# actually happened, which a string comparison cannot see.
CHECK_CREDIT = 0.25
# Gated on a solve. Small on purpose: it breaks ties between two correct agents, it does not decide
# between a correct and an incorrect one.
EFFICIENCY_WEIGHT = 0.10
# Tool calls treated as "free" before the efficiency term starts decaying.
TOOL_BUDGET = 10

_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|[>|]{1,2}\s*\S+|\$\(|`",
)


def looks_like_a_command(answer: str) -> bool:
    """True when the 'answer' is really the shell line that would produce it.

    Checked before any credit: `echo -n "42" > out.txt` contains 42 and would otherwise score as a
    correct answer of 42.
    """
    if not answer:
        return False
    return bool(_COMMAND_SHAPED.search(answer.strip()))


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding quotes and trailing punctuation."""
    t = (text or "").strip().strip("'\"").strip()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".").lower()


def answers_match(submitted: str, gold: str) -> bool:
    """Exact after normalisation, then numeric with a tolerance.

    Numeric comparison is separate because `4`, `4.0` and `4 ` are the same answer and a string
    comparison says they are not -- which reads as a model that cannot count.
    """
    s, g = normalise(submitted), normalise(gold)
    if not s:
        return False
    if s == g:
        return True
    try:
        sv, gv = float(s.replace(",", "")), float(g.replace(",", ""))
    except (TypeError, ValueError):
        return False
    return abs(sv - gv) <= max(1e-6, abs(gv) * 1e-6)


@dataclass
class Verdict:
    """The outcome of one episode.

    `reward` of `None` means UNGRADED -- see the module docstring.
    """

    reward: float | None
    correct: bool = False
    check_passed: bool | None = None
    submitted: str = ""
    n_tool_calls: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "reward": self.reward,
            "correct": self.correct,
            "check_passed": self.check_passed,
            "submitted": self.submitted,
            "n_tool_calls": self.n_tool_calls,
            "note": self.note,
        }


def grade(
    *,
    submitted: str | None,
    gold: str,
    n_tool_calls: int,
    check_passed: bool | None = None,
    sandbox_alive: bool = True,
    correct_override: bool | None = None,
) -> Verdict:
    """Score one episode.

    Args:
        submitted (`str`, *optional*):
            What the agent passed to `submit`. `None` means it never submitted.
        gold (`str`):
            The task's gold answer.
        n_tool_calls (`int`):
            How many tools the agent actually invoked. Observable here because this is a white-box
            environment; the black-box path had to infer it from free text.
        check_passed (`bool`, *optional*):
            Result of the task's `check` script, or `None` when the task ships none.
        sandbox_alive (`bool`, *optional*, defaults to `True`):
            False when the sandbox died mid-episode, which makes the rollout UNGRADED.
        correct_override (`bool`, *optional*):
            Correctness decided elsewhere, used for data-agent tasks so this environment and the
            black-box one agree on what is right. The shaping below (efficiency, the command-shaped
            guard) still applies; only the correct/incorrect decision is taken from here.

    Returns:
        `Verdict`: with `reward=None` when the episode could not be judged.
    """
    if not sandbox_alive:
        return Verdict(reward=None, n_tool_calls=n_tool_calls,
                       note="sandbox died; ungraded, not wrong")
    if submitted is None:
        # Never submitting IS a failure of the task, not an infrastructure fault: the agent had the
        # tool and the budget and did not use it. Scored 0.0, not None.
        return Verdict(reward=0.0, n_tool_calls=n_tool_calls, note="no answer submitted")
    if looks_like_a_command(submitted):
        return Verdict(reward=0.0, submitted=submitted, n_tool_calls=n_tool_calls,
                       note="submission is a command, not an answer")

    correct = answers_match(submitted, gold) if correct_override is None else correct_override
    reward = CORRECT_CREDIT if correct else 0.0
    if check_passed:
        reward += CHECK_CREDIT
    if correct:
        # Decays from 1 to 0 across TOOL_BUDGET extra calls; never negative, never paid on a failure.
        over = max(0, n_tool_calls - TOOL_BUDGET)
        reward += EFFICIENCY_WEIGHT * max(0.0, 1.0 - over / float(TOOL_BUDGET))
    return Verdict(
        reward=round(reward, 6),
        correct=correct,
        check_passed=check_passed,
        submitted=submitted,
        n_tool_calls=n_tool_calls,
    )
