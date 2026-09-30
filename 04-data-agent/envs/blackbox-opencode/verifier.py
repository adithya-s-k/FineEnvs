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

"""Grade one data-agent rollout: read the answer the agent filed, compare it to gold.

Grading runs on the HOST, not by executing the suite's own `tests/test.sh` inside the sandbox. Three
reasons: that script pip-installs at grade time and can call out to an LLM judge, which is
nondeterministic, billable and a network dependency inside every rollout; the host already holds the
authoritative tool-call count from the capture document, which the in-container grader cannot see;
and one more in-sandbox exec is one more thing that fails in a way indistinguishable from a wrong
answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .grader import grade
from .reward import CHAT_ANSWER_CREDIT
from .task import ANSWER_PATH, DataAgentTask


logger = logging.getLogger(__name__)


def looks_like_a_command(text: str) -> bool:
    """True if `text` is the agent NARRATING a command rather than stating a value.

    Load-bearing, because without it partial credit rewards exactly the wrong behaviour. Measured on
    job 72695: 8 of 19 awards (42%) went to strings like

        echo -n "2.1410474569861977" > /workdir/answer.txt

    which the grader scored correct on the embedded number. That pays the model for SAYING it will
    file the answer while never running the command -- a worse incentive than the inaction it
    replaced, and one a policy learns quickly because narrating is cheaper than executing.

    Rejects redirection and piping, a leading shell verb, and any mention of the answer path. A bare
    value never contains these: `2.14`, `936`, `C`, `Wii Sports` and `212,357` all pass.
    """
    t = text.strip()
    if ">" in t or "|" in t:
        return True
    if "answer.txt" in t or ANSWER_PATH in t:
        return True
    return bool(re.match(r"(?i)^(echo|cat|printf|tee|python3?|bash|sh|open|with)\b", t))


@dataclass
class Grade:
    """The outcome of grading one rollout.

    Attributes:
        correctness (`float`, *optional*):
            1.0 filed and right, `CHAT_ANSWER_CREDIT` right in chat only, 0.0 wrong, `None` ungradable.
        answer (`str`, *optional*):
            What the agent actually submitted, for reporting.
        source (`str`):
            `"file"`, `"chat"`, or `"none"` -- which path produced the answer.
        graded_by (`str`):
            Which comparison tier matched, straight from the grader.
    """

    correctness: float | None
    answer: str | None
    source: str
    graded_by: str = ""


def read_filed_answer(
    read_text: Callable[[str], str | None], paths: tuple[str, ...]
) -> str | None:
    """Read the answer file, trying each candidate path in order.

    There are TWO workdirs in the sandbox image -- `/workdir` (what the instruction names) and
    `{home}/workdir` (what the harness cds into) -- so an agent that writes `echo -n 42 > answer.txt`
    RELATIVE to its cwd lands in the second. Confirmed live: after a relative write,
    `/home/user/workdir/answer.txt` is present and `/workdir/answer.txt` is absent, and neither path
    errors, so the rollout scores 0 as though the agent never answered.

    EMPTY COUNTS AS ABSENT. A sandbox handle may RETURN `""` for a missing file rather than raising.
    An earlier version broke out of the loop on `""`, fell through to the no-answer branch, and
    returned 0.0 without ever consulting the transcript -- so partial credit silently never fired.
    That was caught twice by a live test expecting 0.3 and getting 0.0.
    """
    for path in paths:
        try:
            got = read_text(path)
        except Exception:  # a missing path is normal; try the next one
            continue
        if got is not None and got.strip():
            return got.strip()
    return None


def grade_rollout(
    task: DataAgentTask,
    read_text: Callable[[str], str | None],
    answer_paths: tuple[str, ...],
    final_message: str | None = None,
) -> Grade:
    """Grade a finished rollout.

    Args:
        task (`DataAgentTask`):
            The task, carrying gold and the comparison mode.
        read_text (`Callable`):
            Reads a path out of the sandbox, returning `None` or raising when absent.
        answer_paths (`tuple[str, ...]`):
            Candidate answer paths, already resolved for this backend's home directory.
        final_message (`str`, *optional*):
            The agent's last assistant message, used only for partial credit.

    Returns:
        `Grade`: correctness, the submitted answer, and which path produced it.
    """
    filed = read_filed_answer(read_text, answer_paths)
    if filed:
        result = grade(
            task.answer,
            filed,
            question=task.question,
            reward_mode=task.reward_mode,
            abs_tol=task.atol,
            rel_tol=task.rtol,
        )
        return Grade(float(result.reward), filed, "file", result.method)

    # No filed answer. Before scoring 0, check whether the agent COMPUTED the right value and merely
    # failed to file it: those are different failures and only one of them is about data analysis.
    if final_message and not looks_like_a_command(final_message):
        result = grade(
            task.answer,
            final_message,
            question=task.question,
            reward_mode=task.reward_mode,
            abs_tol=task.atol,
            rel_tol=task.rtol,
        )
        if result.reward >= 1.0:
            logger.info(
                "no answer file, but the CHAT answer is correct (%r) -> partial %.2f",
                final_message[:40],
                CHAT_ANSWER_CREDIT,
            )
            return Grade(
                CHAT_ANSWER_CREDIT, final_message.strip(), "chat", result.method
            )

    logger.info("no %s and no correct chat answer -> correctness 0.0", ANSWER_PATH)
    return Grade(0.0, None, "none")


def answer_paths_for(home: str) -> tuple[str, ...]:
    """Candidate answer paths for a backend whose sandbox home is `home`.

    The home differs by backend -- E2B runs as `user`, HF sandboxes as root -- which is why this is
    computed from the backend rather than hardcoded anywhere.
    """
    from .task import ANSWER_PATHS

    return tuple(dict.fromkeys(p.format(home=home.rstrip("/")) for p in ANSWER_PATHS))


def metadata_for(
    task: DataAgentTask, grade_result: Grade, n_tool_calls: int | None
) -> dict[str, Any]:
    """Auditable record of how this rollout was scored, carried on the result."""
    return {
        "task_id": task.task_id,
        "difficulty_tier": task.difficulty_tier,
        "reward_mode": task.reward_mode,
        "answer_source": grade_result.source,
        "graded_by": grade_result.graded_by,
        "tool_calls": n_tool_calls,
    }
