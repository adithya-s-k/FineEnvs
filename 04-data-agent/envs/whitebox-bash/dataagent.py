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

"""The real data-agent tasks, borrowed from the black-box environment next door.

WHY REUSE RATHER THAN REIMPLEMENT
`blackbox-opencode` already loads `HuggingEnvs/data-agent`, stages each task's tables out of an HF
bucket, and grades answers with the right numeric tolerance and list handling. Re-deriving any of
that here would produce a second source of truth for the same decisions, and the two would drift --
which is the failure this repo's conventions exist to prevent.

Reusing it buys something better than convenience: PARITY. The white-box and black-box environments
then run the SAME tasks, staged the SAME way, graded by the SAME code, and the only remaining
difference is who owns the agent loop. That makes white-box vs black-box a controlled comparison
instead of two numbers that cannot be put beside each other.

What this module does NOT reuse is the efficiency shaping in `../grader.py`: the black-box reward was
tuned for an agent whose tool calls are only inferrable from text, whereas here every call is
counted exactly. Correctness comes from the shared grader; the shaping stays local and explicit.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .tasks import Task


logger = logging.getLogger(__name__)

# The instruction the black-box agent gets tells it to write its answer to a file, because that is
# how a loop-owning agent submits. Here the agent submits through a TOOL, so that sentence would be
# actively misleading -- it would spend turns writing a file nobody reads. Rewritten at load time.
_FILE_SUBMIT_MARKERS = ("answer.txt", "/workdir/answer")


def _reword_submission(instruction: str) -> str:
    """Replace file-submission wording with tool-submission wording.

    Left verbatim when no marker is present, so a task that never mentioned a file is untouched.
    """
    if not any(m in instruction for m in _FILE_SUBMIT_MARKERS):
        return instruction
    lines = [ln for ln in instruction.splitlines()
             if not any(m in ln for m in _FILE_SUBMIT_MARKERS)]
    return "\n".join(lines).rstrip() + (
        "\n\nWhen you have the answer, call submit_solution with it. Submit only the value itself, "
        "not the command that would produce it."
    )


# How many times to attempt the per-episode bucket staging, and how long to back off.
# Every episode lists and downloads from `hf://buckets/AdithyaSK/jupyter-agent-kaggle-all`, so at
# num_generations=8 that is 8 concurrent tree listings per step and ~800 over a 100-step run against
# ONE bucket. A single transient 504 from the bucket API killed a whole run at step 6/100:
#     HfHubHTTPError: Server error '504 Gateway Timeout'
#     for url 'https://huggingface.co/api/buckets/AdithyaSK/jupyter-agent-kaggle-all/tree...'
# A transient upstream error should cost ONE episode, not the run. Safe to retry because the staging
# script short-circuits when the input directory already has files, so a partial success is not
# repeated -- it is resumed.
STAGING_ATTEMPTS = int(os.environ.get("WHITE_BOX_BASH_STAGING_ATTEMPTS", "4"))
STAGING_BACKOFF_S = int(os.environ.get("WHITE_BOX_BASH_STAGING_BACKOFF_S", "5"))


def _with_retry(setup: str, attempts: int = STAGING_ATTEMPTS,
                backoff_s: int = STAGING_BACKOFF_S) -> str:
    """Wrap the staging shell in a bounded retry with linear backoff.

    The original runs in a SUBSHELL so its `set -e` cannot abort the retry loop, and the loop fails
    loudly at the end rather than letting the agent start against an empty input directory -- which
    would score 0 for a reason indistinguishable from a wrong answer.
    """
    if not setup.strip():
        return setup
    return (
        "__staged=0\n"
        f"for __attempt in $(seq 1 {attempts}); do\n"
        f"  if ( {setup}\n  ); then __staged=1; break; fi\n"
        f"  echo \"[stage] attempt $__attempt failed; retrying\" >&2\n"
        f"  sleep $(( __attempt * {backoff_s} ))\n"
        "done\n"
        f"[ \"$__staged\" = 1 ] || {{ echo '[stage] FATAL: staging failed after {attempts} attempts' >&2; exit 1; }}\n"
    )


def available() -> bool:
    """Whether the sibling environment is importable. Reported, never silently worked around."""
    try:
        import data_agent_env  # noqa: F401
    except Exception:
        return False
    return True


def load(split: str, limit: int = 0) -> tuple[Task, ...]:
    """Data-agent tasks for `split`, as white-box `Task`s.

    Args:
        split (`str`):
            A data-agent split name: `<base>` or `<base>:<tier>`, e.g. `train:medium`. Tier is part
            of the NAME, never a filter, because the index is the task's identity downstream.
        limit (`int`, *optional*):
            Keep only the first `limit` tasks. `0` keeps all.

    Returns:
        `tuple[Task, ...]`: with `setup` staging the task's tables and `answer` the withheld gold.
    """
    from data_agent_env import tasks as da_tasks

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or None
    if not token:
        # Staging pulls from an HF bucket. Without a token the pull fails inside the sandbox and the
        # agent is handed an empty input directory -- it then scores 0 for a reason indistinguishable
        # from a wrong answer. Say so here rather than let every rollout fail plausibly.
        logger.warning(
            "HF_TOKEN is not set; data-agent bucket staging will fail and every rollout will score "
            "zero in a way that looks like a bad policy"
        )

    rows = da_tasks.rows_for(split)
    if limit:
        rows = rows[:limit]
    out: list[Task] = []
    for i, row in enumerate(rows):
        t = da_tasks.task_at(split, i)
        out.append(
            Task(
                instruction=_reword_submission(t.instruction),
                answer=str(t.answer),
                difficulty=str(getattr(t, "difficulty_tier", "") or "medium"),
                setup=_with_retry(t.setup_shell(token)),
                check="",
                metadata={
                    "source": "data-agent",
                    "split": split,
                    "index": i,
                    # Carried so the shared grader can apply the right comparison. A numeric answer
                    # compared as a string reads as a model that cannot count.
                    "reward_mode": getattr(t, "reward_mode", "") or "",
                    "atol": getattr(t, "atol", None),
                    "rtol": getattr(t, "rtol", None),
                    "env": t.env(token),
                },
            )
        )
    return tuple(out)


def grade_answer(task: Task, submitted: str) -> bool:
    """Correctness via the BLACK-BOX environment's grader, so both envs agree on what is right.

    Falls back to this package's own comparison only if the sibling is unavailable, and says so --
    a silent fallback would mean the two environments disagreed about correctness without anyone
    noticing.
    """
    meta = task.metadata or {}
    try:
        from data_agent_env import grader as da_grader

        # `grade` returns a GradeResult(score, method), not a bool, and its tolerances DEFAULT
        # rather than accept None -- passing None straight through would blow up inside the numeric
        # comparison. `judge=False` keeps grading deterministic and free of an external model; the
        # four tiers below it (exact, numeric with percent/fraction bridging, order-insensitive list,
        # math-verify) are what actually decide these answers.
        tol = {}
        if meta.get("atol") is not None:
            tol["abs_tol"] = float(meta["atol"])
        if meta.get("rtol") is not None:
            tol["rel_tol"] = float(meta["rtol"])
        result = da_grader.grade(
            gold=task.answer,
            candidate=submitted,
            question=task.instruction,
            reward_mode=str(meta.get("reward_mode") or ""),
            judge=False,
            **tol,
        )
        # The field is `reward`, not `score` (GradeResult(reward=..., method=...)).
        return float(result.reward) >= 1.0
    except Exception as exc:
        logger.warning("data-agent grader unavailable (%s); falling back to local comparison", exc)
        from .grader import answers_match

        return answers_match(submitted, task.answer)
