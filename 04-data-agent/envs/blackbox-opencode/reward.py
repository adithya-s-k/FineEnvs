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

"""The training reward: correctness, plus a small efficiency bonus that is gated on a solve.

The SHAPE of this function matters far more than its constants, and it was arrived at by walking a
policy into an absorbing state twice. Do not simplify it without reading `data_agent_reward`.
"""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

# Credit for stating the right value in chat without filing it. Filing is worth 3.3x chat, so the
# gradient points at the contract; chat is worth infinitely more than silence, so there is a gradient
# at all. Measured: this lifts the fraction of rollouts with a non-zero reward from ~0.6% to ~17.5%,
# which is what makes groups non-uniform enough to learn from.
CHAT_ANSWER_CREDIT = float(os.environ.get("CHAT_ANSWER_CREDIT", "0.3"))

# The dataset's own default (its `grader.py` `_tool_efficiency`). Kept rather than re-chosen so this
# efficiency term means the same thing the benchmark's does.
TOOL_BUDGET = float(os.environ.get("TOOL_BUDGET", "15"))

# Weight on the efficiency BONUS. Deliberately small, and deliberately a bonus rather than a penalty.
EFFICIENCY_WEIGHT = float(os.environ.get("EFFICIENCY_WEIGHT", "0.1"))


def tool_efficiency(
    n_tool_calls: int | None, budget: float = TOOL_BUDGET
) -> float | None:
    """`1 - n/budget`, clamped to [0, 1]. The dataset's own definition."""
    if n_tool_calls is None or budget <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / budget))


def data_agent_reward(
    correctness: float | None, n_tool_calls: int | None
) -> float | None:
    """`correctness + EFFICIENCY_WEIGHT * tool_efficiency`, with the bonus gated on a solve.

    THE SHAPE IS CHOSEN AGAINST A FAILURE THAT ACTUALLY HAPPENED, TWICE.

    The first version subtracted a penalty of up to 0.5 for tool calls beyond a budget of 30. On
    Qwen3.5-4B that inverted the objective: opencode emits several tool calls per assistant message
    (~6.5/iteration measured, against ~0.3-0.8 on dense models), so the budget was crossed in about
    five turns, the penalty saturated, task reward went to 0, and the only remaining gradient was
    "use fewer tools". The policy took the cheapest route and stopped calling tools altogether -- and
    because `train_turn_fn=has_tool_call` reinforces only action turns, a policy that takes no
    actions produces no training rows, hence no gradient, and cannot climb back out. An absorbing
    state, reached by following the reward exactly as written.

    Reshaping the penalty into a BONUS was not enough. Ungated, the table read

        wrong + no tools    0.0 + 0.10 = 0.10
        wrong + wasteful    0.0 + 0.00 = 0.00

    and 0.10 is tied for the best outcome among failures while strictly dominating every real
    attempt that fails. For a policy whose solve rate is a few percent, correctness is effectively
    unreachable, so 0.10-by-inaction IS the achievable maximum. Measured, jobs 72452 (Qwen3.5-2B) and
    72473 (Qwen3-4B-Instruct-2507): all 495 empty groups scored reward_mean=0.1000 with
    reward_std=0.0000 -- every generation making zero tool calls -- and both runs froze at steps 7
    and 10 of 100, never recovering across 294 and 201 consecutive empty groups.

    GATED, the table is

        correct + wasteful    1.0 + 0.00 = 1.00     still beats everything incorrect
        correct + efficient   1.0 + 0.10 = 1.10     efficiency breaks ties among CORRECT runs only
        wrong   + anything    0.0 + 0.00 = 0.00     no reward for inaction

    Cost of the gate, stated plainly: a group of eight failures has zero reward variance and yields
    no gradient. That does not stall the trainer -- zero variance gives zero advantages, which is
    harmless -- but throughput depends on groups containing both outcomes.

    Args:
        correctness (`float`, *optional*):
            The graded score, or `None` if the rollout could not be graded at all.
        n_tool_calls (`int`, *optional*):
            Tool calls the agent made, used only for the efficiency bonus.

    Returns:
        `float` or `None`: The training reward. `None` is NOT zero -- an ungraded rollout is dropped
        from the group baseline rather than averaged in as a failure.
    """
    if correctness is None:
        return None
    correctness = float(correctness)
    # LOUD, because this collapse was silent for 28 minutes across 495 groups while every other gate
    # stayed green (capture_level=tokens, no OOM, reward_std>0 at the step level). A rollout that made
    # zero tool calls yields no trainable turns under `has_tool_call`, so a sustained run of them
    # starves the trainer with no error raised anywhere.
    if n_tool_calls == 0:
        logger.warning(
            "rollout made ZERO tool calls -> no trainable turns under has_tool_call. A sustained run "
            "of these wedges the trainer (jobs 72452/72473 froze exactly this way)."
        )
    eff = tool_efficiency(n_tool_calls)
    # THE GATE. Ungated, `1 - n/15` makes zero tool calls the highest-scoring behaviour available to
    # a policy that cannot solve the task.
    if eff is None or correctness < 1.0:
        return correctness
    return correctness + EFFICIENCY_WEIGHT * eff
