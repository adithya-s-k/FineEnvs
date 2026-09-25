"""Trainer-side reward: correctness from Harbor's verifier, efficiency from the trace.

    reward = correctness x (1 + W_EFF * TOOL_BUDGET / (TOOL_BUDGET + n_tool_calls))

MULTIPLICATIVE, NOT ADDITIVE-WITH-A-GATE. An efficiency bonus that is merely ADDED has a trapdoor:
zero tool calls scores the MAXIMUM efficiency, so for a policy that cannot solve the task, doing
nothing becomes the best move available (0.300 vs 0.030 for a real attempt that fails). The policy
stops calling tools, `train_turn_fn=has_tool_call` then yields no trainable turns, and the group is
empty. Jobs 72452 and 72473 wedged at step 7 and 10 of 100 exactly this way, spending 4,076 E2B
sandboxes on 11 productive groups. A `if correct` gate patches that; multiplying STRUCTURALLY removes
it -- `correctness == 0` zeroes the product, and reward is monotone non-decreasing in BOTH arguments,
so efficiency can never be traded for correctness. The property survives refactoring; a gate may not.

WHY IT IS COMPUTED HERE AND NOT IN THE VERIFIER. A reward belongs in the sandbox only if the sandbox
is what makes it computable. Correctness needs the data, the gold answer and the tolerances. A
tool-call count needs the TRACE, which lives in the capture proxy. The suite's own grader tried to
read it from `/workdir/.n_tool_calls` and `$N_TOOL_CALLS`; nothing writes either, so it emitted `null`
forever, Harbor's `dict[str, float|int]` rejected the whole dict, and `correctness` went down with it
-- 86 of 250 tasks silently unscored until that was fixed at source (dataset rev 291c8e50).

WHY 1/(1+n/B) AND NOT THE REFERENCE'S clip(1 - n/B). Measured over run 77284 (Qwen3.5-2B, opencode,
118 logged steps), tool calls per rollout: p10 8.4, p50 14.7, p75 25.5, p90 56.6, max 125.5 -- a 15x
range, drifting 13.1 -> 43.0 between the first and last 30 steps because nothing bounded it. A linear
clamp cannot be both sensitive at 10 and unsaturated at 100:

    n_tool_calls      5      8     15     30     66     95    125
    clip(1-n/15)  0.667  0.467  0.000  0.000  0.000  0.000  0.000   <- inert above the MEDIAN
    clip(1-n/60)  0.917  0.867  0.750  0.500  0.000  0.000  0.000   <- inert exactly where it is needed
    15/(15+n)     0.750  0.652  0.500  0.333  0.185  0.136  0.107   <- graded across the whole range

The reciprocal keeps a gradient everywhere, and keeps it strongest near the budget, which is where we
want the policy to land. `TOOL_BUDGET` stays the reference's 15 and now reads as a half-credit point
rather than a cliff: efficiency is 0.5 at exactly 15 calls.

THE SECOND REASON, WHICH IS THE LARGER ONE. 19 of 118 steps in run 77284 logged `reward_std == 0`,
and ALL NINETEEN were groups where every generation SOLVED the task. Under pure-correctness reward an
all-correct group has zero advantage for every member: 8 sandboxes, no gradient, 16% of the run. Those
generations were not identical -- they differed in how long they took. Efficiency makes precisely
those groups trainable. This term buys signal from rollouts already paid for.

EFFICIENCY IS A TIE-BREAKER, NOT A COMPETING OBJECTIVE. Over the observed range the efficiency term
moves the reward by at most 0.3 x (0.750 - 0.107) = 0.193, against the 1.0 swing of correctness. When
a group disagrees about correctness, correctness dominates ~5:1; only when it agrees does efficiency
decide. That ratio is the design, so keep `W_EFF` well under 1.

A SOFT INCENTIVE IS NOT A BOUND. This shapes behaviour over many steps; it does not stop one rollout
running 235 turns (77284's observed `turns_max`) and blowing the packed row, which grows with the
SQUARE of the turn count -- `row_tokens_max` reached 40,598 against a 40,960 budget. The hard bound is
`max_model_calls` in the capture proxy, the only harness-agnostic step limit, since only 1 of 29 seams
honours `agent_step_limit` at all. Ship both; this one alone will not save the row.

Reads `train/tools/call_frequency` in trackio, which is `float(n_calls)` per rollout
(`async_rollout_worker.py:1052`) -- the exact quantity this reward acts on, already on the dashboard.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only, so this module imports (and unit-tests) without torch or vLLM
    from trl.experimental.async_grpo.openenv_harness import HarnessRolloutOutcome

logger = logging.getLogger(__name__)

W_EFF = float(os.environ.get("REWARD_W_TOOL_EFFICIENCY", "0.3"))
TOOL_BUDGET = float(os.environ.get("TOOL_BUDGET", "15"))

_announced = False


def tool_efficiency(n_tool_calls: int | None) -> float | None:
    """`B / (B + n)` in `(0, 1]`, or `None` when the count is unknown.

    Args:
        n_tool_calls (`int`, *optional*):
            Tool calls the agent made across every real turn, framework aux calls already dropped.

    Returns:
        `float` or `None`: `1.0` at zero calls, `0.5` at exactly `TOOL_BUDGET`, asymptotically `0` --
            never actually `0`, which is what keeps a gradient in the 60-125 call regime where the
            reference's `clip(1 - n/15)` is flat.
    """
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return TOOL_BUDGET / (TOOL_BUDGET + max(0, int(n_tool_calls)))


def data_agent_reward(outcome: "HarnessRolloutOutcome") -> float | None:
    """`correctness x (1 + W_EFF * efficiency)`, or `None` when the rollout is unscorable.

    Args:
        outcome (`HarnessRolloutOutcome`):
            What the rollout produced -- verifier reward, transcript, tool-call counts, timeout flag.

    Returns:
        `float` or `None`: `None` means UNSCORABLE and DROPS the rollout from its group baseline.
            Scoring it `0.0` would teach the policy that a crashed sandbox is as bad as a wrong
            answer, and would poison the baseline with a value nothing produced.
    """
    global _announced

    correctness = outcome.env_reward
    if correctness is None:
        logger.warning(
            "verifier did not run (tool_calls=%d); rollout unscorable, dropped from the baseline",
            outcome.tool_call_count,
        )
        return None

    correctness = float(correctness)
    if not _announced:
        _announced = True
        logger.warning(
            "reward = correctness x (1 + %.2f * %.0f/(%.0f + tool_calls)); "
            "max %.3f at 0 calls, %.3f at %.0f calls, ->1.0 as calls->inf",
            W_EFF, TOOL_BUDGET, TOOL_BUDGET, 1.0 + W_EFF, 1.0 + W_EFF * 0.5, TOOL_BUDGET,
        )
    if outcome.timed_out:
        # Kept, not zeroed: the verifier graded whatever work landed, and that is a measurement.
        logger.warning("agent timed out; keeping the verifier's %.3f on the partial work", correctness)

    eff = tool_efficiency(outcome.tool_call_count)
    if eff is None or correctness <= 0.0:
        # Nothing to scale, or scaling would invert the sign on a negative verifier score.
        return correctness
    return correctness * (1.0 + W_EFF * eff)
