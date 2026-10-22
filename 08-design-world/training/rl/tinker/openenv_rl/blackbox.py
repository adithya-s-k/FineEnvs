"""Training Tinker on loop-owning sessions (the black-box path, e.g. Harbor).

Here an agent harness (OpenCode, Claude Code, Codex, ...) runs its own loop inside a sandbox and
calls an OpenAI-compatible model endpoint through OpenEnv's capture proxy. Point that endpoint at
`TinkerChatServer`, which samples with Tinker and returns the token ids and logprobs the proxy
needs to capture at the `tokens` level. After each rollout the session hands back a validated
`TrainingTrace`; this module turns traces into Tinker `Datum`s with group-centred advantages,
using the trace's own loss masks unchanged (OpenEnv never tokenises and neither does this).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker
import torch
from openenv.core.harness import ResourceSessionFactory
from openenv.core.harness.training import TrainingTrace
from tinker import TensorData

logger = logging.getLogger(__name__)


@dataclass
class BlackBoxRollout:
    """One graded rollout of a loop-owning session."""

    trace: TrainingTrace
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BlackBoxGroup:
    """The usable rollouts of one task (a GRPO group); failed or ungraded ones are left out."""

    task: Any
    rollouts: list[BlackBoxRollout]
    failures: int = 0

    @property
    def rewards(self) -> list[float]:
        return [r.reward for r in self.rollouts]


def _sequences(trace: TrainingTrace) -> list[tuple[list[int], list[float], list[int]]]:
    """Merge turns whose prompt extends the previous turn's full sequence into one sequence.

    Returns `(tokens, logprobs, mask)` per merged sequence, each the length of `tokens`. Prompt
    positions carry logprob 0 and mask 0; completion positions carry the captured logprob and
    the trace's own loss-mask bit.
    """
    out: list[tuple[list[int], list[float], list[int]]] = []
    tokens: list[int] = []
    logprobs: list[float] = []
    mask: list[int] = []
    for turn in trace.turns:
        prompt, completion = turn.prompt_token_ids, turn.completion_token_ids
        turn_mask = list(turn.loss_mask[len(prompt) :])
        if tokens and prompt[: len(tokens)] == tokens:
            new_prompt = prompt[len(tokens) :]
        else:
            if tokens:
                out.append((tokens, logprobs, mask))
            tokens, logprobs, mask, new_prompt = [], [], [], list(prompt)
        tokens += new_prompt + list(completion)
        logprobs += [0.0] * len(new_prompt) + list(turn.per_token_logps)
        mask += [0] * len(new_prompt) + turn_mask
    if tokens:
        out.append((tokens, logprobs, mask))
    return out


def trace_to_datums(trace: TrainingTrace, advantage: float, *, keep_mask: bool = False) -> list[tinker.Datum]:
    """A captured trace as Tinker RL datums (`target_tokens`, `logprobs`, `advantages`).

    Args:
        trace: A producer-validated trace (`TrainableSession.fetch_training_trace()`).
        advantage: The rollout's advantage; applied only where the trace's loss mask is 1.
        keep_mask: Also emit the `mask` input (the cookbook strips it before `forward_backward`).

    Returns:
        One datum per merged sequence, ready for `forward_backward(..., "importance_sampling")`.
    """
    datums = []
    for tokens, logprobs, mask in _sequences(trace):
        if not any(mask) or len(tokens) < 2:
            continue
        inputs = {
            "target_tokens": TensorData.from_torch(torch.tensor(tokens[1:])),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs[1:], dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor([advantage * m for m in mask[1:]], dtype=torch.float32)),
        }
        if keep_mask:
            inputs["mask"] = TensorData.from_torch(torch.tensor(mask[1:], dtype=torch.float32))
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs=inputs,
            )
        )
    return datums


def group_advantages(rewards: Sequence[float], *, normalize_std: bool = False) -> list[float]:
    """Rewards centred within their group (the cookbook's convention), optionally scaled by std."""
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    centred = [r - mean for r in rewards]
    if normalize_std and len(rewards) > 1:
        std = statistics.pstdev(rewards)
        if std > 0:
            centred = [c / std for c in centred]
    return centred


def groups_to_datums(groups: Sequence[BlackBoxGroup], *, normalize_std: bool = False) -> list[tinker.Datum]:
    """Datums for a batch of groups; groups whose rewards are all equal teach nothing and are skipped."""
    datums: list[tinker.Datum] = []
    for group in groups:
        if len(group.rollouts) < 2 or len(set(group.rewards)) == 1:
            continue
        for rollout, adv in zip(group.rollouts, group_advantages(group.rewards, normalize_std=normalize_std)):
            datums.extend(trace_to_datums(rollout.trace, adv))
    return datums


def _one_rollout(
    factory: ResourceSessionFactory,
    task: Any,
    seed: int | None,
    timeout_s: float | None,
):
    session = factory.create(task, seed=seed)
    try:
        code = session.wait_for_completion(timeout_s)
        if code != 0:
            return None, {"exit_code": float(code)}
        trace = session.fetch_training_trace()
        verify = session.verify([], {"exit_code": code})
        if verify.env_reward is None:  # ungraded is not a zero
            return None, {"ungraded": 1.0}
        metrics = {
            k: float(v)
            for k, v in (verify.metrics or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return BlackBoxRollout(trace=trace, reward=float(verify.env_reward), metrics=metrics), {}
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("session close failed: %s", exc)


async def run_black_box_group(
    factory: ResourceSessionFactory,
    task: Any,
    group_size: int,
    *,
    seed: int | None = None,
    timeout_s: float | None = None,
) -> BlackBoxGroup:
    """Run `group_size` rollouts of one task concurrently on a loop-owning session factory.

    The factory's sessions must implement OpenEnv's `TrainableSession` protocol
    (`wait_for_completion`, `fetch_training_trace`, `verify`), as `HarborSessionFactory`'s do.
    """
    results = await asyncio.gather(
        *[asyncio.to_thread(_one_rollout, factory, task, seed, timeout_s) for _ in range(group_size)],
        return_exceptions=True,
    )
    rollouts, failures = [], 0
    for res in results:
        if isinstance(res, BaseException):
            logger.warning("rollout failed: %r", res)
            failures += 1
        elif res[0] is None:
            failures += 1
        else:
            rollouts.append(res[0])
    return BlackBoxGroup(task=task, rollouts=rollouts, failures=failures)


async def train_step(
    training_client: tinker.TrainingClient,
    groups: Sequence[BlackBoxGroup],
    *,
    learning_rate: float,
    loss_fn: str = "importance_sampling",
    normalize_std: bool = False,
) -> dict[str, float]:
    """One optimiser step on a batch of black-box groups. Returns summary metrics."""
    datums = groups_to_datums(groups, normalize_std=normalize_std)
    rewards = [r for g in groups for r in g.rewards]
    metrics = {
        "reward/mean": statistics.fmean(rewards) if rewards else 0.0,
        "rollouts/usable": float(len(rewards)),
        "rollouts/failed": float(sum(g.failures for g in groups)),
        "datums": float(len(datums)),
    }
    if not datums:
        return metrics
    fwd = await training_client.forward_backward_async(datums, loss_fn=loss_fn)
    opt = await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate))
    await fwd.result_async()
    await opt.result_async()
    return metrics
