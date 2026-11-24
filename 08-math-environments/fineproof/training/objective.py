"""The training objective's two departures from the pinned TRL AsyncGRPO, made by subclassing (TRL is unchanged).

Our first runs rose for about 40 updates and then fell back to the base model, with proofs growing until most hit the
token cap. Two parts of TRL's defaults push that way (Dr. GRPO, https://huggingface.co/papers/2503.20783), and
QED-Nano's RL recipe drops both:

1. Advantages divided by the group's reward std. With a noisy judge, a group of near-equal proofs (std 0.02, say)
   turns judge noise into advantages of +-2. Here the advantage is the reward minus the group mean, computed from the
   group's own rewards (not by undoing TRL's division with a std TRL reports, whose metric name changes between TRL
   versions: on TRL main it is `reward_group_std`, and a lookup that misses would silently restore the division).
2. Each rollout's loss divided by its own length. AsyncGRPO trains one rollout per micro-step (packing is off, see
   train.py), so its token mean is a per-rollout mean: a 16k-token wrong proof is pushed down 1/16k per token, a 4k
   one 1/4k, and long wrong proofs get cheap. Here every rollout's token sum is divided by the same constant, the
   output cap.

Adam rescales the update, so neither change alters the step size much; they change how rollouts and tokens are
weighted against each other.

The third change, reward 0 for a proof cut off at the cap, is in tasks.py: it acts on the reward, not the loss.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, _HarnessRolloutLoop


def mean_centered(rewards: Sequence[float | None]) -> list[float]:
    """One advantage per rollout: its reward minus the mean of the group's scored rewards, never divided by their std.

    An unscored rollout (reward None or NaN: no grade) gets 0 and is left out of the mean, as TRL does.
    """
    scored = [r for r in rewards if r is not None and math.isfinite(r)]
    mean = sum(scored) / len(scored) if scored else 0.0
    return [float(r) - mean if r is not None and math.isfinite(r) else 0.0 for r in rewards]


class _MeanCenteredLoop(_HarnessRolloutLoop):
    """The rollout loop, run in the worker's child process: reward minus the group mean, no std division.

    TRL builds the samples (one per training row; a rollout gives one row per training sequence, in rollout order)
    with its std-divided advantage; each row's advantage is then replaced by its rollout's mean-centred reward. The
    rewards are the group's own: the harness's per-rollout reward is this trainer's only reward source.
    """

    async def _score_group(self, group):
        samples = await super()._score_group(group)
        if not samples:  # newer TRL drops a group whose rewards all tie; its advantages would all be 0 anyway
            return samples
        if getattr(self, "reward_funcs", None) or getattr(self, "_env_reward_types", None):
            raise RuntimeError("mean-centred advantages assume the rollout reward is the only reward source")
        advantages = mean_centered(group.rollout_rewards)
        rows = [len(sequences) for sequences in group.completions_sequences]
        if len(rows) != len(advantages) or sum(rows) != len(samples):
            raise RuntimeError(f"cannot match {len(samples)} training rows to {len(advantages)} rollouts {rows}; "
                               "TRL's sample layout changed")
        position = 0
        for advantage, count in zip(advantages, rows):
            for sample in samples[position:position + count]:
                sample.advantage = advantage
            position += count
        return samples


class MeanCenteredHarnessWorker(HarnessRolloutWorker):
    _loop_cls = _MeanCenteredLoop


class ConstantNormTrainer(AsyncGRPOTrainer):
    """AsyncGRPO with each micro-batch's token sum divided by `loss_norm_tokens` rather than by its token count."""

    def __init__(self, *args, loss_norm_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_norm_tokens = float(loss_norm_tokens)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # TRL divides the micro-batch's summed token loss by its tokens per rank (global_n_tokens / world size);
        # multiplying by tokens / loss_norm_tokens leaves the sum divided by the constant instead.
        tokens = (inputs["global_n_tokens"][0] / self.accelerator.num_processes).clamp(min=1.0)
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        return loss * (tokens / self.loss_norm_tokens).to(loss.dtype)
