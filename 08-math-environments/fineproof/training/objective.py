"""The training objective's two departures from the pinned TRL AsyncGRPO, made by subclassing (TRL is unchanged).

Our first runs rose for about 40 updates and then fell back to the base model, with proofs growing until most hit the
token cap. Two parts of TRL's defaults push that way (Dr. GRPO, https://huggingface.co/papers/2503.20783), and
QED-Nano's RL recipe drops both:

1. Advantages divided by the group's reward std. With a noisy judge, a group of near-equal proofs (std 0.02, say)
   turns judge noise into advantages of +-2. Here the advantage is the reward minus the group mean.
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

from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, _HarnessRolloutLoop

STD_EPS = 1e-8  # TRL's: advantage = (reward - mean) / (std + 1e-8)


def mean_centered(advantage: float, reward_std: float | None) -> float:
    """Undo TRL's division by the group's reward std (the std it used is stamped on each sample's metrics)."""
    if reward_std is None or not math.isfinite(reward_std):
        return advantage  # an unscored row: TRL already set its advantage to 0
    return advantage * (reward_std + STD_EPS)


class _MeanCenteredLoop(_HarnessRolloutLoop):
    """The rollout loop, run in the worker's child process: reward minus the group mean, no std division."""

    async def _score_group(self, group):
        samples = await super()._score_group(group)
        for sample in samples:
            sample.advantage = mean_centered(sample.advantage, sample.metrics.get("reward_std"))
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
