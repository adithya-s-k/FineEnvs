"""Advantages that still exist when a GRPO group has no spread.

GeoGuesser already tells you to watch `frac_reward_zero_std`. If it is near
1, most groups teach nothing. TRL still has no dynamic sampling (the DAPO
paper's filter; the TRL docs mark it unsupported), so a step whose eight
rollouts share a reward still runs a backward pass on a zero advantage.

Two ways that happens, and they need opposite responses:

1. Cliff. Different trajectories, identical rewards. GeoGuesser's
   subtract-and-floor scored a 3,324 km miss the same as an 18,723 km miss
   — 77 of 200 episodes at exactly 0.0. The reward is blind. Resampling the
   task, or densifying the reward, can recover a gradient.
2. Collapse. The same trajectory, G times. Run 1's policy converged to a
   one-glance city guess; group std fell to 0.001 and the next 750 steps
   spent about $70 moving 0.013. Resampling draws the same action. Raise
   temperature or stop; more samples of the same task will not help.

`classify_group` tells those apart. `advantages` can use group-std (run 1's
amplifier), a leave-one-out baseline, or centered ranks — ranks stay
bounded when std collapses, which is the 99× blow-up run 1 logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Literal, Sequence

import numpy as np

Kind = Literal["live", "cliff", "collapse"]
AdvantageKind = Literal["grpo", "dr_grpo", "rank", "loo"]

# Matches TRL's 1e-4 in spirit; slightly looser so float32 reward ties count.
DEAD_STD = 1e-6


def _as_float(rewards: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(rewards), dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("need a group of at least 2 rewards")
    return array


def group_std(rewards: Sequence[float]) -> float:
    """Population std, the one GRPO divides by."""
    array = _as_float(rewards)
    return float(array.std(ddof=0))


def is_dead(rewards: Sequence[float], eps: float = DEAD_STD) -> bool:
    return group_std(rewards) <= eps


def classify_group(
    rewards: Sequence[float],
    fingerprints: Sequence[object] | None = None,
    eps: float = DEAD_STD,
) -> Kind:
    """Live, cliff, or collapse.

    A fingerprint is anything that identifies the trajectory: the guess
    sequence, a hash of tool calls, the completion text. If it is omitted,
    a dead group is called a cliff — the conservative label, because
    treating collapse as cliff makes you resample (wasteful) rather than
    treating cliff as collapse (which would stop you densifying the reward).
    """
    array = _as_float(rewards)
    if float(array.std(ddof=0)) > eps:
        return "live"
    if fingerprints is None:
        return "cliff"
    unique = {tuple(f) if isinstance(f, (list, tuple)) else f for f in fingerprints}
    if len(unique) <= 1:
        return "collapse"
    return "cliff"


def average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based average ranks, ties share the mean rank."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def advantages(
    rewards: Sequence[float],
    kind: AdvantageKind = "rank",
    eps: float = 1e-4,
) -> np.ndarray:
    """Group-relative advantages.

    - `grpo`: (r - mean) / (std + eps). Run 1. As std -> 0 the scale
      goes to 1/eps. With TRL's 1e-4 that is a 10,000× ceiling; run 1
      measured a typical 60× and a grad-norm spike above 11.
    - `dr_grpo`: r - mean. Run 2. No amplifier, and a tied group is still
      zeros.
    - `loo`: r_i - mean of the others. Same zeros on a tie, slightly
      higher variance than `dr_grpo` on a live group.
    - `rank`: centered average ranks / (n - 1). Bounded in [-0.5, 0.5],
      zero-sum, and still zero on a tie — but a one-unit ordering is
      always the same size, which is what you want when the reward's
      numeric gaps are an artefact of a cliff rather than of skill.
    """
    array = _as_float(rewards)
    n = array.size
    mean = float(array.mean())
    if kind == "grpo":
        std = float(array.std(ddof=0))
        return (array - mean) / (std + eps)
    if kind == "dr_grpo":
        return array - mean
    if kind == "loo":
        if n == 1:
            return array * 0.0
        total = float(array.sum())
        baseline = (total - array) / (n - 1)
        return array - baseline
    if kind == "rank":
        ranks = average_ranks(array)
        return (ranks - ranks.mean()) / max(n - 1, 1)
    raise ValueError(f"unknown advantage kind {kind!r}")


@dataclass
class ReplayQueue:
    """Oversample tasks that produced cliff groups.

    Collapse is not queued: the policy already agrees with itself on those.
    Live groups are not queued either. Cliff tasks are the ones whose
    reward could not tell eight different rollouts apart, which is exactly
    the 29.5% of GeoGuesser base episodes that scored zero, seen once, and
    never repeated.
    """

    weights: dict[int, float] = field(default_factory=dict)
    cliff_boost: float = 4.0
    decay: float = 0.5

    def observe(self, task_id: int, kind: Kind) -> None:
        current = self.weights.get(task_id, 1.0)
        if kind == "cliff":
            self.weights[task_id] = max(current, 1.0) * self.cliff_boost
        elif kind == "live":
            self.weights[task_id] = 1.0 + self.decay * (current - 1.0)
        else:
            # collapse: let the weight die off, do not boost
            self.weights[task_id] = 1.0 + self.decay * (current - 1.0)

    def sample(self, task_ids: Sequence[int], rng: np.random.Generator) -> int:
        ids = list(task_ids)
        w = np.array([self.weights.get(i, 1.0) for i in ids], dtype=np.float64)
        w = np.maximum(w, 1e-6)
        w /= w.sum()
        return int(rng.choice(ids, p=w))


@dataclass
class AliveStats:
    groups: int = 0
    live: int = 0
    cliff: int = 0
    collapse: int = 0
    skipped: int = 0
    resampled: int = 0

    def record(self, kind: Kind, *, skipped: bool = False, resampled: bool = False) -> None:
        self.groups += 1
        if kind == "live":
            self.live += 1
        elif kind == "cliff":
            self.cliff += 1
        else:
            self.collapse += 1
        if skipped:
            self.skipped += 1
        if resampled:
            self.resampled += 1

    def as_dict(self) -> dict[str, float]:
        n = max(self.groups, 1)
        return {
            "groups": self.groups,
            "frac_live": self.live / n,
            "frac_cliff": self.cliff / n,
            "frac_collapse": self.collapse / n,
            "frac_dead": (self.cliff + self.collapse) / n,
            "frac_skipped": self.skipped / n,
            "resampled": self.resampled,
        }


def should_skip(kind: Kind, *, skip_cliff: bool = True, skip_collapse: bool = True) -> bool:
    """Whether this group should be excluded from the backward pass."""
    if kind == "live":
        return False
    if kind == "cliff":
        return skip_cliff
    return skip_collapse


def entropy_nats(logprobs: Iterable[float]) -> float:
    """Mean negative logprob, a cheap policy-entropy proxy for sampled tokens."""
    values = [float(x) for x in logprobs]
    if not values:
        return 0.0
    return float(-sum(values) / len(values))


def clamp_grpo_scale(std: float, eps: float = 1e-4, max_scale: float = 10.0) -> float:
    """The factor GRPO would have multiplied by, capped.

    Run 1's uncapped factor was ~60× typical and 10,000× at a tie. Capping
    it is the cheap version of switching to ranks.
    """
    return min(max_scale, 1.0 / (std + eps))


def drop_zero_advantage_group(skip_dead: bool, adv: Sequence[float]) -> bool:
    """Whether to omit a tied group from the backward pass.

    Vanilla GRPO keeps the zero terms in the batch mean. Dropping them
    rescales the live groups that share the step. Only the alive arms
    skip a dead group.
    """
    array = np.asarray(list(adv), dtype=np.float64)
    if float(np.max(np.abs(array))) >= 1e-12:
        return False
    return bool(skip_dead)


def choose_task(
    index: int,
    n_tasks: int,
    replay: ReplayQueue,
    rng: np.random.Generator,
    mix: float,
) -> int:
    """Pick a task index, mixing replayed cliffs in with probability `mix`.

    Uniform `index % n` is the TRL dataset order. Replay is sampled from
    the queue's weights, not by rewriting a random other row — that row
    may already have been consumed in a one-pass dataset.
    """
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    index = int(index) % n_tasks
    if mix <= 0 or not replay.weights:
        return index
    if float(rng.random()) >= mix:
        return index
    return replay.sample(range(n_tasks), rng)


def group_rank_rewards(raws: Sequence[float], group_size: int) -> list[float]:
    """Centered ranks, one GRPO group at a time, for a TRL reward_func.

    TRL still subtracts the group mean afterwards; ranks are already
    zero-mean, so `scale_rewards=none` leaves them as advantages.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    values = [float(x) for x in raws]
    out: list[float] = []
    for i in range(0, len(values), group_size):
        chunk = values[i : i + group_size]
        if len(chunk) < 2:
            out.extend(chunk)
            continue
        out.extend(advantages(chunk, kind="rank").tolist())
    return out


def collapse_stop(frac_collapse: float, threshold: float) -> bool:
    """True when a run should halt because most groups are the same trajectory."""
    return float(frac_collapse) >= float(threshold)
