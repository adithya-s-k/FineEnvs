# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""How often GeoGuesser's published reward would have produced a dead GRPO group.

No checkpoints, no GPU. The numbers in LEARNINGS.md are enough to build a
within-task rollout model and ask: of 8 samples on the same location, how
often is the advantage exactly zero?

Two facts from the GeoGuesser write-up:

- Untrained 4B: median error 1,226 km, 29.5% of episodes scoring zero.
- Subtract-and-floor with mean cost 0.13 zeroed 77 of 200 episodes — a
  3,324 km miss and an 18,723 km miss both scored 0.0.

A dead group is a *task* property, not a population property. 29.5% zeros
drawn independently almost never fill a group of 8 (0.295^8 ≈ 5.7e-5).
Groups die when the eight rollouts of one location all land on the same
side of the cliff. That is the regime `classify_group` calls a cliff, and
it is the regime dynamic sampling and a denser reward are for.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from alive import classify_group, is_dead  # noqa: E402

DECAY_KM = 1492.7
LONG_DECAY_KM = 5000.0
COST = 0.13
MAX_COST_FRACTION = 0.2
NO_GUESS = 0.295
MEDIAN_KM = 1226.0
GROUP = 8
N_TASKS = 20_000


def game_subtract(distance_km: float | None, cost: float = COST) -> float:
    if distance_km is None:
        return 0.0
    return max(0.0, math.exp(-distance_km / DECAY_KM) - cost)


def mixture_multiply(distance_km: float | None, cost: float = COST) -> float:
    if distance_km is None:
        return 0.0
    short = math.exp(-distance_km / DECAY_KM)
    long = math.exp(-distance_km / LONG_DECAY_KM)
    score = 0.5 * short + 0.5 * long
    return min(1.0, score) * (1.0 - min(max(cost, 0.0), MAX_COST_FRACTION))


def sample_task_distances(rng: np.random.Generator, n_tasks: int, group: int) -> np.ndarray:
    """Within-task distances.

    Each task has a typical error drawn log-normal around the published
    median. The eight rollouts jitter that error with a 0.35 log-std — about
    a 1.4× typical ratio between the best and worst of eight, which is in
    the range of an untrained 4B guessing the same city eight different ways.
    Independently, each rollout fails to submit with probability 0.295.
    """
    task_median = rng.lognormal(mean=math.log(MEDIAN_KM), sigma=0.85, size=n_tasks)
    jitter = rng.normal(0.0, 0.35, size=(n_tasks, group))
    distances = task_median[:, None] * np.exp(jitter)
    missing = rng.random((n_tasks, group)) < NO_GUESS
    distances = distances.astype(np.float64)
    distances[missing] = np.nan
    return distances


def rewards_from(distances: np.ndarray, fn) -> np.ndarray:
    out = np.empty(distances.shape, dtype=np.float64)
    it = np.nditer(distances, flags=["multi_index"])
    for value in it:
        d = None if np.isnan(value) else float(value)
        out[it.multi_index] = fn(d)
    return out


def summarise(rewards: np.ndarray, distances: np.ndarray) -> dict:
    n = rewards.shape[0]
    dead = 0
    cliff = 0
    collapse = 0
    live = 0
    grpo_scale = []
    for i in range(n):
        row = rewards[i]
        # Fingerprint on the *distance*, not the reward. Eight different
        # kilometre errors that all floor to 0.0 are a cliff. Eight missing
        # guesses (nan) are a collapse of "never submit".
        fps = []
        for d in distances[i]:
            fps.append("none" if np.isnan(d) else round(float(d), 1))
        kind = classify_group(row, fingerprints=fps)
        if kind == "live":
            live += 1
        elif kind == "cliff":
            cliff += 1
            dead += 1
        else:
            collapse += 1
            dead += 1
        std = float(np.std(row))
        grpo_scale.append(1.0 / (std + 1e-4))
    return {
        "groups": n,
        "frac_dead": dead / n,
        "frac_live": live / n,
        "frac_cliff": cliff / n,
        "frac_collapse": collapse / n,
        "median_grpo_scale": float(np.median(grpo_scale)),
        "p95_grpo_scale": float(np.percentile(grpo_scale, 95)),
        "p99_grpo_scale": float(np.percentile(grpo_scale, 99)),
    }


def dynamic_sampling_rate(rewards: np.ndarray, extra_draws: np.ndarray) -> dict:
    """If the first group is dead, replace it once with a fresh group of 8."""
    n = rewards.shape[0]
    recovered = 0
    still_dead = 0
    already_live = 0
    for i in range(n):
        if not is_dead(rewards[i]):
            already_live += 1
            continue
        if is_dead(extra_draws[i]):
            still_dead += 1
        else:
            recovered += 1
    return {
        "already_live": already_live / n,
        "recovered_on_resample": recovered / n,
        "still_dead_after_one_resample": still_dead / n,
        "frac_dead_after_one_resample": still_dead / n,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    distances = sample_task_distances(rng, N_TASKS, GROUP)
    extra = sample_task_distances(rng, N_TASKS, GROUP)
    # Converged run-1 policy: one glance at a capital, ~1 turn, tiny jitter.
    # Distances cluster around 662 km (the published median of ckpt1000).
    collapse_d = rng.normal(662.0, 15.0, size=(N_TASKS, GROUP))
    collapse_d = np.clip(collapse_d, 1.0, None)
    report = {
        "model": {
            "median_km": MEDIAN_KM,
            "p_no_guess": NO_GUESS,
            "mean_cost": COST,
            "group": GROUP,
            "tasks": N_TASKS,
            "independent_p_all_zero": NO_GUESS**GROUP,
        },
        "game_subtract": summarise(rewards_from(distances, game_subtract), distances),
        "mixture_multiply": summarise(rewards_from(distances, mixture_multiply), distances),
        "run1_near_collapse_mixture": summarise(
            rewards_from(collapse_d, mixture_multiply), collapse_d
        ),
        # Eight identical guesses of the same city. Distance std is zero, so
        # group-std scaling hits the 1e-4 ceiling: 10,000×. This is the
        # plateau run 1 spent 750 steps on.
        "run1_identical_guess": summarise(
            rewards_from(np.full((N_TASKS, GROUP), 662.0), mixture_multiply),
            np.full((N_TASKS, GROUP), 662.0),
        ),
        "dynamic_sampling_game_subtract": dynamic_sampling_rate(
            rewards_from(distances, game_subtract),
            rewards_from(extra, game_subtract),
        ),
        "dynamic_sampling_mixture_multiply": dynamic_sampling_rate(
            rewards_from(distances, mixture_multiply),
            rewards_from(extra, mixture_multiply),
        ),
    }
    # Sanity: subtract-and-floor should zero a large minority of *episodes*.
    flat = rewards_from(distances, game_subtract).ravel()
    report["episode_frac_zero_game_subtract"] = float(np.mean(flat == 0.0))
    mix = rewards_from(distances, mixture_multiply).ravel()
    report["episode_frac_zero_mixture_multiply"] = float(np.mean(mix == 0.0))
    out = Path(__file__).resolve().parents[1] / "results" / "geoguesser-dead-groups.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
