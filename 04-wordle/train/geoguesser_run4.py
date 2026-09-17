"""Drop-in pieces for a GeoGuesser run that classifies dead groups.

Does not launch GeoGuesser. It is the bit of 03-geoguesser/train that is
missing: given eight rewards and eight turn counts from one task, decide
whether the group is live, a reward cliff, or a collapsed policy, and
whether the step should still run a backward pass.

Wire it into `grpo_geoguesser.py` by recording `(reward, turns, actions)`
per rollout and calling `inspect_group` at the end of the generation batch.
The recommended run-4 config, from LEARNINGS.md plus this classifier:

    SCALE_REWARDS=none          # ranks / loo, not the 60× group-std amplifier
    BETA=0
    ACCUM=2                     # one task per step, so std is within-task
    COST_SCALE=0.2              # run 3's remaining lever, not 1.0
    MAX_STEPS=250               # run 1 plateaued here; 750 more steps did nothing
    ALIVE_SKIP_CLIFF=1          # do not train on zero-std cliffs
    ALIVE_STOP_COLLAPSE=0.8     # stop if 80% of a window is collapse

Run 1's gain came from the amplifier plus a cost that made 'not looking'
optimal. Run 4 is the opposite bet: keep the policy looking, throw away
groups that cannot teach, and stop when it has collapsed rather than
paying another $70 to watch the plateau.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from alive import Kind, classify_group, is_dead, should_skip


@dataclass(frozen=True)
class GroupReport:
    kind: Kind
    dead: bool
    skip: bool
    mean_turns: float
    mean_reward: float
    recommendation: str


def inspect_group(
    rewards: list[float],
    fingerprints: list[object],
    turns: list[int] | None = None,
    *,
    skip_cliff: bool = True,
    skip_collapse: bool = True,
    collapse_turn_threshold: float = 1.5,
) -> GroupReport:
    """Classify one GRPO group the way GeoGuesser should have.

    `fingerprints` should be the action sequence, not the reward. Using the
    reward as a fingerprint mis-labels a cliff (many distances, one zero)
    as collapse (one trajectory, G times).
    """
    kind = classify_group(rewards, fingerprints)
    dead = is_dead(rewards)
    skip = should_skip(kind, skip_cliff=skip_cliff, skip_collapse=skip_collapse)
    mean_turns = float(sum(turns) / len(turns)) if turns else float("nan")
    mean_reward = float(sum(rewards) / len(rewards))
    if kind == "collapse" and (turns and mean_turns <= collapse_turn_threshold):
        recommendation = (
            "collapsed to a one-glance policy — raise temperature or stop; "
            "resampling this task will draw the same guess"
        )
    elif kind == "cliff":
        recommendation = (
            "reward cannot see the difference between these rollouts — "
            "resample, densify the distance curve, or skip the backward pass"
        )
    else:
        recommendation = "train"
    return GroupReport(
        kind=kind,
        dead=dead,
        skip=skip,
        mean_turns=mean_turns,
        mean_reward=mean_reward,
        recommendation=recommendation,
    )


def run4_env() -> dict[str, str]:
    """Environment for a GeoGuesser launch that uses this classifier."""
    return {
        "SCALE_REWARDS": "none",
        "BETA": "0",
        "ACCUM": "2",
        "COST_SCALE": "0.2",
        "MAX_STEPS": "250",
        "NUM_GENERATIONS": "8",
        "SAVE_STEPS": "25",
        "MAX_TURNS": "12",
    }
