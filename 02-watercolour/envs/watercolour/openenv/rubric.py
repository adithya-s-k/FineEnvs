# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition for the watercolour environment.

The children are thin readers: the work happens once in
[`~envs.watercolour_env.server.scoring.evaluate_submission`], which hangs its
results on the observation. Composing them therefore costs nothing and the tree
stays introspectable via `env.rubric.named_rubrics()`.

The weights are the ones from the final rubric in Narreddi's write-up, and they
live in [`~envs.watercolour_env.server.scoring`], the same place the
observation's own reward reads them from, so the two cannot disagree.

HPSv3's 0.30 is carried by [`CoverageInBand`], for the reasons in
[`~envs.watercolour_env.server.scoring`]. It previously sat here as a child that
always returned zero, and this tree is the one that matters: it produces the
reward on the observation, which is the number a trainer optimises. Adding the
coverage term to `scoring` alone left the two paths disagreeing by 0.30 on the
same submission, with the feedback string reporting a reward the policy was never
given. The claim below that they cannot disagree is only true while every term
lives in both, so a new term goes in both or in neither.

See RFC 004 for the rubric design: `rfcs/004-rubrics.md`.
"""


from __future__ import annotations

import sys
from pathlib import Path

# Two deliberate departures from the layout in CONTRIBUTING.md, both forced by the
# same thing: this folder is called `openenv` and so is the installed package.
#
# 1. There is no `__init__.py` here. With one, `import openenv` resolves to this
#    folder whenever the interpreter's working directory is the environment root,
#    which is what the Dockerfile sets, and `openenv.core` then does not exist.
# 2. The path entry is appended rather than inserted, and points at the folder
#    holding `core/`, so an installed package always wins over a local directory.
_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if not __package__ and _ENV_ROOT not in sys.path:
    sys.path.append(_ENV_ROOT)




from typing import Any

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

if __package__:
    from ..core.scoring import (
        GATE_WEIGHT,
        JUDGE_WEIGHT,
        LENGTH_WEIGHT,
        QUALITY_WEIGHT,
    )
else:
    from core.scoring import (
        GATE_WEIGHT,
        JUDGE_WEIGHT,
        LENGTH_WEIGHT,
        QUALITY_WEIGHT,
    )


class GatePassed(Rubric):
    """One if the submission painted honestly with the library."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the gate verdict as a score."""
        return 1.0 if observation.gate_passed else 0.0


class LengthRamp(Rubric):
    """How close the sketch is to the target elaboration."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the length score carried on the observation."""
        return observation.length_score


class JudgeScore(Rubric):
    """How the painting placed against the sampled references.

    Zero when no judge is configured, which is the same number this would
    contribute if it were left out of the tree entirely.
    """

    def forward(self, action: Any, observation: Any) -> float:
        """Return the comparative score carried on the observation."""
        return observation.judge_score


class QualityScore(Rubric):
    """The painting's absolute mark, judged on its own.

    Occupies the weight HPSv3 carries in the write-up, and its role: a dense
    reference-free term, so a policy has something to climb before it can beat
    anything in the pool. Zero when no scorer is configured, which is the same
    number this would contribute if it were left out of the tree entirely.
    """

    def forward(self, action: Any, observation: Any) -> float:
        """Return the absolute mark carried on the observation."""
        return observation.quality_score


def build_rubric() -> Rubric:
    """Assemble the reward tree.

    The gate leads and zeroes everything when it fails, so a submission that
    loaded somebody else's painting cannot collect credit for it, including the
    credit for having compiled. Past it the components are summed at their
    weights.

    Returns:
        [`~openenv.core.rubrics.Rubric`]: The composed rubric.

    Examples:

    ```python
    rubric = build_rubric()
    reward = rubric(action, observation)
    ```
    """
    return Sequential(
        Gate(GatePassed(), threshold=1.0),
        WeightedSum(
            [GatePassed(), LengthRamp(), JudgeScore(), QualityScore()],
            weights=[GATE_WEIGHT, LENGTH_WEIGHT, JUDGE_WEIGHT, QUALITY_WEIGHT],
        ),
    )
