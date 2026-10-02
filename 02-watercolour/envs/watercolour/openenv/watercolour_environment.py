# SPDX-License-Identifier: BSD-3-Clause

"""The watercolour environment.

One episode is one painting. `reset()` names a subject and returns the prompt
along with the API reference the model needs, `step()` paints the submitted
sketch and judges it against the reference pool, and the episode ends.

Keeping episodes to a single exchange is deliberate. What is being measured is
whether a model can write code that paints well without ever seeing the result,
and letting it look and retry would measure something else.
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
if _ENV_ROOT not in sys.path:
    sys.path.append(_ENV_ROOT)




import asyncio
import base64
import os
from typing import Any, Optional

from openenv.core.env_server import Environment

try:
    from models import WatercolourAction, WatercolourObservation, WatercolourState
    from core.domains import get_domain
    from core.pairwise_judge import DEFAULT_JUDGE_MODEL, HFVisionClient, PairwiseJudge
    from core.prompt import system_prompt
    from core.quality import HPSv3Scorer, QualityScorer
    from core.render import shared_renderer
    from .rubric import build_rubric
    from core.scoring import Evaluation, evaluate_submission
    from core.tasks import make_task, sample_task, Task
except ImportError:
    from models import WatercolourAction, WatercolourObservation, WatercolourState
    from core.domains import get_domain
    from core.pairwise_judge import (
        DEFAULT_JUDGE_MODEL,
        HFVisionClient,
        PairwiseJudge,
    )
    from core.prompt import system_prompt
    from core.quality import HPSv3Scorer, QualityScorer
    from core.render import shared_renderer
    from rubric import build_rubric
    from core.scoring import Evaluation, evaluate_submission
    from core.tasks import make_task, sample_task, Task


def _judge_from_env(references: int, domain) -> PairwiseJudge | None:
    """Build a judge from environment variables, or `None` to run offline.

    A missing token is not an error. The environment stays usable on gate
    admission alone, which is what tests and a first smoke run want.
    """
    if os.environ.get("WATERCOLOUR_DISABLE_JUDGE", "").lower() in {"1", "true", "yes"}:
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model = os.environ.get("WATERCOLOUR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    try:
        pool = None
        if domain.pool.exists() and any(domain.pool.glob("*.png")):
            pool = [(p.name, p.read_bytes()) for p in sorted(domain.pool.glob("*.png"))]
        return PairwiseJudge(
            HFVisionClient(model=model, api_key=token),
            pool=pool,
            references=references,
            criteria=domain.judge_criteria,
        )
    except Exception:
        # No usable credentials. Fall back to gate-only scoring rather than
        # failing every episode.
        return None


def _scorer_from_env(domain):
    """Build the dense scorer from environment variables, or `None`.

    `WATERCOLOUR_HPSV3_URL` selects the real preference model, served by its own
    Space. Without it the stand-in is used: the judge model asked for a mark out
    of ten, sharing the judge's credentials because both are vision calls to the
    same endpoint. See [`~envs.watercolour_env.server.quality`] for what the two
    measure and how far apart they are on our own pool.
    """
    hpsv3 = os.environ.get("WATERCOLOUR_HPSV3_URL")
    if hpsv3:
        return HPSv3Scorer(hpsv3)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model = os.environ.get("WATERCOLOUR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    try:
        return QualityScorer(
            HFVisionClient(model=model, api_key=token),
            criteria=domain.judge_criteria,
        )
    except Exception:
        # Same fallback as the judge: score on what is free rather than fail
        # every episode.
        return None


class WatercolourEnvironment(
    Environment[WatercolourAction, WatercolourObservation, WatercolourState]
):
    """Paints submitted p5.brush sketches and judges them against references.

    Args:
        subject (`str`, *optional*):
            Fix the subject instead of sampling one, which is what a
            leaderboard run wants: without it every reset draws a fresh subject
            and two models are never asked the same question.
        domain (`str`, *optional*):
            Which painting subject to serve, `"hibiscus"` or `"jellyfish"`.
            Defaults to the hibiscus task, which reproduces Narreddi's. The
            domain carries its own subjects, composition guidance, judge criteria
            and reference pool; everything else is shared.
        scorer ([`QualityScorer`], *optional*):
            Absolute scorer for the dense term. Built alongside the judge when
            omitted, and left out when a judge is injected.
        judge ([`PairwiseJudge`], *optional*):
            Comparative scorer. When omitted and `enable_judge` is `True`, one
            is built from a Hugging Face Inference Providers token if the
            ambient environment supplies one.
        enable_judge (`bool`, *optional*, defaults to `True`):
            Set `False` to score on gate admission alone, with no vision calls.
        references (`int`, *optional*, defaults to `2`):
            How many references each painting is compared against. Cost is two
            vision calls per reference, since every comparison runs in both
            presentation orders.
        return_image (`bool`, *optional*, defaults to `False`):
            Include the painting in the observation as base64. Useful for
            building a gallery, wasteful during training.

    Examples:

    ```python
    env = WatercolourEnvironment(subject="two ripe plums")
    observation = env.reset()
    print(observation.prompt)
    ```
    """

    # Each session gets its own environment, and the renderer hands every
    # submission a fresh browser page, so sessions do not share sketch state.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        subject: str | None = None,
        judge: PairwiseJudge | None = None,
        scorer: QualityScorer | None = None,
        enable_judge: bool = True,
        references: int = 4,
        return_image: bool = False,
        revisions: int = 0,
        domain: str | None = None,
    ):
        self._domain = get_domain(domain)
        if not enable_judge:
            self._judge = None
        else:
            self._judge = (
                judge
                if judge is not None
                else _judge_from_env(references, self._domain)
            )
        # The absolute mark rides on the same switch and the same credentials as the
        # pairwise judge, because both are vision calls to the same model, and a run
        # that cannot make one cannot make the other. Built from the environment
        # only when the judge was too: an injected judge means a caller supplying
        # its own vision client, and building a real one alongside it would put a
        # network call in the middle of every test that passes a stub.
        if scorer is not None:
            self._scorer = scorer
        elif judge is None and self._judge is not None:
            self._scorer = _scorer_from_env(self._domain)
        else:
            self._scorer = None
        super().__init__(rubric=build_rubric())
        self._fixed_subject = subject
        self._references = references
        self._return_image = return_image
        self._revisions = revisions
        # Shared across sessions: see `shared_renderer`.
        self._renderer = shared_renderer()
        self._state = WatercolourState()
        self._task: Task | None = None
        # Per-episode overrides, reapplied from the constructor values on every
        # reset. A deployed Space is configured once but serves callers who want
        # different things from it: a training loop wants one reference and no
        # image, and the probe that reports on it wants the picture back.
        self._episode_seed: int | None = None
        self._episode_revisions = revisions
        self._episode_references = references
        self._episode_return_image = return_image

    @property
    def state(self) -> WatercolourState:
        """[`WatercolourState`]: The current episode state."""
        return self._state

    @property
    def task(self) -> Task | None:
        """[`Task`] or `None`: The task sampled for this episode."""
        return self._task

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WatercolourObservation:
        """Name a subject and return its prompt.

        Args:
            seed (`int`, *optional*):
                Makes both subject selection and the reference draw reproducible,
                and it has to arrive here rather than on `step`: OpenEnv's client
                accepts `**kwargs` on `step` and documents them as ignored, so
                they never reach the wire. Passing the same seed to every rollout
                in a GRPO group is what makes them face identical references, and
                identical references are what turn a within-group reward
                difference into a statement about the paintings instead of about
                the draw. Measured on one painting scored six times, the draw
                alone moves the judge score from 0.250 to 0.625.
            episode_id (`str`, *optional*):
                Identifier recorded on the state.
            subject (`str`, *optional*):
                Pin the subject for this episode, taking precedence over the
                constructor and over sampling.
            references (`int`, *optional*):
                Override how many references this episode is judged against.
                Per-episode because the client cannot reach the constructor of
                an already-deployed environment, and a training loop paying two
                vision calls per reference wants a different number than a
                one-off report does.
            return_image (`bool`, *optional*):
                Override whether this episode's painting comes back in the
                observation.

        Returns:
            [`WatercolourObservation`]: Carrying the prompt and the API
                reference, with `done` false and no reward yet.
        """
        subject = kwargs.get("subject") or self._fixed_subject
        task = (
            make_task(subject)
            if subject
            else sample_task(seed=seed, subjects=self._domain.subjects)
        )
        self._task = task
        self._episode_seed = seed
        revisions = kwargs.get("revisions")
        self._episode_revisions = int(
            self._revisions if revisions is None else revisions
        )
        self._state.revisions_used = 0
        references = kwargs.get("references")
        self._episode_references = int(
            self._references if references is None else references
        )
        self._episode_return_image = bool(
            self._return_image
            if kwargs.get("return_image") is None
            else kwargs["return_image"]
        )
        self._state = WatercolourState(
            episode_id=episode_id, step_count=0, task_id=task.task_id, submitted=False
        )
        return self._apply_transform(
            WatercolourObservation(
                prompt=task.prompt,
                system_prompt=system_prompt(self._domain),
                task_id=task.task_id,
                subject=task.subject,
                done=False,
                reward=None,
            )
        )

    def step(
        self,
        action: WatercolourAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WatercolourObservation:
        """Paint and score a submission. Synchronous wrapper around [`step_async`]."""
        return asyncio.run(self.step_async(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: WatercolourAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WatercolourObservation:
        """Paint a submission, judge it, and end the episode.

        Args:
            action ([`WatercolourAction`]):
                The model's reply, from which the sketch is extracted.
            timeout_s (`float`, *optional*):
                Unused. Rendering is bounded by its own deadline and judging by
                the vision client's timeout.
            seed (`int`, *optional*):
                Makes the sampled references reproducible.

        Returns:
            [`WatercolourObservation`]: With `done` true, the reward, and the
                full scoring breakdown.

        Raises:
            RuntimeError: If called before [`reset`].
        """
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")

        evaluation = await evaluate_submission(
            action.response,
            self._task,
            self._renderer,
            self._judge,
            # From `reset`, not from here: `step` kwargs are dropped by the
            # client before they reach the server.
            seed=self._episode_seed,
            references=self._episode_references,
            scorer=self._scorer,
        )
        self._state.step_count += 1
        # Revisions. Zero by default, which is the single-shot episode every run so
        # far has trained on and the only shape the reward was ever measured
        # against. With a budget, the episode stays open: the painting comes back
        # with the image and a critique of its own properties, and the policy gets
        # to look and try again. That is a different task, easier and arguably more
        # interesting, and the reward is the last painting's rather than the best of
        # the series, so a revision has to actually improve on what it replaced.
        #
        # The critique never mentions the references or the score. See
        # `Evaluation.critique` for why: with a shared seed the whole group faces
        # the same eight references, so naming the outcome names the opponents, and
        # the policy would learn to beat a draw instead of to paint.
        self._state.revisions_used += 1
        last = self._state.revisions_used > self._episode_revisions
        self._state.submitted = last
        observation = self._to_observation(evaluation)
        observation.critique = evaluation.critique
        observation.revisions_left = max(
            0, self._episode_revisions - self._state.revisions_used + 1
        )
        observation.done = last
        # The rubric containers hand back a coroutine whenever they are called
        # from inside a running loop, even with entirely synchronous children,
        # so the reward has to be awaited through the async helper here.
        observation.reward = await self._apply_rubric_async(action, observation)
        return self._apply_transform(observation)

    def _to_observation(self, evaluation: Evaluation) -> WatercolourObservation:
        render = evaluation.gate.render
        image: str | None = None
        if self._episode_return_image and render is not None:
            image = base64.b64encode(render.png).decode()
        return WatercolourObservation(
            prompt=evaluation.task.prompt,
            system_prompt=system_prompt(self._domain),
            task_id=evaluation.task.task_id,
            subject=evaluation.task.subject,
            feedback=evaluation.feedback,
            gate_passed=evaluation.gate_passed,
            length_score=evaluation.length_score,
            judge_score=evaluation.judge_score,
            judged=evaluation.judged,
            quality_score=evaluation.quality_score,
            quality_scored=evaluation.quality_scored,
            render_unavailable=evaluation.render_unavailable,
            judge_weight=evaluation.to_dict()["judge_weight"],
            paint_fraction=render.paint_fraction if render else 0.0,
            finished=render.finished if render else False,
            violations=list(evaluation.gate.violations),
            js_errors=render.errors if render else [],
            breakdown=evaluation.to_dict(),
            image_png_base64=image,
            done=True,
        )

    async def close(self) -> None:
        """End the session.

        The browser is deliberately left running: it is shared with every other
        session in the process, so closing it here would pull it out from under
        them. It goes away with the process.
        """
        return None
