"""Scott Wu Math: deal cards, name a target, make it with + - * / using every card once.

One episode is one deal and one graded submission. `reset` deals (from a split of the task set,
from a level, or for a custom target or hand); `step` takes the final expression, grades it once,
and ends the episode. The reference solution is only returned after that.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from ..game.cards import Card
from ..game.prompts import SYSTEM_PROMPT, user_prompt
from ..game.reward import grade
from ..game.tasks import CURRENT_YEAR, make_task, public
from ..models import CardAction, CardObservation, CardState

TASKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tasks"
SPLITS = {"train": "train", "test": "test", "video": "test"}
_TASKS: dict[str, list[dict]] = {}


def load_split(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose one of {', '.join(SPLITS)}")
    if split not in _TASKS:
        path = TASKS_DIR / f"{split}.jsonl"
        _TASKS[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return _TASKS[split]


class ScottWuMathEnvironment(Environment[CardAction, CardObservation, CardState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task: dict | None = None
        self._state = CardState(episode_id=str(uuid4()))
        self._started = 0.0

    # -- episode ------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: Optional[str] = None,
        index: Optional[int] = None,
        level: Optional[str] = None,
        target: Optional[int] = None,
        n_cards: Optional[int] = None,
        cards: Optional[list[int]] = None,
        reward_mode: str = "shaped",
        **kwargs: Any,
    ) -> CardObservation:
        """Deal a task.

        - `split` + `index`: a fixed task from `train`, `test` or `video` (the two hands from the video).
        - `level`: a fresh deal at `24` (4 cards), `three-digit` (6), `year` (8) or `four-digit` (9).
        - `target` (and optionally `n_cards`): a fresh deal for your own number, as in the video.
        - `cards` (values 1-13) with `target`: play a hand you choose; it may have no solution.
        - nothing: eight cards for the current year.
        """
        if reward_mode not in ("shaped", "binary"):
            raise ValueError("reward_mode is 'shaped' or 'binary'")
        rng = random.Random(seed)
        if split is not None:
            tasks = load_split(split)
            i = rng.randrange(len(tasks)) if index is None else int(index)
            task = tasks[i]
        elif cards:
            if target is None:
                raise ValueError("give a target with your cards")
            task = make_task(rng, level=level, target=target, ranks=[int(c) for c in cards])
        elif target is not None or level is not None:
            task = make_task(rng, level=level, target=target, n_cards=n_cards)
        else:
            task = make_task(rng, level="year", target=CURRENT_YEAR)
        self._task = task
        self._started = time.monotonic()
        self._state = CardState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task["id"],
            level=task["level"],
            target=task["target"],
            cards=task["cards"],
            submitted=False,
            reward_mode=reward_mode,
        )
        return self._observation(done=False, reward=None)

    def step(self, action: CardAction, timeout_s: Optional[float] = None, **kwargs: Any) -> CardObservation:
        if self._task is None:
            raise RuntimeError("call reset before step")
        self._state.step_count += 1
        if self._state.submitted:
            obs = self._observation(done=True, reward=0.0)
            obs.result = {"error": "this deal was already answered; reset for a new one"}
            return obs
        self._state.submitted = True
        values = [c["rank"] for c in self._task["cards"]]
        result = grade(action.expression, values, self._task["target"], shaped=self._state.reward_mode == "shaped")
        obs = self._observation(done=True, reward=result["reward"])
        obs.result = result
        obs.solution = self._task.get("solution")
        obs.solution_steps = self._task.get("solution_steps", [])
        obs.elapsed_s = round(time.monotonic() - self._started, 2)
        return obs

    @property
    def state(self) -> CardState:
        return self._state

    def _observation(self, done: bool, reward: float | None) -> CardObservation:
        task = self._task
        cards = [Card.from_dict(c) for c in task["cards"]]
        return CardObservation(
            done=done,
            reward=reward,
            task_id=task["id"],
            level=task["level"],
            target=task["target"],
            cards=task["cards"],
            prompt=user_prompt(cards, task["target"]),
            system_prompt=SYSTEM_PROMPT,
        )

    # -- OpenEnv Task API ---------------------------------------------------

    def list_splits(self) -> list[dict]:
        return [{"name": name, "type": kind} for name, kind in SPLITS.items()]

    def list_tasks(self, split: str) -> list[dict]:
        return [public(t) for t in load_split(split)]

    def num_tasks(self, split: str) -> int:
        return len(load_split(split))

    def get_task(self, split: str, index: int) -> dict:
        return public(load_split(split)[index])

    def get_task_range(self, split: str, start: int, stop: int) -> list[dict]:
        return [public(t) for t in load_split(split)[start:stop]]
