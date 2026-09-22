import os
import random
from functools import lru_cache
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..data.catalog import SPLITS, Catalog
from ..models import AsrAction, AsrObservation
from .rewards import GRADING_POLICY, score


@lru_cache(maxsize=4)
def get_catalog(directory):
    return Catalog(directory)


def configured_catalog():
    directory = os.environ.get("ASR_SNAPSHOT")
    if not directory:
        raise RuntimeError("Set ASR_SNAPSHOT to a prepared snapshot directory")
    return get_catalog(os.path.realpath(directory))


class AsrEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, catalog=None):
        super().__init__()
        self.catalog = catalog if catalog is not None else configured_catalog()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None

    def list_splits(self):
        return list(SPLITS)

    def num_tasks(self, split):
        return self.catalog.count(split)

    def get_task(self, split, index):
        return {**self.catalog.public(self.catalog.at(split, index)), "index": index}

    def get_task_range(self, split, start=None, stop=None):
        return self.catalog.task_range(split, start, stop)

    def list_tasks(self, split):
        return self.get_task_range(split)

    def reset(self, task_id=None, split=None, index=None, seed=None, episode_id=None):
        # A failed reset must not leave the previous answer gradable.
        self._task = None
        if task_id is not None:
            if split is not None or index is not None:
                raise ValueError("Select by task_id OR split/index")
            task = self.catalog.get(task_id)
        else:
            split = split or "train"
            count = self.num_tasks(split)
            if count == 0:
                raise ValueError(f"Snapshot has no {split} tasks")
            index = random.Random(seed).randrange(count) if index is None else index
            task = self.catalog.at(split, index)
        self._task = task
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._observation(task)

    def _observation(self, task, reward=None, metrics=None, done=False):
        public = self.catalog.public(task)
        return AsrObservation(
            done=done,
            reward=reward,
            metrics=metrics or {},
            grading_policy_id=GRADING_POLICY,
            **{
                key: public[key]
                for key in (
                    "task_id",
                    "snapshot_id",
                    "split",
                    "language",
                    "language_name",
                    "family",
                    "prompt",
                    "asset_path",
                    "asset_sha256",
                    "mime",
                    "sampling_rate",
                    "num_samples",
                    "duration_seconds",
                    "error_unit",
                )
            },
        )

    def step(self, action: AsrAction):
        if self._task is None:
            raise ValueError("Call reset before step")
        task = self._task
        # One answer per episode: a second step must not re-grade a consumed task.
        self._task = None
        self._state.step_count += 1
        reward, metrics = score(task, action.transcript)
        return self._observation(task, reward=reward, metrics=metrics, done=True)

    @property
    def state(self):
        return self._state
