import os
import random
from functools import lru_cache
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..data.catalog import SPLITS, Catalog
from ..models import NayanaAction, NayanaObservation
from .rewards import score


@lru_cache(maxsize=4)
def get_catalog(directory):
    return Catalog(directory)


def configured_catalog():
    directory = os.environ.get("NAYANA_SNAPSHOT")
    if not directory:
        raise RuntimeError(
            "Set NAYANA_SNAPSHOT to a finalized nayana-prepare directory"
        )
    return get_catalog(os.path.realpath(directory))


class NayanaEnvironment(Environment):
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
        self._task = None  # A failed reset cannot leave the previous answer gradable.
        if task_id is not None:
            if split is not None or index is not None:
                raise ValueError("Select by task_id OR split/index")
            task = self.catalog.get(task_id)
        else:
            split = split or "train"
            count = self.num_tasks(split)
            if count == 0:
                raise ValueError(f"Prepared snapshot has no {split} tasks")
            index = random.Random(seed).randrange(count) if index is None else index
            task = self.catalog.at(split, index)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task = task
        return self._observation()

    def _observation(self, **kwargs):
        fields = NayanaObservation.model_fields
        return NayanaObservation(
            **{k: v for k, v in self.catalog.public(self._task).items() if k in fields},
            **kwargs,
        )

    def step(self, action: NayanaAction, timeout_s=None):
        if self._task is None or self._state.step_count:
            raise RuntimeError("Reset before submitting a single answer")
        reward, metrics = score(
            self._task["family"], action.answer, self._task["reference"]
        )
        self._state.step_count = 1
        return self._observation(done=True, reward=reward, metrics=metrics)

    @property
    def state(self):
        return self._state

    def close(self):
        self._task = None
