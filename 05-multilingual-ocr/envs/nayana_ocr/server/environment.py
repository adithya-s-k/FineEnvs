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
    if os.environ.get("NAYANA_CORPUS_MANIFEST"):
        from ..data.corpus import CorpusCatalog

        return CorpusCatalog(
            directory,
            os.environ.get("NAYANA_CACHE_DIR", "/tmp/nayana-cache"),
            source_root=os.environ.get("NAYANA_SOURCE_ROOT"),
            local_source=os.environ.get("NAYANA_LOCAL_SOURCE") == "true",
            index_cache_bytes=int(
                os.environ.get("NAYANA_INDEX_CACHE_BYTES", "4000000000")
            ),
            group_cache_bytes=int(
                os.environ.get("NAYANA_GROUP_CACHE_BYTES", "4000000000")
            ),
            asset_cache_bytes=int(
                os.environ.get("NAYANA_ASSET_CACHE_BYTES", "512000000")
            ),
            max_group_bytes=int(os.environ.get("NAYANA_MAX_GROUP_BYTES", "512000000")),
            prefetch_workers=int(os.environ.get("NAYANA_PREFETCH_WORKERS", "2")),
            prefetch_pending=int(os.environ.get("NAYANA_PREFETCH_PENDING", "4")),
        )
    return Catalog(directory)


def configured_catalog():
    corpus = os.environ.get("NAYANA_CORPUS_MANIFEST")
    if corpus:
        return get_catalog(corpus)
    directory = os.environ.get("NAYANA_SNAPSHOT")
    if not directory:
        raise RuntimeError(
            "Set NAYANA_CORPUS_MANIFEST to a ready corpus index or NAYANA_SNAPSHOT to a prepared directory"
        )
    return get_catalog(os.path.realpath(directory))


class NayanaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, catalog=None, judge=None):
        super().__init__()
        self.catalog = catalog if catalog is not None else configured_catalog()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None
        self.judge = judge

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
        self._task = self.catalog.materialize(task)
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
        family = self._task["family"]
        policy_id = ""
        if family == "descriptive_vqa":
            from .judge import configured_judge

            judge = self.judge or configured_judge()
            reward, metrics = judge.score(self._task, action.answer)
            policy_id = judge.policy_id
        elif family == "layout_detection":
            from .layout import POLICY, score_layout

            reward, metrics = score_layout(
                action.answer,
                self._task["reference"],
                self._task["width"],
                self._task["height"],
            )
            policy_id = POLICY
        else:
            reward, metrics = score(family, action.answer, self._task["reference"])
        self._state.step_count = 1
        return self._observation(
            done=True, reward=reward, metrics=metrics, grading_policy_id=policy_id
        )

    @property
    def state(self):
        return self._state

    def close(self):
        self._task = None
