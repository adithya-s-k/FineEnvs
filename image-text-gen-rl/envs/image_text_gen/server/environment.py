import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..data.catalog import SPLITS, configured_catalog
from ..models import ImageTextGenAction, ImageTextGenObservation
from .audit import configured_audit
from .images import InvalidImage, prepare
from .scoring import POLICY, combine

PUBLIC_READING_FIELDS = (
    "model", "provider", "text", "truncated", "cached", "latency_s",
    "text_accuracy", "malformed_glyphs", "extra_chars", "prompt_text_chars", "span",
)


class ImageTextGenEnvironment(Environment):
    """One prompt, one submitted image, one reward.

    Discovery methods implement OpenEnv's TaskProvider protocol, so tasks can be listed
    and replayed by ID through /image_text_gen/splits, /num_tasks, /task and /task_range.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, catalog=None, verifier=None, audit=None):
        super().__init__()
        self.catalog = catalog if catalog is not None else configured_catalog()
        self.verifier = verifier
        self.audit = audit if audit is not None else configured_audit()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None

    def list_splits(self):
        return self.catalog.splits()

    def num_tasks(self, split):
        return self.catalog.count(split)

    def get_task(self, split, index):
        return {**self.catalog.public(self.catalog.at(split, index)), "index": index}

    def get_task_range(self, split, start=None, stop=None):
        return self.catalog.task_range(split, start, stop)

    def list_tasks(self, split):
        return self.get_task_range(split, 0, min(1000, self.num_tasks(split)))

    def reset(self, task_id=None, split=None, index=None, seed=None, episode_id=None):
        self._task = None  # A failed reset cannot leave the previous task gradable.
        if task_id is not None:
            if split is not None or index is not None:
                raise ValueError("Select by task_id OR split/index")
            task = self.catalog.get(task_id)
        else:
            split = split or "train"
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r}")
            count = self.num_tasks(split)
            if count == 0:
                raise ValueError(f"No {split} tasks")
            index = random.Random(seed).randrange(count) if index is None else index
            task = self.catalog.at(split, index)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task = task
        return self._observation()

    def _observation(self, **kwargs):
        return ImageTextGenObservation(**self.catalog.public(self._task), **kwargs)

    def _verifier(self):
        if self.verifier is None:
            from .verifier import configured_verifier

            self.verifier = configured_verifier()
        return self.verifier

    def step(self, action: ImageTextGenAction, timeout_s=None):
        if self._task is None or self._state.step_count:
            raise RuntimeError("Reset before submitting a single image")
        verifier = self._verifier()
        try:
            png, info = prepare(action.image)
        except InvalidImage as error:
            # A malformed submission is the policy's fault: it scores 0 and ends the episode.
            self._state.step_count = 1
            return self._observation(
                done=True, reward=0.0,
                metrics={"invalid_image": True},
                metadata={"error": str(error)},
                grading_policy_id=verifier.policy_id,
            )
        if info["blank"]:
            readings = [{"model": "blank-image", "text": "", "cached": False}]
            reward, metrics, scored = combine(self._task["target_text"], [""], self._task["prompt"])
        else:
            # VerifierUnavailable propagates: no reward, and step_count stays 0 so the
            # caller can retry this step without resetting.
            readings = verifier.transcribe(png, info)
            reward, metrics, scored = combine(
                self._task["target_text"], [r["text"] for r in readings], self._task["prompt"]
            )
        self._state.step_count = 1
        transcriptions = [
            {key: value for key, value in {**reading, **detail}.items()
             if key in PUBLIC_READING_FIELDS}
            for reading, detail in zip(readings, scored, strict=True)
        ]
        metrics.update(
            invalid_image=False,
            blank_image=info["blank"],
            verifier_width=info["verifier_width"],
            verifier_height=info["verifier_height"],
        )
        if self.audit is not None:
            self.audit.record(
                self._task, reward, metrics, transcriptions, info,
                verifier.policy_id, POLICY["name"], png=None if info["blank"] else png,
            )
        return self._observation(
            done=True,
            reward=reward,
            metrics=metrics,
            transcriptions=transcriptions,
            metadata={"image_sha256": info["image_sha256"], "scoring": POLICY["name"]},
            grading_policy_id=verifier.policy_id,
        )

    @property
    def state(self):
        return self._state

    def close(self):
        self._task = None
