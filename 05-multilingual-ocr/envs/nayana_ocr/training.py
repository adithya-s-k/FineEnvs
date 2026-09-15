"""CPU-testable TRL adapter. Only immutable task IDs go through the sampler."""

import hashlib
import io
import math
import random
import re
import threading
from collections import OrderedDict, defaultdict

import requests
from PIL import Image

from .client import connect
from .models import NayanaAction


class AssetCache:
    def __init__(self, url, max_bytes=32_000_000):
        if max_bytes < 1:
            raise ValueError("Asset cache budget must be positive")
        self.url = url.rstrip("/")
        self.max_bytes = max_bytes
        self._bytes = 0
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self.downloads = 0

    def image(self, observation):
        sha = observation.asset_sha256
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid asset hash")
        with self._lock:
            raw = self._items.pop(sha, None)
            if raw is None:
                # Do not follow an arbitrary observation URL or cache unverified content.
                with requests.get(
                    f"{self.url}/assets/{sha}",
                    params={"task_id": observation.task_id},
                    timeout=120,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_content(65536):
                        content.extend(chunk)
                        if len(content) > self.max_bytes:
                            raise ValueError(
                                "Single asset exceeds the trainer byte budget"
                            )
                raw = bytes(content)
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise ValueError("Asset hash mismatch")
                while self._items and self._bytes + len(raw) > self.max_bytes:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= len(old)
                self._bytes += len(raw)
                self.downloads += 1
            self._items[sha] = raw
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 50_000_000:
                raise ValueError("Image exceeds the trainer pixel budget")
            return image.convert("RGB")


class TrainingEnvironment:
    # TRL exposes public methods as tools; only reset is needed for this one-step task.
    def __init__(self, url, cache, snapshot_id):
        self.client = connect(url)
        self.cache = cache
        self.snapshot_id = snapshot_id
        self.task_id = None

    def reset(self, task_id, **kwargs):
        observation = self.client.reset(task_id=task_id).observation
        if (
            observation.task_id != task_id
            or observation.snapshot_id != self.snapshot_id
        ):
            raise RuntimeError("Rollout task or snapshot changed")
        self.task_id = task_id
        return [
            {"type": "image", "image": self.cache.image(observation)},
            {"type": "text", "text": observation.prompt},
        ]

    def _close(self):
        self.client.close()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    return (
        content
        if isinstance(content, str)
        else "".join(b.get("text", "") for b in content if b.get("type") == "text")
    )


def env_reward(completions, environments, task_id, **kwargs):
    scores = []
    metrics = defaultdict(list)
    for completion, environment, expected in zip(
        completions, environments, task_id, strict=True
    ):
        if environment.task_id != expected:
            raise RuntimeError("Reward was routed to the wrong rollout task")
        result = environment.client.step(
            NayanaAction(answer=completion_text(completion))
        )
        if (
            not result.done
            or result.reward is None
            or not math.isfinite(float(result.reward))
        ):
            raise RuntimeError("Invalid terminal reward")
        scores.append(float(result.reward))
        observation = result.observation
        prefix = f"nayana/{observation.language}/{observation.family}"
        for key, value in {"reward": result.reward, **observation.metrics}.items():
            metrics[f"{prefix}/{key}"].append(float(value))
    if kwargs.get("log_metric") is not None:
        for name, values in metrics.items():
            kwargs["log_metric"](name, sum(values) / len(values))
    return scores


def task_rows(url, split, languages=None, families=None):
    with connect(url) as client:
        count = client.num_tasks(split)
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            for task in client.get_task_range(split, start, stop):
                if languages and task["language"] not in languages:
                    continue
                if families and task["family"] not in families:
                    continue
                yield {
                    "task_id": task["task_id"],
                    "language": task["language"],
                    "family": task["family"],
                }


def balanced_rows(rows, languages, families, seed=42, per_group=64):
    """Downsample to the smallest available language/family group, then interleave."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["family"])].append(row)
    keys = [(language, family) for language in languages for family in families]
    missing = [key for key in keys if not groups[key]]
    if missing:
        raise ValueError(
            f"Missing language/task groups: {missing}. Prepare more documents for this split"
        )
    count = min(per_group, *(len(groups[key]) for key in keys))
    if count < 1:
        raise ValueError("per_group must be positive")
    rng = random.Random(seed)
    for key in keys:
        groups[key].sort(key=lambda row: row["task_id"])
        rng.shuffle(groups[key])
    return [groups[key][i] for i in range(count) for key in keys]
