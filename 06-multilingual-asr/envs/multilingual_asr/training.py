"""CPU-testable TRL adapter. Only immutable task IDs go through the sampler."""

import hashlib
import io
import math
import random
import re
import threading
from collections import OrderedDict, defaultdict

import requests

from .client import connect
from .models import AsrAction

SAMPLING_RATE = 16_000


def decode_audio(raw, expected_rate=SAMPLING_RATE):
    """Decode an utterance to a mono float array in [-1, 1].

    FLEURS publishes 16 kHz mono **IEEE float** WAV, not PCM. An earlier version decoded
    with the standard library, which rejects that subtype outright, and the fixtures were
    PCM so the tests passed while real audio failed on the GPU. soundfile reads every
    subtype the corpus uses, and the rate and channel count are still checked here rather
    than left to reach the feature extractor as silent nonsense.
    """
    import soundfile

    audio, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if rate != expected_rate:
        raise ValueError(f"Expected {expected_rate} Hz audio, got {rate}")
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio, got {audio.shape[1]} channels")
    return audio[:, 0]


class AssetCache:
    def __init__(self, url, max_bytes=64_000_000):
        if max_bytes < 1:
            raise ValueError("Asset cache budget must be positive")
        self.url = url.rstrip("/")
        self.max_bytes = max_bytes
        self._bytes = 0
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self.downloads = 0

    def audio(self, observation):
        sha = observation.asset_sha256
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid asset hash")
        with self._lock:
            raw = self._items.pop(sha, None)
            if raw is None:
                # Do not follow an arbitrary observation URL or cache unverified content.
                with requests.get(
                    f"{self.url}/assets/{sha}", timeout=120, stream=True
                ) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_content(65536):
                        content.extend(chunk)
                        if len(content) > self.max_bytes:
                            raise ValueError("Single asset exceeds the trainer budget")
                raw = bytes(content)
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise ValueError("Asset hash mismatch")
                while self._items and self._bytes + len(raw) > self.max_bytes:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= len(old)
                self._bytes += len(raw)
                self.downloads += 1
            self._items[sha] = raw
        return decode_audio(raw)


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
            {"type": "audio", "audio": self.cache.audio(observation)},
            {"type": "text", "text": observation.prompt},
        ]

    def _close(self):
        self.client.close()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"] if isinstance(completion, list) else completion
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
            AsrAction(transcript=completion_text(completion))
        )
        if (
            not result.done
            or result.reward is None
            or not math.isfinite(float(result.reward))
        ):
            raise RuntimeError("Invalid terminal reward")
        scores.append(float(result.reward))
        observation = result.observation
        prefix = f"asr/{observation.language}/{observation.family}"
        for key, value in {"reward": result.reward, **observation.metrics}.items():
            metrics[f"{prefix}/{key}"].append(float(value))
    if kwargs.get("log_metric") is not None:
        for name, values in metrics.items():
            kwargs["log_metric"](name, sum(values) / len(values))
    return scores


def task_rows(url, split, languages=None, families=None):
    with connect(url) as client:
        count = client.num_tasks(split)
        rows = []
        for start in range(0, count, 256):
            for task in client.get_task_range(split, start, min(start + 256, count)):
                if languages and task["language"] not in languages:
                    continue
                if families and task["family"] not in families:
                    continue
                rows.append(task)
    return rows


def balanced_rows(rows, languages, families, seed, per_group):
    """Equal counts per language/family, so a short run is not one language's score."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"], row["family"]].append(row)
    missing = [
        (lang, fam) for lang in languages for fam in families if not groups[lang, fam]
    ]
    if missing:
        raise ValueError(f"Missing language/task groups: {missing}")
    rng = random.Random(seed)
    selected = []
    for lang in languages:
        for fam in families:
            pool = sorted(groups[lang, fam], key=lambda r: r["task_id"])
            selected.extend(rng.sample(pool, min(per_group, len(pool))))
    rng.shuffle(selected)
    return selected
