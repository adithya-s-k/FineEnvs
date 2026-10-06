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
    """Verified, size-bounded cache of the utterances a run fetches.

    Audio is returned exactly as served. Padding used to happen here, on the waveform,
    so every clip in a batch shared one feature shape. That was a mistake. The feature
    extractor cannot tell padded zeros from speech, so it marked all 30 seconds as valid,
    and the model heard about 18 seconds of silence after a typical clip. That never
    happens in evaluation, where vLLM sees the bare clip. On the same held-out clips the
    trainer's harness scored 0.43 against vLLM's 0.50. Padding now happens on the
    features (see pad_audio_features), where the mask marks it.
    """

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
                # Do not follow an arbitrary observation URL or cache unverified
                # content. The task id is passed as a parameter because the indexed
                # corpus needs it to find the row group holding this audio; a prepared
                # snapshot addresses by hash alone and ignores it.
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


def pad_audio_features(processor, seconds):
    """Make the processor pad every clip's features to one span, with padding masked.

    TRL left-pads input_ids but stacks every other multimodal field with
    torch.tensor(np.array(v)), which needs one shape across the batch, and audio
    features have one frame per 10 ms. Padding to a fixed number of samples inside the
    extractor gives one shape and an input_features_mask that is False on the padded
    frames. The processor then inserts audio tokens only for the real frames, and the
    model drops the masked encoder outputs. So a clip goes in exactly as it does unpadded,
    the way vLLM serves it.

    The extractor is given a subclass of its own class with the same name, so a saved
    processor config is unchanged. Returns the processor, for chaining.
    """
    if seconds <= 0:
        return processor
    extractor = processor.feature_extractor
    base = type(extractor)
    if getattr(base, "_padded_samples", None):
        return processor
    samples = int(seconds * SAMPLING_RATE)
    multiple = getattr(extractor, "pad_to_multiple_of", None) or 128
    # A span that is not a multiple of the extractor's pad unit would be rounded past it.
    samples = -(-samples // multiple) * multiple

    def call(self, raw_speech, *args, **kwargs):
        kwargs.update(padding="max_length", max_length=samples, truncation=True)
        return base.__call__(self, raw_speech, *args, **kwargs)

    extractor.__class__ = type(
        base.__name__, (base,), {"__call__": call, "_padded_samples": samples}
    )
    return processor


AUDIO_FIELDS = ("input_features", "input_features_mask")


def audio_rows(fields, device=None):
    """The audio fields of one tokenized batch as tensors, one row per prompt."""
    import numpy
    import torch

    rows = {}
    for key in AUDIO_FIELDS:
        value = fields.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(numpy.stack([numpy.asarray(v) for v in value]))
        rows[key] = value.to(device) if device is not None else value
    if rows and len(rows) != len(AUDIO_FIELDS):
        raise RuntimeError(f"Expected both {AUDIO_FIELDS}, got {sorted(rows)}")
    return rows


class audio_forward:
    """Feed one batch's audio to every forward pass of `model` made inside the block.

    A forward pass that already carries features, as generation does, is left alone.
    Otherwise the next rows of the batch are attached, in order. TRL scores a batch in
    consecutive chunks, so a cursor is enough to keep each chunk on its own clips. On
    exit, every row must have been used exactly once.
    """

    def __init__(self, model, rows):
        self.model, self.rows, self.cursor, self.handle = model, rows, 0, None

    def _inject(self, module, args, kwargs):
        if kwargs.get("input_features") is not None:
            return None
        ids = kwargs.get("input_ids")
        if ids is None:
            raise RuntimeError("Audio can only be attached to a forward over input_ids")
        start, stop = self.cursor, self.cursor + ids.size(0)
        total = self.rows["input_features"].size(0)
        if stop > total:
            raise RuntimeError(f"Forward wants rows {start}:{stop} of {total}")
        self.cursor = stop
        kwargs.update({k: v[start:stop] for k, v in self.rows.items()})
        return args, kwargs

    def __enter__(self):
        if self.rows:
            self.handle = self.model.register_forward_pre_hook(
                self._inject, with_kwargs=True
            )
        return self

    def __exit__(self, kind, error, trace):
        if self.handle is not None:
            self.handle.remove()
            total = self.rows["input_features"].size(0)
            if error is None and self.cursor != total:
                raise RuntimeError(
                    f"Audio rows misaligned: forwards used {self.cursor} of {total}"
                )
        return False


def audio_grpo_trainer():
    """GRPOTrainer that lets the clip reach the forward pass the loss is taken from.

    TRL 1.13 builds the loss inputs from images only. Audio features from the prompt
    are used to generate and then dropped, so the trainer scored each transcript against
    a prompt whose audio tokens were empty. The policy gradient therefore raised
    p(transcript | no audio), which teaches a language model the training sentences and
    not how to listen. Training reward rose while held-out ASR barely moved. Images
    take a separate path and were never affected, which is why OCR could adapt its
    vision tower.

    This subclass stashes the batch's audio rows when the prompts are tokenized and
    carries them in the generation batch. TRL's shuffle and split then keep them
    aligned with their rows. Each log-prob forward gets the matching clips through
    audio_forward. The first batch also checks that the audio reaches the loss: with
    the clip, the sampled transcripts must be clearly more likely than without it.
    """
    import torch
    from trl import GRPOTrainer

    class AudioGRPOTrainer(GRPOTrainer):
        # Mean per-token log-prob the clip must add to its own transcripts.
        MIN_AUDIO_GAIN = 0.5

        _pending_audio = None
        _logp_audio = None
        audio_check = None

        def _tokenize_prompts(self, prompts):
            prompt_ids, images, fields = super()._tokenize_prompts(prompts)
            self._pending_audio = audio_rows(fields, self.accelerator.device)
            # Old and reference log-probs are computed before the batch is returned.
            self._logp_audio = self._pending_audio or None
            return prompt_ids, images, fields

        def _generate_and_score_completions(self, inputs):
            self._pending_audio = None
            try:
                output = super()._generate_and_score_completions(inputs)
            finally:
                self._logp_audio = None
            audio = self._pending_audio or {}
            rows = output["prompt_ids"].size(0)
            for key, value in audio.items():
                if value.size(0) != rows:
                    raise RuntimeError(f"{key} has {value.size(0)} rows for {rows} prompts")
                output[key] = value
            if audio and self.audio_check is None and self.model.training:
                self.audio_check = self._check_audio(output)
            return output

        def _compute_loss(self, model, inputs):
            rows = {k: inputs[k] for k in AUDIO_FIELDS if inputs.get(k) is not None}
            self._logp_audio = rows or None
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._logp_audio = None

        def _get_per_token_logps_and_entropies(self, model, input_ids, *args, **kwargs):
            rows = self._logp_audio
            if not rows:
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, *args, **kwargs
                )
            if rows["input_features"].size(0) != input_ids.size(0):
                raise RuntimeError("Audio rows do not match the batch being scored")
            with audio_forward(model, rows):
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, *args, **kwargs
                )

        def _check_audio(self, output, limit=4):
            ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)[:limit]
            mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)[:limit]
            keep = output["completion_ids"].size(1)
            tokens = output["completion_mask"][:limit].float()
            rows = {k: output[k][:limit] for k in AUDIO_FIELDS}
            with torch.no_grad():
                self._logp_audio = rows
                try:
                    heard, _, _ = self._get_per_token_logps_and_entropies(
                        self.model, ids, mask, keep
                    )
                finally:
                    self._logp_audio = None
                deaf, _, _ = self._get_per_token_logps_and_entropies(
                    self.model, ids, mask, keep
                )
            count = tokens.sum().clamp(min=1)
            result = {
                "with_audio": float((heard * tokens).sum() / count),
                "without_audio": float((deaf * tokens).sum() / count),
            }
            result["gain"] = result["with_audio"] - result["without_audio"]
            print(f"audio reaches the loss: {result}", flush=True)
            if result["gain"] < self.MIN_AUDIO_GAIN:
                raise RuntimeError(
                    f"The clip adds only {result['gain']:.3f} nats per token to its own "
                    "transcripts, so the loss is not hearing the audio"
                )
            return result

    return AudioGRPOTrainer


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
READY = "ready.json"


def mark_checkpoint_ready(directory, step):
    """Record a saved checkpoint's adapter by size and hash, for a watcher to verify."""
    import json
    from pathlib import Path

    directory = Path(directory)
    files = {}
    for name in ADAPTER_FILES:
        digest = hashlib.sha256()
        with (directory / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        files[name] = {"size": (directory / name).stat().st_size, "sha256": digest.hexdigest()}
    (directory / READY).write_text(json.dumps({"step": step, "files": files}) + "\n")
    return files


def checkpoint_complete(directory, ready):
    """True when every adapter file in `directory` matches its ready.json record."""
    from pathlib import Path

    directory = Path(directory)
    for name, expected in ready["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            return False
    return True


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


def task_rows(url, split):
    """Every task in a split. Only sound for the small frozen evaluation splits."""
    with connect(url) as client:
        count = client.num_tasks(split)
        rows = []
        for start in range(0, count, 256):
            rows.extend(client.get_task_range(split, start, min(start + 256, count)))
    return rows


POSITIONS_PER_REQUEST = 1000


def sampled_rows(url, split, languages, families, seed, per_group):
    """Equal counts per language/family, so a short run is not one language's score.

    The corpus train split holds hundreds of thousands of tasks, so paging it to keep a
    handful per language is not an option. Each group's size comes from the index and
    only the drawn positions are fetched. Positions are the index's own per-family order,
    which is fixed for a snapshot, so a seed reproduces the selection.
    """
    rng = random.Random(seed)
    selected, missing = [], []
    with connect(url) as client:
        for lang in languages:
            for fam in families:
                count = client.num_group_tasks(split, lang, fam)
                if not count:
                    missing.append((lang, fam))
                    continue
                drawn = sorted(rng.sample(range(count), min(per_group, count)))
                # The server answers at most 1000 positions per request. A run that asked
                # for all 2,282 Kannada training clips in one call died at startup with a
                # 400, so the draw is fetched in pieces the endpoint accepts.
                for start in range(0, len(drawn), POSITIONS_PER_REQUEST):
                    piece = drawn[start : start + POSITIONS_PER_REQUEST]
                    selected.extend(client.get_group_tasks(split, lang, fam, piece))
    if missing:
        raise ValueError(f"Missing language/task groups: {missing}")
    rng.shuffle(selected)
    return selected
