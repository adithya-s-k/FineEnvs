"""SkyRL generator: DesignGym GUI episodes as step-wise rows, one row per image-window chunk, served by vLLM.

SkyRL's own VLM generator re-renders the whole conversation every turn and keeps only the last render's images, so
it cannot carry an image window (a drop rewrites earlier tokens) and refuses step-wise output. This generator plays
episodes with the shared core (`rl.common`: token-in/token-out, window 8,3, qwen3_coder parsing) and
emits each chunk as its own row:

* `prompt_token_ids` = the chunk's context, `response_ids` / `loss_masks` / `rollout_logprobs` = the rest (0 on
  observations), `pixel_values` / `image_grid_thw` = exactly that chunk's images;
* rows of one episode are contiguous and share its `TrajectoryID`; `is_last_step` is True on the last chunk only,
  which carries the episode reward on its last token (SkyRL computes advantages from last-step rows, grouped by
  `instance_id`, and broadcasts them to the earlier steps).

Images reach vLLM the way SkyRL's Tinker renderer does it (`skyrl/backends/renderer.py`): each new image is rendered
once alone through `/v1/chat/completions/render`, which returns its placeholder tokens, its multimodal hash and its
processed tensors (`kwargs_data`); those are spliced into `mm_features` at the image's position in our token ids.
The same `kwargs_data` decode into the trainer's `pixel_values`, so engine and trainer see identical pixels.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

import numpy as np

from skyrl.backends.renderer import decode_mm_kwargs
from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput, TrajectoryID

from rl.common.episode import Chunk, EpisodeConfig, EpisodeResult, GenOut, run_episode
from rl.common.qwen_chat import ImageItem, QwenChat

logger = logging.getLogger(__name__)


class DesignGymVLMGenerator(GeneratorInterface):
    def __init__(self, generator_cfg, inference_engine_client, tokenizer, policy_model_name: str, model_path: str):
        self.generator_cfg = generator_cfg
        self.client = inference_engine_client
        self.model = policy_model_name
        self.chat = QwenChat(model_path, tokenizer=tokenizer)
        self.env_url = os.environ.get("DESIGNGYM_URL", "http://127.0.0.1:8027")
        sp = generator_cfg.sampling_params
        self.cfg = EpisodeConfig(
            max_turns=generator_cfg.max_turns, max_new_tokens=sp.max_generate_length,
            max_context=generator_cfg.max_input_length + sp.max_generate_length,
            max_images=int(os.environ.get("DESIGNGYM_MAX_IMAGES", 8)),
            keep_images=int(os.environ.get("DESIGNGYM_KEEP_IMAGES", 3)))
        self._mm: dict[str, dict] = {}  # image digest -> {"tokens", "hash", "kwargs"}
        self._mm_lock = asyncio.Lock()

    # ------------------------------------------------------------------ images -> vLLM features

    @staticmethod
    def _digest(item: ImageItem) -> str:
        return hashlib.sha1(item.b64.encode()).hexdigest()

    async def _render(self, items: list[ImageItem]) -> None:
        """Render images that have no features yet, all in one request (one user message with N images)."""
        async with self._mm_lock:
            todo, seen = [], set()
            for it in items:
                d = self._digest(it)
                if d not in self._mm and d not in seen:
                    todo.append(it)
                    seen.add(d)
            if not todo:
                return
            body = {"json": {"model": self.model, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": it.data_url}} for it in todo]}]}}
            resp = await self.client.render_chat_completion(body)
            feats = resp.get("features") or {}
            places = feats.get("mm_placeholders", {}).get("image", [])
            hashes = feats.get("mm_hashes", {}).get("image", [])
            kwargs = (feats.get("kwargs_data") or {}).get("image", [])
            if not (len(places) == len(hashes) == len(kwargs) == len(todo)):
                raise RuntimeError(f"render returned {len(places)}/{len(hashes)}/{len(kwargs)} features for "
                                   f"{len(todo)} images")
            ids = resp["token_ids"]
            for it, pl, h, kw in zip(todo, places, hashes, kwargs, strict=True):
                self._mm[self._digest(it)] = {"tokens": ids[pl["offset"]:pl["offset"] + pl["length"]], "hash": h,
                                              "kwargs": kw}

    def _features(self, tokens: list[int], images: list[ImageItem]) -> dict | None:
        """`mm_features` for our token ids: each image's placeholder located inside its vision span."""
        if not images:
            return None
        chat = self.chat
        starts = [i for i, t in enumerate(tokens) if t == chat.vision_start]
        if len(starts) != len(images):
            raise RuntimeError(f"{len(starts)} vision spans for {len(images)} images")
        places, hashes, kwargs = [], [], []
        for s, it in zip(starts, images, strict=True):
            f = self._mm[self._digest(it)]
            ph = f["tokens"]
            span = chat.image_tokens(it)
            if tokens[s:s + len(span)] != span:
                raise RuntimeError("vision span does not match the image's token count")
            # vLLM's placeholder is the pads (or the whole span, depending on the model's prompt update)
            off = s if ph == span else s + 1
            if tokens[off:off + len(ph)] != ph:
                raise RuntimeError(f"render placeholder ({len(ph)} tokens) does not match our image tokens "
                                   f"({it.n_tokens} pads): engine and local image processors disagree")
            places.append({"offset": off, "length": len(ph)})
            hashes.append(f["hash"])
            kwargs.append(f["kwargs"])
        return {"mm_hashes": {"image": hashes}, "mm_placeholders": {"image": places}, "kwargs_data": {"image": kwargs}}

    # ------------------------------------------------------------------ engine

    def _engine(self, session_id: str, sampling_params: dict):
        async def generate(tokens: list[int], images: list[ImageItem], max_new_tokens: int) -> GenOut:
            await self._render(images)
            sp = {**sampling_params, "max_tokens": max_new_tokens, "skip_special_tokens": False}
            out = await self.client.generate({"prompt_token_ids": [tokens], "session_ids": [session_id],
                                              "sampling_params": sp,
                                              "mm_features": [self._features(tokens, images)] if images else None},
                                             model=self.model)
            ids = list(out["response_ids"][0])
            lps = out.get("response_logprobs")
            lps = list(lps[0]) if lps and lps[0] is not None else [0.0] * len(ids)
            return GenOut(ids, lps, out["stop_reasons"][0])

        return generate

    # ------------------------------------------------------------------ rows

    def _pixels(self, images: list[ImageItem]):
        mm = decode_mm_kwargs({"image": [self._mm[self._digest(it)]["kwargs"] for it in images]})
        return mm["pixel_values"], mm["image_grid_thw"]

    async def _dummy_chunk(self) -> Chunk:
        """An inert row (one tiny image, nothing trained) for an episode that failed before its first turn."""
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, "PNG")
        item = self.chat.image_item("image/png", base64.b64encode(buf.getvalue()).decode())
        await self._render([item])
        tokens = [t for seg in self.chat.opening("", [], [item], "x") for t in seg.tokens]
        return Chunk(0, tokens + [self.chat.im_end], [0] * (len(tokens) + 1), [0.0] * (len(tokens) + 1), [item],
                     len(tokens))

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        extras = input_batch.get("env_extras") or []
        tids: list[TrajectoryID] = input_batch["trajectory_ids"]
        sampling = input_batch.get("sampling_params")
        if sampling is None:
            from skyrl.backends.skyrl_train.inference_servers.engine_utils import get_vllm_sampling_params

            sampling = get_vllm_sampling_params(self.generator_cfg.sampling_params)
        tasks = [_task(e) for e in extras]
        results: list[EpisodeResult] = await asyncio.gather(*[
            run_episode(self.env_url, self.chat, task, self._engine(f"{tid.instance_id}_{tid.repetition_id}",
                                                                      dict(sampling)), self.cfg)
            for task, tid in zip(tasks, tids, strict=True)])

        out: dict[str, list] = {k: [] for k in ("prompt_token_ids", "response_ids", "rewards", "loss_masks",
                                                "stop_reasons", "rollout_logprobs", "trajectory_ids", "is_last_step",
                                                "env_metrics", "pixel_values", "image_grid_thw")}
        for res, tid in zip(results, tids, strict=True):
            chunks = res.chunks or [await self._dummy_chunk()]
            if res.status == "error":
                logger.warning("designgym episode failed (%s): %s", res.task.get("task_id"), res.error)
            for k, c in enumerate(chunks):
                last = k == len(chunks) - 1
                p = c.prompt_len
                resp = c.tokens[p:]
                reward = [0.0] * len(resp)
                if last:
                    reward[-1] = float(res.reward)
                pv, thw = self._pixels(c.images)
                out["prompt_token_ids"].append(c.tokens[:p])
                out["response_ids"].append(resp)
                out["rewards"].append(reward)
                out["loss_masks"].append(c.loss_mask[p:])
                out["rollout_logprobs"].append(c.logprobs[p:])
                out["stop_reasons"].append("stop")
                out["trajectory_ids"].append(tid)
                out["is_last_step"].append(bool(last))
                out["env_metrics"].append({**res.metrics, "chunk": k})
                out["pixel_values"].append(pv)
                out["image_grid_thw"].append(thw)
        out["rollout_metrics"] = _rollout_metrics(results)
        out["trajectory_generation_times"] = None
        out["trajectory_time_splits"] = None
        out["rollout_expert_indices"] = None
        out["rollout_sample_support"] = None
        return out  # type: ignore[return-value]


def _task(extras: dict[str, Any]) -> dict:
    import json

    t = extras.get("task")
    if isinstance(t, str):
        return json.loads(t)
    if isinstance(t, dict):
        return dict(t)
    return {k: v for k, v in extras.items() if k in ("task_id", "mode", "surface", "skin", "coordinates")}


def _rollout_metrics(results: list[EpisodeResult]) -> dict[str, float]:
    m: dict[str, float] = {}
    keys = sorted({k for r in results for k in r.metrics})
    for k in keys:
        vals = [r.metrics[k] for r in results if k in r.metrics]
        if vals:
            m[f"designgym/{k}_mean"] = float(np.mean(vals))
    m["designgym/errors"] = float(sum(r.status == "error" for r in results))
    m["designgym/policy_submitted"] = float(sum(r.status == "completed" for r in results))
    m["designgym/rows"] = float(sum(max(1, len(r.chunks)) for r in results))
    if results:
        m["designgym/reward_max"] = float(max(r.reward for r in results))
    return m

