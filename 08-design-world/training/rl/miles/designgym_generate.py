"""Miles custom generate function: one DesignGym GUI episode per sample, screenshots served by SGLang.

    --custom-generate-function-path rl.miles.designgym_generate.generate

The episode loop, image window and tool parsing are the shared core's (`rl.common`). This adapter only
talks to SGLang (`/generate` with `input_ids` whose image pads are already expanded, `image_data` in the same order,
`return_logprob`, the LoRA adapter when LoRA rollout is on) and turns the episode's chunks into Miles `Sample`s:

* one Sample per image-window chunk: `tokens` = chunk context + response, `loss_mask` / `rollout_log_probs` over the
  response (0 on observations), `multimodal_train_inputs` = the image processor's `pixel_values` / `image_grid_thw`
  for exactly the images in that chunk;
* every chunk of an episode shares `rollout_id` (= the input sample's index) and the episode's reward
  (`train_data_conversion.py` requires one reward per rollout), so GRPO normalises once per episode;
* the reward is set here, so Miles' reward model is skipped (no --rm-type needed);
* every episode returns exactly `--designgym-samples-per-episode` samples (default: the window's chunk bound for
  `--designgym-max-turns`): the episode stops (and is submitted) rather than open one chunk more, and short episodes
  are padded with inert samples (`remove_sample`, same rollout_id and reward, one tiny image). Miles skips its
  sample trimming for multi-sample rollouts and its rollout-side scheduler for multimodal data, and the trainer
  then runs floor(samples / global batch) steps while the rewards list keeps every sample, so the sample count per
  rollout must be fixed and a multiple of --global-batch-size (rollout batch x group x samples per episode).

The task (reset kwargs: task_id, mode, surface, skin, coordinates) is the dataset row's metadata
(`--input-key prompt --metadata-key metadata`).
"""

from __future__ import annotations

import argparse
import logging
import time
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.types import Sample

from rl.common.episode import EpisodeConfig, EpisodeResult, GenOut, run_episode
from rl.common.qwen_chat import QwenChat

logger = logging.getLogger(__name__)
_CHAT: QwenChat | None = None


def _chat(input: GenerateFnInput) -> QwenChat:
    global _CHAT
    if _CHAT is None:
        proc = input.state.processor
        _CHAT = QwenChat(input.args.hf_checkpoint, tokenizer=input.state.tokenizer,
                         image_processor=getattr(proc, "image_processor", None))
    return _CHAT


def _config(args) -> EpisodeConfig:
    cfg = EpisodeConfig(max_turns=args.designgym_max_turns, max_new_tokens=args.rollout_max_response_len,
                        max_context=args.rollout_max_context_len or args.designgym_max_context,
                        max_images=args.designgym_max_images, keep_images=args.designgym_keep_images)
    cfg.max_chunks = args.designgym_samples_per_episode or cfg.chunk_bound()
    return cfg


def _engine(input: GenerateFnInput):
    args = input.args
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    base = {k: v for k, v in input.sampling_params.items() if k != "max_new_tokens"}

    async def generate(tokens, images, max_new_tokens) -> GenOut:
        payload = {"input_ids": tokens, "sampling_params": {**base, "max_new_tokens": max_new_tokens},
                   "return_logprob": True}
        if images:
            payload["image_data"] = [im.data_url for im in images]
        if lora_rollout_enabled(args):
            payload["lora_path"] = LORA_ADAPTER_NAME
        out = await post(url, payload)
        meta = out["meta_info"]
        pairs = meta.get("output_token_logprobs") or []
        return GenOut([p[1] for p in pairs], [float(p[0]) for p in pairs], meta["finish_reason"]["type"])

    return generate


def _train_inputs(chat: QwenChat, images) -> dict:
    out = chat.image_processor(images=[im.pil for im in images], return_tensors="pt")
    return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}


def _inert(src: Sample, chat: QwenChat, rollout_id: int, reward: float, metadata: dict) -> Sample:
    """A sample that trains nothing (remove_sample, loss mask 0) but runs the vision tower on one tiny image."""
    from PIL import Image

    s = deepcopy(src)
    item = chat.image_item("image/png", _png_b64(Image.new("RGB", (64, 64), (255, 255, 255))))
    s.tokens = [t for seg in chat.opening("", [], [item], "x") for t in seg.tokens] + [chat.im_end]
    s.response_length = 1
    s.loss_mask = [0]
    s.rollout_log_probs = [0.0]
    s.response = ""
    s.multimodal_train_inputs = _train_inputs(chat, [item])
    s.reward = reward
    s.rollout_id = rollout_id
    s.remove_sample = True
    s.status = Sample.Status.COMPLETED
    s.metadata = {**metadata, "designgym_inert": 1}
    return s


def _samples(input: GenerateFnInput, chat: QwenChat, res: EpisodeResult, n: int) -> list[Sample]:
    src = input.sample
    rollout_id = src.index if src.rollout_id is None else src.rollout_id
    meta = {**(src.metadata or {}), **{f"designgym/{k}": v for k, v in res.metrics.items()},
            "designgym_chunks": len(res.chunks), "designgym_status": res.status, "designgym_error": res.error}
    out = []
    for chunk in res.chunks[:n]:
        s = deepcopy(src)
        p = chunk.prompt_len
        s.tokens = list(chunk.tokens)
        s.response_length = len(chunk.tokens) - p
        s.loss_mask = list(chunk.loss_mask[p:])
        s.rollout_log_probs = list(chunk.logprobs[p:])
        s.response = chat.decode(chunk.tokens[p:])[-2000:]
        s.multimodal_train_inputs = _train_inputs(chat, chunk.images)
        s.reward = res.reward
        s.rollout_id = rollout_id
        s.status = Sample.Status.COMPLETED
        s.metadata = {**meta, "designgym_chunk": chunk.index}
        out.append(s)
    while len(out) < n:  # short (or failed) episodes: inert padding keeps the count fixed
        out.append(_inert(src, chat, rollout_id, res.reward, meta))
    return out


def _png_b64(im) -> str:
    import base64
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    chat = _chat(input)
    task = dict(input.sample.metadata or {})
    t0 = time.time()
    res = await run_episode(args.designgym_url, chat, task, _engine(input), _config(args))
    if res.status == "error":
        logger.warning("designgym episode failed (%s): %s", task.get("task_id"), res.error)
    else:
        logger.info("designgym %s reward=%.3f status=%s turns=%d chunks=%d %.0fs", task.get("task_id"), res.reward,
                    res.status, res.metrics.get("turns", 0), len(res.chunks), time.time() - t0)
    return GenerateFnOutput(samples=_samples(input, chat, res, _config(args).max_chunks))


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--designgym-url", type=str, default="http://127.0.0.1:8027")
    parser.add_argument("--designgym-max-turns", type=int, default=24)
    parser.add_argument("--designgym-max-images", type=int, default=8)
    parser.add_argument("--designgym-keep-images", type=int, default=3)
    parser.add_argument("--designgym-max-context", type=int, default=32768)
    parser.add_argument("--designgym-samples-per-episode", type=int, default=0,
                        help="samples (chunks) each episode returns; 0: the window's chunk bound for max turns")


generate.add_arguments = _add_arguments
