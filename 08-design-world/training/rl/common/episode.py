"""One DesignGym GUI episode played by a policy served by any engine, as trainable token chunks.

The loop is `envs/designgym/rollout.py`'s (the evaluation harness), so a policy trained here is evaluated on the
prompt layout it learned:

* reset with the task (max_steps = max_turns, so the instructions state the action budget), `start_task` hidden;
* the opening user turn is the target image, then the editor screenshot, then the intro text;
* each assistant turn's tool calls run in order; their text goes back as tool responses and every image they
  returned follows in one user turn ("Image returned by <tool>."); without a `screenshot` call, a fresh screenshot is
  attached to the last call, as any computer-use harness does;
* a turn without tool calls gets the nudge (at most twice, then the episode ends);
* the episode ends when a tool reports done (submit / Export); otherwise it is submitted for it at the turn cap;
* the reward is the environment's grade total (reward v3 on the server); it is never computed here.

Image window (`--image-window 8,3`): the opening images never drop; once more than `max_images` screenshots are in
the conversation, all but the newest `keep_images` become the placeholder text. A drop rewrites the prompt, so the
current chunk closes and the next one starts from the windowed history: one chunk is one training sample whose
context is exactly what the engine saw, with loss on the tokens sampled inside it. Every chunk of an episode carries
the episode's reward.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .env_client import EnvSession, ToolOutcome
from .qwen_chat import ImageItem, QwenChat, Seg, parse_tool_calls

NUDGE = ("Continue working with the tools. When the design matches, call the submit tool; "
         "do not stop without submitting.")
INTRO = {"reference": "This is the target design. Recreate it.",
         "description": "Build the design from the brief in your instructions."}
HIDDEN_TOOLS = ("start_task",)


@dataclass
class EpisodeConfig:
    max_turns: int = 24
    max_new_tokens: int = 1024
    max_context: int = 32768
    max_images: int = 8
    keep_images: int = 3
    max_nudges: int = 2
    env_timeout_s: float = 300.0
    max_chunks: int = 0  # >0: the episode ends (and is submitted) rather than open chunk max_chunks + 1

    def chunk_bound(self) -> int:
        """Chunks an episode of `max_turns` one-screenshot turns produces: 1, plus one per window drop."""
        extra = max(0, self.max_turns - self.max_images)
        return 1 + -(-extra // max(1, self.max_images - self.keep_images))


@dataclass
class GenOut:
    """What the engine sampled for one turn."""

    tokens: list[int]
    logprobs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"


GenerateFn = Callable[[list[int], list[ImageItem], int], Awaitable[GenOut]]


@dataclass
class Chunk:
    """One training sample: `tokens[:prompt_len]` is context; after it, `loss_mask` marks the sampled tokens.

    `loss_mask` and `logprobs` are aligned with `tokens` (0 / 0.0 on context). `images` are the images whose vision
    tokens appear in `tokens`, in order.
    """

    index: int
    tokens: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    images: list[ImageItem]
    prompt_len: int

    @property
    def response_length(self) -> int:
        return len(self.tokens) - self.prompt_len

    @property
    def n_trained(self) -> int:
        return sum(self.loss_mask)


@dataclass
class EpisodeResult:
    task: dict
    chunks: list[Chunk] = field(default_factory=list)
    reward: float = 0.0
    graded: bool = False
    grade: dict | None = None
    status: str = "completed"  # completed (policy submitted) | auto_submitted | error
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)


class Trajectory:
    """The conversation as token segments, the image window, and the chunks it closes."""

    def __init__(self, chat: QwenChat, max_images: int, keep_images: int, max_chunks: int = 0):
        self.chat = chat
        self.max_images = max_images
        self.keep_images = keep_images
        self.max_chunks = max_chunks
        self.capped = False
        self.segs: list[Seg] = []
        self.n_dropped = 0
        self.chunk_start = 0
        self.chunks: list[Chunk] = []

    def extend(self, segs: list[Seg]) -> None:
        self.segs.extend(segs)

    def assemble(self, end: int | None = None) -> Chunk:
        chat = self.chat
        tokens: list[int] = []
        mask: list[int] = []
        lps: list[float] = []
        images: list[ImageItem] = []
        prompt_len = None
        dropped_seen = 0
        for i, s in enumerate(self.segs[:end]):
            if s.kind == "image":
                if s.droppable and dropped_seen < self.n_dropped:
                    dropped_seen += 1
                    ids = chat.placeholder_tokens
                else:
                    if s.droppable:
                        dropped_seen += 1
                    ids = s.tokens
                    images.append(s.image)
                tokens += ids
                mask += [0] * len(ids)
                lps += [0.0] * len(ids)
            elif s.kind == "gen" and i >= self.chunk_start:
                if prompt_len is None:
                    prompt_len = len(tokens)
                tokens += s.tokens
                mask += [1] * len(s.tokens)
                lps += list(s.logprobs or [0.0] * len(s.tokens))
            else:
                tokens += s.tokens
                mask += [0] * len(s.tokens)
                lps += [0.0] * len(s.tokens)
        return Chunk(len(self.chunks), tokens, mask, lps, images, len(tokens) if prompt_len is None else prompt_len)

    def close_chunk(self) -> None:
        """Close the current chunk at its last sampled turn (trailing context the engine never answered is cut)."""
        last = max((i for i, s in enumerate(self.segs) if s.kind == "gen" and i >= self.chunk_start), default=None)
        if last is None:
            return
        chunk = self.assemble(end=last + 1)
        if chunk.n_trained:
            self.chunks.append(chunk)

    def apply_window(self) -> bool:
        """Drop old screenshots once more than `max_images` are visible; True when the prompt was rewritten.

        With `max_chunks`, a drop that would open one chunk too many sets `capped` instead (the episode ends)."""
        droppable = sum(1 for s in self.segs if s.kind == "image" and s.droppable)
        if droppable - self.n_dropped <= self.max_images:
            return False
        if self.max_chunks and len(self.chunks) + 1 >= self.max_chunks:
            self.capped = True
            return False
        self.close_chunk()
        self.n_dropped = droppable - self.keep_images
        self.chunk_start = len(self.segs)
        return True


def _task_reset_kwargs(task: dict, cfg: EpisodeConfig) -> dict:
    kwargs = {k: v for k, v in task.items() if k in ("task_id", "split", "index", "mode", "surface", "skin",
                                                       "coordinates", "theme", "layout")}
    kwargs.setdefault("surface", "gui")
    kwargs.setdefault("mode", "reference")
    kwargs.setdefault("coordinates", "normalized")
    kwargs["max_steps"] = cfg.max_turns
    return kwargs


async def run_episode(env_url: str, chat: QwenChat, task: dict, generate: GenerateFn,
                      cfg: EpisodeConfig | None = None, keep_transcript: bool = False) -> EpisodeResult:
    """Play one episode; never raises for env or engine failures (status="error", reward 0, no chunks)."""
    cfg = cfg or EpisodeConfig()
    res = EpisodeResult(task=dict(task))
    m = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "parse_errors": 0, "nudges": 0, "auto_submitted": 0,
         "truncated_turns": 0, "context_cut": 0, "window_drops": 0, "chunk_cap": 0, "gen_s": 0.0, "env_s": 0.0,
         "gen_tokens": 0}
    t0 = time.time()
    traj = Trajectory(chat, cfg.max_images, cfg.keep_images, cfg.max_chunks)
    session = EnvSession(env_url, timeout_s=cfg.env_timeout_s)
    done, grade = False, None
    try:
        t_env = time.time()
        meta = await session.reset(**_task_reset_kwargs(task, cfg))
        tools = [t for t in await session.tools() if t["name"] not in HIDDEN_TOOLS]
        mode = meta.get("mode", task.get("mode", "reference"))
        opening: list[ImageItem] = []
        if mode == "reference" and meta.get("reference_jpeg_b64"):
            opening.append(chat.image_item("image/jpeg", meta["reference_jpeg_b64"]))
        shot = await session.call("screenshot", {})
        opening += [chat.image_item(mime, b64) for mime, b64 in shot.images]
        intro = INTRO[mode] + (" The second image is the editor as it is now." if mode == "reference"
                               else " The image is the editor as it is now.")
        traj.extend(chat.opening(meta["instructions"], tools, opening, intro))
        m["env_s"] += time.time() - t_env
        res.task.update(episode_id=meta.get("episode_id"), task_id=meta.get("task_id"))

        nudges = 0
        for turn in range(1, cfg.max_turns + 1):
            current = traj.assemble()
            if len(current.tokens) + cfg.max_new_tokens > cfg.max_context:
                m["context_cut"] = 1
                break
            t_gen = time.time()
            out = await generate(current.tokens, current.images, cfg.max_new_tokens)
            m["gen_s"] += time.time() - t_gen
            if out.finish_reason == "abort" or not out.tokens:
                raise RuntimeError(f"engine aborted the turn (finish_reason={out.finish_reason})")
            m["turns"] = turn
            m["gen_tokens"] += len(out.tokens)
            traj.extend([Seg("gen", list(out.tokens), logprobs=list(out.logprobs), turn=turn)])
            sampled = list(out.tokens)
            if sampled[-1] == chat.im_end:
                sampled = sampled[:-1]
            else:  # cut at max_new_tokens: close the turn as context so the history stays well-formed
                m["truncated_turns"] += 1
                traj.extend([Seg("text", chat.closing_tokens(), turn=turn)])
            text = chat.decode(sampled)
            content, calls = parse_tool_calls(text, tools)
            entry: dict[str, Any] = {"turn": turn, "text": text, "calls": []} if keep_transcript else {}

            t_env = time.time()
            if not calls:
                nudges += 1
                m["nudges"] += 1
                if keep_transcript:
                    res.transcript.append(entry)
                if nudges > cfg.max_nudges:
                    break
                traj.extend(chat.observation([], [], NUDGE, turn))
                traj.apply_window()
                m["env_s"] += time.time() - t_env
                if traj.capped:
                    m["chunk_cap"] = 1
                    break
                continue

            pairs: list[tuple[str, ToolOutcome]] = []
            for call in calls:
                m["tool_calls"] += 1
                if call.parse_error:
                    m["parse_errors"] += 1
                    outcome = ToolOutcome(f"Error: {call.parse_error}", [], True)
                else:
                    try:
                        outcome = await session.call(call.name, call.args)
                    except RuntimeError as exc:  # a tool raising on the server: the model sees it
                        outcome = ToolOutcome(f"Error: {exc}", [], True)
                m["tool_errors"] += int(outcome.is_error)
                pairs.append((call.name or "the tool", outcome))
                if keep_transcript:
                    entry["calls"].append({"name": call.name, "args": call.args, "ok": not outcome.is_error,
                                           "result": outcome.text[:300], "images": len(outcome.images)})
                if outcome.done:
                    done, grade = True, outcome.grade
                    break
            if keep_transcript:
                res.transcript.append(entry)
            if done:
                m["env_s"] += time.time() - t_env
                break
            if not any(name == "screenshot" for name, _ in pairs):
                shot = await session.call("screenshot", {})
                if shot.done:
                    done, grade = True, shot.grade
                    break
                pairs[-1][1].images.extend(shot.images)
            images, notes = [], []
            for name, outcome in pairs:
                if outcome.images:
                    images += [chat.image_item(mime, b64) for mime, b64 in outcome.images]
                    notes.append(f"Image returned by {name}.")
            traj.extend(chat.observation([(n, o.text) for n, o in pairs], images, "\n".join(notes) or None, turn))
            m["window_drops"] += int(traj.apply_window())
            m["env_s"] += time.time() - t_env
            if traj.capped:
                m["chunk_cap"] = 1
                break

        if not done:
            t_env = time.time()
            out_submit = await session.call("submit", {})
            grade = out_submit.grade
            m["auto_submitted"] = 1
            res.status = "auto_submitted"
            m["env_s"] += time.time() - t_env
        traj.close_chunk()
        res.chunks = traj.chunks
        res.grade = grade
        res.graded = bool(grade)
        res.reward = float(grade["total"]) if grade else 0.0
    except Exception as exc:  # noqa: BLE001 - one broken episode must not kill a training step
        res.status, res.error = "error", f"{type(exc).__name__}: {exc}"[:800]
        res.chunks, res.reward = [], 0.0
    finally:
        try:
            await asyncio.wait_for(session.close(), timeout=30)
        except Exception:  # noqa: BLE001
            pass
    m["chunks"] = len(res.chunks)
    m["seconds"] = time.time() - t0
    m["reward"] = res.reward
    if grade:
        for k in ("structure", "visual", "coverage", "order", "background"):
            if isinstance(grade.get(k), (int, float)):
                m[f"grade_{k}"] = float(grade[k])
    res.metrics = {k: float(v) for k, v in m.items()}
    return res
