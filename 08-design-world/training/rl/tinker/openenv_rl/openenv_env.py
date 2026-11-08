"""Tinker cookbook RL environments over OpenEnv `ResourceSession`s (the white-box path).

The trainer owns sampling: Tinker samples tokens, the cookbook renderer parses them into tool calls,
and each call runs against the environment through its session. Any `ResourceSessionFactory` works
(every `envs/<env>/harness.py` factory, `StepEnvSessionAdapter`, or `MCPSessionFactory` for a plain
MCP environment). The reward is the environment's own, read back through `ResourceSession.verify()`
or the rewards its tool results carried; this module never computes one.

Typical use with the cookbook's RL loop::

    from tinker_cookbook.rl import train
    from openenv_rl import OpenEnvRLDatasetBuilder

    dataset = OpenEnvRLDatasetBuilder(
        session_factory=factory, tasks=tasks, model_name="Qwen/Qwen3.6-35B-A3B",
        batch_size=16, group_size=8, max_turns=40,
    )
    await train.main(train.Config(dataset_builder=dataset, model_name="Qwen/Qwen3.6-35B-A3B", ...))
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import chz
from openenv.core.harness import ResourceSession, ResourceSessionFactory
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from .content import count_images, openai_content_to_parts
from .openenv_tools import EpisodeLedger, SessionTool

logger = logging.getLogger(__name__)

RewardSource = Literal["verify", "sum"]


class MissingRewardError(RuntimeError):
    """The episode ended without the environment producing a reward.

    Raised rather than scored as 0: a rollout that was never graded is not a wrong answer. The
    cookbook's rollout strategies (`FailFast`, `RetryOnFailure`) decide what to do with it.
    """


@functools.cache
def make_renderer(model_name: str, renderer_name: str | None = None, *, extension: bool = True) -> Renderer:
    """The cookbook renderer for a model, with an image processor attached when the model has one.

    Args:
        model_name: The Tinker base model.
        renderer_name: Override the recommended renderer.
        extension: Render history exactly as it was sampled (keep earlier turns' thinking blocks),
            so every turn's prompt extends the previous one. That lets the cookbook train a
            multi-turn episode as one sequence instead of one sequence per turn, and lets the
            sampler reuse its prompt cache. Turn off to match the Hugging Face chat template,
            which strips historical thinking.

    Cached per process: tokenizers and image processors are expensive to load and renderers are
    stateless.
    """
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    image_processor = None
    try:
        from tinker_cookbook.image_processing_utils import get_image_processor

        image_processor = get_image_processor(model_name)
    except Exception:  # noqa: BLE001 - text-only models have no image processor
        image_processor = None
    renderer = get_renderer(name, tokenizer, image_processor=image_processor, model_name=model_name)
    if extension and hasattr(renderer, "strip_thinking_from_history"):
        renderer.strip_thinking_from_history = False
    if extension:
        _trim_text_before_tool_calls(renderer)
    return renderer


def _trim_text_before_tool_calls(renderer: Renderer) -> None:
    """Qwen3.5-family renderers (Qwen3.5, Qwen3.6, Qwen3.8) put a blank line between an assistant's text and its
    first tool call, and trim the text first, but only when the content is a plain string. A parsed sampled
    message carries its content as parts, so the text keeps the blank line the model sampled and the history
    renders `\\n\\n\\n\\n<tool_call>` where the model wrote `\\n\\n<tool_call>`. Every turn then fails to extend the
    last, so each one trains as its own sequence (tinker-cookbook 0.5.7). Trimming the text parts' trailing
    whitespace, as the template does, renders the history exactly as sampled."""
    try:
        from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
    except ImportError:
        return
    if not isinstance(renderer, Qwen3_5Renderer):
        return
    render_message = renderer.render_message

    def trimmed(message, ctx):
        content = message.get("content")
        if message.get("role") == "assistant" and message.get("tool_calls") and isinstance(content, list):
            texts = [i for i, p in enumerate(content) if p.get("type") == "text"]
            if texts:
                parts = list(content)
                parts[texts[-1]] = {**parts[texts[-1]], "text": parts[texts[-1]]["text"].rstrip()}
                message = {**message, "content": parts}
        return render_message(message, ctx)

    renderer.render_message = trimmed


@dataclass(frozen=True)
class ImageWindow:
    """Keep long screenshot histories inside the context window.

    Once more than `max_images` screenshots are in the conversation, the oldest are replaced by a short
    text placeholder until only `keep_images` remain. Images in the episode's opening messages (the task, e.g. a
    target image) are never dropped and do not count toward the window. Dropping in chunks rather than one per step
    keeps each rendered prompt a prefix of the next between drops, so the cookbook can still merge
    consecutive turns into one training sequence and the sampler can reuse its prompt cache.

    Args:
        max_images: Images allowed in the prompt before a drop.
        keep_images: Images kept (the most recent) after a drop.
        placeholder: Text that stands in for a dropped image.
    """

    max_images: int = 8
    keep_images: int = 3
    placeholder: str = "[earlier screenshot omitted]"

    def __post_init__(self):
        if not 0 <= self.keep_images <= self.max_images:
            raise ValueError("ImageWindow needs 0 <= keep_images <= max_images")


class _WindowedMessageEnv(MessageEnv):
    """Wraps the cookbook's tool-use MessageEnv: windows images and pays out step rewards.

    The inner env keeps the full history (for grading); the renderer sees the windowed view.
    """

    def __init__(
        self,
        inner: Any,
        ledger: EpisodeLedger,
        window: ImageWindow | None,
        reward_source: RewardSource,
        tool_images_as_user: bool = False,
    ):
        self._inner = inner
        self._ledger = ledger
        self._window = window
        self._reward_source = reward_source
        self._tool_images_as_user = tool_images_as_user
        self._dropped = 0  # how many of the oldest images are hidden; only ever grows

    @staticmethod
    def _images_to_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tool results keep their text; their images follow in one user turn, "Image returned by <tool>." per tool.

        This is how OpenAI-compatible chat APIs (and so the eval harness, `rollout.py`) carry tool images, which they
        cannot put in tool messages. Applied to the whole history each step, so every prompt still extends the last.
        """
        out: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        notes: list[str] = []

        def flush() -> None:
            if images or notes:
                out.append({"role": "user", "content": [*images, {"type": "text", "text": "\n".join(notes) or "Continue."}]})
                images.clear()
                notes.clear()

        for message in messages:
            content = message.get("content")
            if message.get("role") == "tool" and count_images(content):
                kept = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image")]
                images.extend(p for p in content if isinstance(p, dict) and p.get("type") == "image")
                notes.append(f"Image returned by {message.get('name') or 'the tool'}.")
                text = "".join(p.get("text", "") for p in kept if isinstance(p, dict))
                out.append({**message, "content": text})
                continue
            if message.get("role") != "tool":
                flush()
            out.append(message)
        flush()
        return out

    def __getattr__(self, name: str) -> Any:  # seams the rollout runner calls (set_max_tool_calls, ...)
        return getattr(self._inner, name)

    def _view(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hide the oldest droppable screenshots; images in the opening messages (e.g. the target) never drop."""
        if self._tool_images_as_user:
            messages = self._images_to_user(messages)
        if self._window is None:
            return messages
        n_initial = len(getattr(self._inner, "initial_messages", []) or [])
        protected = sum(count_images(m.get("content")) for m in messages[:n_initial])
        droppable = sum(count_images(m.get("content")) for m in messages) - protected
        if droppable - self._dropped > self._window.max_images:
            self._dropped = droppable - self._window.keep_images
        if self._dropped == 0:
            return messages
        out, seen = [], 0
        for message in messages:
            content = message.get("content")
            if count_images(content) and seen < protected + self._dropped:
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        seen += 1
                        if protected < seen <= protected + self._dropped:
                            parts.append({"type": "text", "text": self._window.placeholder})
                            continue
                    parts.append(part)
                message = {**message, "content": parts}
            out.append(message)
        return out

    def _step_reward(self) -> float:
        if self._reward_source != "sum":
            return 0.0
        new = self._ledger.rewards[self._ledger.unread :]
        self._ledger.unread = len(self._ledger.rewards)
        return float(sum(new))

    async def initial_observation(self) -> list[dict[str, Any]]:
        return self._view(await self._inner.initial_observation())

    async def step(self, message: dict[str, Any]) -> MessageStepResult:
        result = await self._inner.step(message)
        result.reward += self._step_reward()
        result.next_messages = self._view(result.next_messages)
        return result

    async def observe_truncated_response(self, message: dict[str, Any]):
        messages = await self._inner.observe_truncated_response(message)
        return None if messages is None else self._view(messages)

    async def add_messages(self, messages: list[dict[str, Any]]):
        out = await self._inner.add_messages(messages)
        return None if out is None else self._view(out)


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull leading system messages out (the renderer re-emits them with the tool schemas)."""
    system, rest = [], list(messages)
    while rest and rest[0].get("role") == "system":
        content = openai_content_to_parts(rest.pop(0).get("content"))
        system.append(content if isinstance(content, str) else "".join(p.get("text", "") for p in content))
    return "\n\n".join(s for s in system if s), rest


def _to_cookbook_message(message: dict[str, Any]) -> dict[str, Any]:
    out = {
        "role": message["role"],
        "content": openai_content_to_parts(message.get("content")),
    }
    for key in ("tool_call_id", "name"):
        if key in message:
            out[key] = message[key]
    return out


class OpenEnvEnvGroupBuilder(EnvGroupBuilder):
    """A GRPO group: `group_size` episodes of one task, each in its own OpenEnv session.

    Pickle-safe like the cookbook's own builders: only configuration is stored; renderers and
    sessions are created in `make_envs()`.

    Args:
        session_factory: Creates one `ResourceSession` per episode (must be picklable).
        task: Passed to `session_factory.create(task, seed=...)`.
        group_size: Episodes per group (all with the same task and seed).
        model_name: The Tinker base model; picks the renderer and image processor.
        renderer_name: Override the recommended renderer (e.g. `"qwen3_5_disable_thinking"`).
        seed: Seed for every session in the group (same task instance across the group).
        max_turns: Assistant turns before the episode is cut off.
        rollout_config: Cookbook rollout configuration (token budgets, parse-error policy, ...).
        max_trajectory_tokens: Context budget for one episode.
        reward_source: `"verify"` grades with `session.verify()` at the end (cross-checked against
            the last tool reward); `"sum"` pays every reward a tool result carries as a step
            reward, for environments that emit dense reward deltas.
        finalize_tool: Tool to call once if the episode ends without a tool reporting `done`
            (e.g. `"submit"`), so cut-off episodes are still graded by the environment.
        missing_reward: Reward for an episode the environment never graded. `None` raises
            `MissingRewardError` instead.
        image_window: Optional `ImageWindow` for screenshot-heavy environments.
        system_prompt: Prepended to the session's own system text.
        tags: Logging tags (default: `["openenv"]`).
    """

    def __init__(
        self,
        session_factory: ResourceSessionFactory,
        task: Any,
        group_size: int,
        model_name: str,
        *,
        renderer_name: str | None = None,
        seed: int | None = None,
        max_turns: int = 20,
        rollout_config: RolloutConfig | None = None,
        max_trajectory_tokens: int | None = None,
        reward_source: RewardSource = "verify",
        finalize_tool: str | None = None,
        missing_reward: float | None = None,
        image_window: ImageWindow | None = None,
        tool_images_as_user: bool = False,
        system_prompt: str = "",
        tags: Sequence[str] = ("openenv",),
    ):
        self.session_factory = session_factory
        self.task = task
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.seed = seed
        self.max_turns = max_turns
        self.rollout_config = rollout_config
        self.max_trajectory_tokens = max_trajectory_tokens
        self.reward_source = reward_source
        self.finalize_tool = finalize_tool
        self.missing_reward = missing_reward
        self.image_window = image_window
        self.tool_images_as_user = tool_images_as_user
        self.system_prompt = system_prompt
        self.tags = list(tags)
        self._sessions: list[ResourceSession] = []

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_sessions"] = []  # live sessions never travel to a rollout worker
        return state

    async def _create_session(self) -> ResourceSession:
        return await asyncio.to_thread(self.session_factory.create, self.task, seed=self.seed)

    async def make_envs(self) -> Sequence[Env]:
        renderer = make_renderer(self.model_name, self.renderer_name)
        self._sessions = list(await asyncio.gather(*[self._create_session() for _ in range(self.group_size)]))
        envs = []
        for session in self._sessions:
            ledger = EpisodeLedger()
            specs = await asyncio.to_thread(session.list_tools)
            tools = [SessionTool(session, spec, ledger) for spec in specs]
            system, rest = _split_system(await asyncio.to_thread(session.initial_messages))
            system = "\n\n".join(s for s in (self.system_prompt, system) if s)
            prefix = renderer.create_conversation_prefix_with_tools(
                tools=[t.to_spec() for t in tools], system_prompt=system
            )
            initial = prefix + [_to_cookbook_message(m) for m in rest]
            env = build_agent_tool_env(
                renderer=renderer,
                tools=tools,
                initial_messages=initial,
                reward_fn=functools.partial(self._grade, session, ledger, tools),
                rollout_config=self.rollout_config,
                model_name=self.model_name,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            # One session is one stateful environment: its tool calls must run in the order the
            # model issued them.
            env.message_env.tool_execution = "sequential"
            env.message_env = _WindowedMessageEnv(env.message_env, ledger, self.image_window, self.reward_source,
                                                  tool_images_as_user=self.tool_images_as_user)
            envs.append(env)
        return envs

    async def _grade(
        self,
        session: ResourceSession,
        ledger: EpisodeLedger,
        tools: list[SessionTool],
        messages,
    ):
        if not ledger.done and self.finalize_tool:
            finalize = next((t for t in tools if t.name == self.finalize_tool), None)
            if finalize is not None:
                from tinker_cookbook.tool_use import ToolInput

                await finalize.run(ToolInput(arguments={}, call_id="finalize"))
        metrics: dict[str, float] = {
            "openenv/tool_calls": float(ledger.calls),
            "openenv/tool_errors": float(ledger.errors),
            "openenv/done": float(ledger.done),
        }
        if self.reward_source == "sum":
            # Rewards were already paid per step; pay anything the final call added.
            late = float(sum(ledger.rewards[ledger.unread :]))
            ledger.unread = len(ledger.rewards)
            return late, metrics
        verify = await asyncio.to_thread(session.verify, list(messages), {"done": ledger.done})
        for key, value in (verify.metrics or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"openenv/{key}"] = float(value)
        reward = verify.env_reward
        last = ledger.last_reward
        if reward is not None and last is not None and not math.isclose(reward, last, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"verify() reward {reward} disagrees with the last tool reward {last}")
        if reward is None:
            reward = last
        if reward is None:
            if self.missing_reward is None:
                raise MissingRewardError("the environment produced no reward for this episode")
            metrics["openenv/ungraded"] = 1.0
            reward = self.missing_reward
        return float(reward), metrics

    async def cleanup(self) -> None:
        sessions, self._sessions = self._sessions, []
        for session in sessions:
            try:
                await asyncio.to_thread(session.close)
            except Exception as exc:  # noqa: BLE001 - cleanup must not raise
                logger.warning("OpenEnv session close failed: %s", exc)

    def logging_tags(self) -> list[str]:
        return list(self.tags)


class OpenEnvRLDataset(RLDataset):
    """Batches of `OpenEnvEnvGroupBuilder`s, `batch_size` groups per batch."""

    def __init__(self, builders: list[OpenEnvEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size : (index + 1) * self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class OpenEnvRLDatasetBuilder(RLDatasetBuilder):
    """Train (and optionally test) datasets over OpenEnv tasks, for `tinker_cookbook.rl.train`.

    Every keyword of `OpenEnvEnvGroupBuilder` is available here; test groups have size 1.
    """

    session_factory: Any
    tasks: list[Any]
    model_name: str
    batch_size: int
    group_size: int
    test_tasks: list[Any] | None = None
    renderer_name: str | None = None
    seed: int | None = None
    max_turns: int = 20
    rollout_config: Any = None
    max_trajectory_tokens: int | None = None
    reward_source: str = "verify"
    finalize_tool: str | None = None
    missing_reward: float | None = None
    image_window: Any = None
    tool_images_as_user: bool = False
    system_prompt: str = ""
    tags: tuple[str, ...] = ("openenv",)

    def _builders(self, tasks: list[Any], group_size: int) -> list[OpenEnvEnvGroupBuilder]:
        return [
            OpenEnvEnvGroupBuilder(
                self.session_factory,
                task,
                group_size,
                self.model_name,
                renderer_name=self.renderer_name,
                seed=self.seed,
                max_turns=self.max_turns,
                rollout_config=self.rollout_config,
                max_trajectory_tokens=self.max_trajectory_tokens,
                reward_source=self.reward_source,
                finalize_tool=self.finalize_tool,
                missing_reward=self.missing_reward,
                image_window=self.image_window,
                tool_images_as_user=self.tool_images_as_user,
                system_prompt=self.system_prompt,
                tags=self.tags,
            )
            for task in tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train = OpenEnvRLDataset(self._builders(self.tasks, self.group_size), self.batch_size)
        test = OpenEnvRLDataset(self._builders(self.test_tasks, 1), self.batch_size) if self.test_tasks else None
        return train, test
