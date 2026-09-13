# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The client side: what a trainer imports.

A trainer gets a `ResourceSessionFactory`. It calls `create(prompt)`, blocks on
`wait_for_completion()`, reads `fetch_proxy_trace()`, and trains. It never learns which sandbox ran,
which harness drove, how the answer was graded, or that an HTTP server was involved -- those are the
environment's business, and keeping them there is what makes the training script short.
"""

from __future__ import annotations

import logging
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    TraceEntry,
    VerifyResult,
)

from .models import DataAgentRolloutResult
from .task import instruction_id
from .tasks import index_of_instruction


logger = logging.getLogger(__name__)


def to_trace_entries(result: DataAgentRolloutResult) -> list[TraceEntry]:
    """`DataAgentRolloutResult` -> `list[TraceEntry]`, carrying the engine's own tokenization.

    Every entry includes `prompt_token_ids` and `loss_mask`, so the consumer never re-renders a
    prompt. `request_messages` is what makes this possible at all: without it the token fields say
    what was produced but not what produced them.

    Turns that are not trainable are dropped rather than emitted with a zero mask. They carry no
    gradient either way, and emitting them would inflate the turn count that the efficiency bonus
    and every turn-based diagnostic read.
    """
    entries: list[TraceEntry] = []
    for turn in result.turns:
        if not turn.trainable or not turn.completion_token_ids:
            continue
        entries.append(
            {
                "request": {
                    "messages": list(turn.request_messages),
                    "tools": turn.request_tools,
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": turn.text,
                                "tool_calls": turn.tool_calls or None,
                            },
                            "finish_reason": turn.finish_reason,
                        }
                    ]
                },
                "prompt_token_ids": list(turn.prompt_token_ids),
                "completion_token_ids": list(turn.completion_token_ids),
                "per_token_logps": list(turn.per_token_logps),
                "loss_mask": [0] * len(turn.prompt_token_ids)
                + [1] * len(turn.completion_token_ids),
                "reward": result.reward,
                "metadata": dict(result.metadata),
            }
        )
    return entries



def opencode_agent_turns(trace: list[TraceEntry]) -> list[TraceEntry]:
    """`agent_turn_fn` for TRL: keep only the REAL agent turns.

    opencode fires extra model calls for its own bookkeeping -- a conversation-title generator and a
    context summarizer -- either without tools or under a different system prompt. They are a
    different task from the one being trained, and leaving them in the trace is not a cosmetic
    problem: measured on a 400-step run against a reference that filtered them,

      * each aux call has its own prompt, so it does not extend the previous turn. The prefix chain
        breaks -- `rollout/fork_frac` 0.02-0.06 against the reference's flat 0, and
        `rollout/drift_tokens_max` reached 32,770, a whole rollout's worth;
      * one rollout therefore stops being one sample (`rollout/samples_per_rollout` 1.04-1.31 vs
        exactly 1.0), which doubles `sample/forwarded_tokens_mean` and roughly doubles `grad_norm`
        at an unchanged learning rate;
      * worst, the aux tokens get TRAINED, carrying the data-analysis task's advantage. The policy is
        optimised to write titles and summaries. Reward fell 0.46 -> 0.00 while entropy rose
        0.195 -> 0.708, from a run whose step-0 reward was HIGHER than the reference's;
      * and the title call fires LAST, so `entries[-1]` is an aux call and every metric read off it
        -- `tools/failure_frequency` above all -- describes the wrong conversation.

    The agent loop reuses ONE tool-enabled system prompt, so anchor on the first tool-enabled turn's
    system prompt and keep only the entries that match it.
    """

    def system_of(messages: list[dict[str, Any]]) -> Any:
        return next((m.get("content") for m in messages if m.get("role") == "system"), None)

    primary = None
    for entry in trace:
        request = entry.get("request") or {}
        if request.get("messages") and request.get("tools"):
            primary = system_of(request["messages"])
            break
    return [
        entry
        for entry in trace
        if (request := entry.get("request") or {}).get("messages")
        and request.get("tools")
        and system_of(request["messages"]) == primary
    ]


class DataAgentSession(ResourceSession):
    """One rollout. The agent owns its loop; this blocks on it and reads back what it did.

    Satisfies `LoopOwningSession` structurally via `wait_for_completion` and `fetch_proxy_trace`.
    The tool methods exist because `ResourceSession` requires them, but nothing calls them in
    loop-owning mode -- the agent inside the sandbox has its own tools.
    """

    def __init__(
        self,
        client: Any,
        split: str,
        index: int,
        instruction: str,
        **rollout_kwargs: Any,
    ):
        self._client = client
        self._split = split
        self._index = index
        self._instruction = instruction
        self._rollout_kwargs = rollout_kwargs
        self._result: DataAgentRolloutResult | None = None

    # --- ResourceSession ---------------------------------------------------------------------

    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self._instruction}]

    def list_tools(self) -> list[Tool]:
        return []  # the agent's tools live inside the sandbox, not on this side of the wire

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError("loop-owning session: the agent calls its own tools")

    def verify(
        self, transcript: list[dict[str, Any]], final_state: Any | None = None
    ) -> VerifyResult:
        """The environment's reward, forwarded. Never synthesised here.

        `env_reward=None` means the rollout could not be graded, and the trainer drops it from the
        group baseline. Coercing it to 0.0 would silently turn an infrastructure failure into a
        training signal that says the policy was wrong.
        """
        if self._result is None:
            return VerifyResult(env_reward=None, done=True)
        return VerifyResult(
            env_reward=self._result.reward,
            done=True,
            metrics={
                "correctness": self._result.correctness,
                "n_tool_calls": self._result.n_tool_calls,
                "timed_out": self._result.timed_out,
            },
            artifacts={"answer": self._result.answer, **self._result.metadata},
        )

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                logger.warning(
                    "closing data-agent session client failed", exc_info=True
                )

    # --- LoopOwningSession -------------------------------------------------------------------

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Run the rollout to completion on the server and keep the result.

        This is one long call rather than a poll loop: the whole rollout -- sandbox boot, data
        staging, the agent's own tool loop, grading -- happens server-side, and there is no
        meaningful intermediate state to observe.
        """
        self._result = self._client.run_rollout(
            split=self._split,
            index=self._index,
            timeout_s=timeout_s,
            **self._rollout_kwargs,
        )
        return 0

    def fetch_proxy_trace(self) -> list[TraceEntry]:
        if self._result is None:
            return []
        if self._result.rollout_type != "train":
            # Better to say so than to hand back rows of zeros. The engine was not serving token ids,
            # so nothing here can be trained on.
            raise ValueError(
                "this rollout is eval-tier: the engine did not return token ids, so it produced no "
                "trainable turns. Serve with --return-tokens-as-token-ids --logprobs-mode "
                "processed_logprobs."
            )
        return to_trace_entries(self._result)


class DataAgentSessionFactory(ResourceSessionFactory[DataAgentSession]):
    """Hands the trainer one session per rollout.

    Args:
        server (`str`):
            Base URL of a running data-agent environment server.
        split (`str`, *optional*, defaults to `"train"`):
            Which split to draw from, e.g. `"train"` or `"train:medium"`.
        llm_url (`str`):
            The engine the agent should call. Chosen per rollout, so one server serves training and
            evaluation against different engines at once.
        model (`str`):
            Served model id.
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Backend name; `"e2b"` or `"hf"`.
        curriculum (`str`, *optional*):
            Order the prompts by difficulty instead of shuffling: `"sprinkle"` or `"warmup:<n>"`.
            Empty keeps the split's own order. See `curriculum.py`.
        seed (`int`, *optional*, defaults to `0`):
            Makes the curriculum reproducible.
    """

    def __init__(
        self,
        server: str,
        *,
        split: str = "train",
        llm_url: str,
        model: str,
        sandbox: str = "e2b",
        agent_step_limit: int = 10,
        agent_timeout_s: float = 600.0,
        curriculum: str = "",
        seed: int = 0,
    ):
        self._server = server.rstrip("/")
        self._split = split
        self._curriculum = curriculum
        self._seed = seed
        self._rollout_kwargs = {
            "llm_url": llm_url,
            "model": model,
            "sandbox": sandbox,
            "agent_step_limit": agent_step_limit,
            "agent_timeout_s": agent_timeout_s,
        }

    def _new_client(self):
        """A CLIENT PER SESSION.

        One shared MCP client across concurrent rollouts raises `ConcurrencyError: cannot call recv
        while another coroutine is already running recv` -- the transport has no request-id
        correlation. With `num_generations` rollouts in flight that is every rollout of every step,
        each returning unscorable: a run that looks alive and trains on nothing.
        """
        from .client import DataAgentEnv

        return DataAgentEnv(base_url=self._server)

    def prompt_rows(self) -> list[dict[str, Any]]:
        """Rows for the trainer's dataset: one prompt per task in the configured split.

        The trainer forwards only the prompt, so `create()` recovers the index by hashing the
        instruction back. That round trip is why this returns instructions rather than indices.

        ORDER IS PART OF THE RECIPE, not a detail. The trainer walks this list in order, so the
        curriculum has to be applied HERE -- once, where the tiers are still known. Requesting
        `curriculum="warmup:125"` puts 125 easy prompts first and then sprinkles hard through medium;
        without it the run sees a shuffled mix from step 0, and a group whose `num_generations`
        rollouts all score zero contributes exactly zero gradient. On the harder tiers, early on, that
        is the likely outcome.
        """
        client = self._new_client()
        try:
            tasks = client.get_task_range(self._split)
        finally:
            client.close()
        # DEDUPLICATED BY INSTRUCTION, and this is a correctness fix rather than tidying.
        #
        # The trainer forwards only the prompt, so `create()` resolves an instruction back to ONE task
        # index. When two tasks share an instruction, every rollout for it is graded against whichever
        # gold that lookup returns. Measured on `train`: 5000 tasks carry 4940 distinct instructions,
        # 48 instructions are shared by 108 tasks, and 15 of those 48 have CONFLICTING gold answers --
        # so without this, a slice of the run is scored against an answer to a different question and
        # nothing anywhere says so.
        #
        # Keeping the first occurrence also reproduces the pool the published +0.2028 run used, which
        # was 4940 prompts: its dataset builder deduplicated the same way.
        seen: set[str] = set()
        rows, kept_tasks = [], []
        for task in tasks:
            instruction = task["instruction"]
            key = instruction_id(instruction)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": [{"role": "user", "content": instruction}]})
            kept_tasks.append(task)
        if len(kept_tasks) != len(tasks):
            logger.warning(
                "%d of %d tasks in %s share an instruction with an earlier one and were dropped; "
                "the prompt cannot distinguish them, so a rollout could only be graded against one "
                "of their golds",
                len(tasks) - len(kept_tasks),
                len(tasks),
                self._split,
            )
        tasks = kept_tasks
        if not self._curriculum:
            return rows

        from .curriculum import apply_curriculum

        # Keyed by id(row), so the tier travels with the row rather than being re-derived from the
        # instruction later. Deriving it from the raw instruction string is how the original version
        # of this silently degraded to a plain shuffle: the task index is keyed by a HASH of the
        # instruction, so a raw-string lookup returns None for every row and the curriculum reports
        # success while doing nothing.
        tier_of = {id(row): task.get("difficulty_tier") for row, task in zip(rows, tasks)}
        return apply_curriculum(rows, tier_of, self._curriculum, self._seed)

    def create(
        self, task: Any, seed: int | None = None, episode_id: str | None = None
    ) -> DataAgentSession:
        instruction = _instruction_of(task)
        index = index_of_instruction(self._split, instruction)
        if index is None:
            raise ValueError(
                "could not map this prompt back to a task in split "
                f"{self._split!r}. The dataset the trainer was built from and the split this factory "
                "serves have diverged."
            )
        return DataAgentSession(
            self._new_client(), self._split, index, instruction, **self._rollout_kwargs
        )


def _instruction_of(task: Any) -> str:
    """The instruction text, from whatever shape the trainer passed."""
    if isinstance(task, str):
        return task
    if isinstance(task, list) and task:  # a message list
        return task[-1].get("content", "")
    if isinstance(task, dict):
        prompt = task.get("prompt")
        if isinstance(prompt, list) and prompt:
            return prompt[-1].get("content", "")
        return task.get("instruction", "")
    raise TypeError(f"cannot read an instruction out of {type(task).__name__}")
