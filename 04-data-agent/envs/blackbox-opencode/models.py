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

"""Wire types for the data-agent environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class DataAgentTurn(BaseModel):
    """One model call, with the tokens the ENGINE produced.

    `prompt_token_ids` is the engine's own tokenization of everything before this turn, not a local
    re-render. That distinction is the reason this environment can be trained on at all: a
    re-rendered prompt matched the engine on 0 of 28 measured turns on Qwen3.5-4B, and training on
    the difference collapsed a run at its first weight update.

    Attributes:
        turn (`int`):
            Position in the rollout.
        prompt_token_ids (`list[int]`):
            Engine tokenization of the conversation before this turn.
        completion_token_ids (`list[int]`):
            Tokens the model sampled.
        per_token_logps (`list[float]`):
            Generator logprobs, aligned with `completion_token_ids`.
        trainable (`bool`):
            Whether this turn may be trained on. False when its logprobs were rejected on ingest:
            the tokens remain as context but must not carry gradient.
        request_messages (`list[dict]`):
            The conversation sent upstream, for inspection and for partial-credit grading.
        text (`str`):
            The assistant's text content.
        tool_calls (`list[dict]`):
            Tool calls the assistant emitted.
        finish_reason (`str`, *optional*):
            Why generation stopped.
        n_tools (`int`):
            Tools offered on this call.
    """

    turn: int = 0
    prompt_token_ids: list[int] = Field(default_factory=list)
    completion_token_ids: list[int] = Field(default_factory=list)
    per_token_logps: list[float] = Field(default_factory=list)
    trainable: bool = True
    request_messages: list[dict[str, Any]] = Field(default_factory=list)
    request_tools: list[dict[str, Any]] | None = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    n_tools: int = 0


class DataAgentRolloutResult(BaseModel):
    """Everything one rollout produced: the turns, the grade, and how it was graded.

    Attributes:
        rollout_type (`str`):
            `"train"` when the engine returned token ids, `"eval"` otherwise. An eval rollout yields
            no trainable turns rather than rows of zeros.
        reward (`float`, *optional*):
            The training reward. `None` means UNGRADED and must not be read as 0.0 -- an ungraded
            rollout is dropped from the group baseline, not counted as a failure.
        correctness (`float`, *optional*):
            The graded score before the efficiency bonus.
        answer (`str`, *optional*):
            What the agent submitted.
        answer_source (`str`):
            `"file"`, `"chat"` or `"none"`.
        graded_by (`str`):
            Which comparison tier matched.
        turns (`list[DataAgentTurn]`):
            Per-turn token records.
        n_tool_calls (`int`):
            Tool calls made, used for the efficiency bonus.
        timed_out (`bool`):
            Whether the agent hit its wall clock.
        metadata (`dict`):
            Task id, tier, sandbox, session id.
    """

    rollout_type: str = "eval"
    reward: float | None = None
    correctness: float | None = None
    answer: str | None = None
    answer_source: str = "none"
    graded_by: str = ""
    turns: list[DataAgentTurn] = Field(default_factory=list)
    n_tool_calls: int = 0
    timed_out: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataAgentState(State):
    """Server state: which task is selected, and what the engine can do."""

    split: str | None = None
    index: int | None = None
    task_id: str | None = None
    rollout_type: str = "eval"
