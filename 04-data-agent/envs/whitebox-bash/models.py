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

"""Typed payloads crossing the wire.

These describe what `start_episode`, a tool call and `grade` return. The MCP tools return plain
dicts -- FastMCP builds the schema from the annotations, and a Pydantic return type there changes
the envelope the client has to unwrap -- so these are the documented shape of those dicts and the
type the client parses them into, rather than the tools' literal return annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeStart:
    """What `start_episode` hands back.

    `prompt` is the task text. It is returned rather than looked up client-side on purpose: the gold
    answer never leaves the server, so the client cannot construct the task itself, and a client that
    could would be one bug away from training on the answer.
    """

    session_id: str
    prompt: str
    task_id: str = ""
    workdir: str = ""

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "EpisodeStart":
        return cls(
            session_id=str(d.get("session_id", "")),
            prompt=str(d.get("prompt", "")),
            task_id=str(d.get("task_id", "")),
            workdir=str(d.get("workdir", "")),
        )


@dataclass
class ToolResult:
    """What one tool call returns.

    `output` is the exact string the model sees next turn, already clipped server-side. `ok` is the
    tool's own verdict and is NOT derivable from the text: a `grep` that matches nothing succeeds
    with empty output, and a `read` of a missing file fails with a message that reads like content.
    """

    output: str = ""
    ok: bool = True

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "ToolResult":
        return cls(output=str(d.get("output", "")), ok=bool(d.get("ok", True)))


@dataclass
class Grade:
    """What `grade` returns.

    `reward` of `None` means UNGRADED -- the episode could not be judged (a dead sandbox), which is
    a different event from a wrong answer and must not be collapsed into `0.0`.
    """

    reward: float | None = None
    correct: bool = False
    check_passed: bool | None = None
    submitted: str = ""
    n_tool_calls: int = 0
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ungraded(self) -> bool:
        return self.reward is None

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "Grade":
        reward = d.get("reward")
        return cls(
            reward=None if reward is None else float(reward),
            correct=bool(d.get("correct", False)),
            check_passed=d.get("check_passed"),
            submitted=str(d.get("submitted", "")),
            n_tool_calls=int(d.get("n_tool_calls", 0) or 0),
            note=str(d.get("note", "")),
        )
