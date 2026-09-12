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

"""Client for a running data-agent environment server.

TWO TRANSPORTS, BECAUSE THE SERVER GENUINELY HAS TWO SURFACES
`run_rollout` and `capabilities` are MCP tools, so they go through `MCPToolClient` -- the same base
every other environment client uses, and the same split `HarborEnv` makes. Task discovery
(`/splits`, `/task`, `/num_tasks`, `/task_range`) is plain HTTP routes on the env server, not MCP
tools, so those are plain HTTP. Neither half is reimplemented here.

ONE CLIENT PER SESSION, NOT ONE SHARED
A single client shared across concurrent rollouts raises `ConcurrencyError: cannot call recv while
another coroutine is already running recv` -- the transport has no request-id correlation. With
`num_generations` rollouts in flight that is every rollout of every step, each returning unscorable: a
run that looks alive and trains on nothing. `DataAgentSessionFactory` mints one of these per session.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

import httpx
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.mcp_client import MCPToolClient
from openenv.core.utils import run_async_safely

from .models import DataAgentRolloutResult


logger = logging.getLogger(__name__)

# A rollout is one long call: sandbox boot, data staging, the agent's own tool loop, then grading.
# 60-600 s is normal, so the default message timeout would abort healthy rollouts.
ROLLOUT_TIMEOUT_S = 1800.0


class DataAgentEnv(MCPToolClient):
    """Talks to a data-agent environment server.

    Args:
        base_url (`str`):
            Where the server is, e.g. `http://127.0.0.1:8200`.
        message_timeout_s (`float`, *optional*, defaults to `1800.0`):
            Per-rollout timeout.

    Examples:

    ```python
    env = DataAgentEnv("http://127.0.0.1:8200")
    print(env.splits())
    task = env.get_task("train:medium", 0)
    result = env.run_rollout(split="train:medium", index=0, llm_url=..., model=...)
    ```
    """

    def __init__(
        self,
        base_url: str,
        *,
        message_timeout_s: float = ROLLOUT_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url, message_timeout_s=message_timeout_s, **kwargs
        )
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)

    # --- discovery (Task API, plain HTTP routes) -----------------------------------------------

    def splits(self) -> list[dict[str, Any]]:
        return self._http.get("/splits").raise_for_status().json()

    def num_tasks(self, split: str = "") -> int:
        return int(self._post("/num_tasks", {"split": split})["num_tasks"])

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self._post("/task", {"split": split, "index": index})["task"]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"split": split}
        if start is not None:
            body["start"] = start
        if stop is not None:
            body["stop"] = stop
        return self._post("/task_range", body)["tasks"]

    # --- execution (MCP tools) ------------------------------------------------------------------

    def run_rollout(
        self,
        *,
        split: str,
        index: int,
        llm_url: str,
        model: str,
        sandbox: str = "e2b",
        agent_step_limit: int = 10,
        agent_timeout_s: float = 600.0,
        require_tokens: bool = True,
    ) -> DataAgentRolloutResult:
        """Run one rollout to completion and return its token-level result.

        The ENGINE is chosen per call rather than baked into the deployment, so one server can serve a
        training run and an evaluation run against different engines at the same time.
        """
        raw = self._call(
            "run_rollout",
            split=split,
            index=index,
            llm_url=llm_url,
            model=model,
            sandbox=sandbox,
            agent_step_limit=agent_step_limit,
            agent_timeout_s=agent_timeout_s,
            require_tokens=require_tokens,
        )
        return DataAgentRolloutResult.model_validate(_as_json(raw))

    def capabilities(self) -> dict[str, Any]:
        """Usable sandboxes, splits, concurrency budget, and whether rollouts will be trainable.

        Worth calling before dispatching a training run. An engine that cannot return token ids
        produces rollouts that look completely normal and carry nothing to train on.
        """
        return _as_json(self._call("capabilities"))

    # --- internals ------------------------------------------------------------------------------

    def _call(self, name: str, **kwargs: Any) -> Any:
        """Call an MCP tool from synchronous code.

        `MCPToolClient.call_tool` cannot be used here. It is a coroutine that internally does
        `await self.step(action)`, but `EnvClient.step` dispatches on execution mode and returns a
        concrete `StepResult` in sync mode, so awaiting it raises `TypeError: object StepResult can't
        be used in 'await' expression`. Driving `step` directly works in both modes.
        """
        result = self.step(CallToolAction(tool_name=name, arguments=kwargs))
        if inspect.isawaitable(result):  # async mode returns an awaitable instead
            result = run_async_safely(result)

        observation = result.observation
        if isinstance(observation, CallToolObservation):
            if observation.error is not None:
                raise RuntimeError(
                    f"tool {name!r} failed: {observation.error.message} "
                    f"({observation.error.error_type.value})"
                )
            return observation.result
        return observation

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._http.post(path, json=body).raise_for_status().json()

    def close(self) -> None:
        try:
            self._http.close()
        finally:
            super().close()


def _as_json(raw: Any) -> Any:
    """MCP tool results arrive as text content; these tools return JSON strings."""
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    # A content-block list, e.g. [{"type": "text", "text": "..."}].
    if isinstance(raw, (tuple, list)) and raw:
        first = raw[0]
        text = (
            first.get("text")
            if isinstance(first, dict)
            else getattr(first, "text", None)
        )
        if text:
            return json.loads(text)
    raise TypeError(f"cannot read a tool result out of {type(raw).__name__}")
