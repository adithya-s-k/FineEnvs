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

"""The server-side environment: task selection over MCP, and one long `run_rollout`.

EVERY PIECE OF STATE HERE IS CLASS- OR MODULE-LEVEL, AND THAT IS NOT STYLE
OpenEnv builds a FRESH environment instance per request and closes it in a `finally`
(`http_server.py:1082-1097`). `/metadata` and `/schema` build one too. So anything cached on `self`
is rebuilt per call -- for this env that would mean re-downloading and re-parsing a dataset of
thousands of rows on every `num_tasks()` -- and any credential read in `__init__` is read on the docs
page. Config arrives through `configure()` onto the class; the dataset cache lives in `tasks.py` at
module scope.

`run_rollout` RATHER THAN `reset`/`step`
No state survives between requests, and the agent owns its own loop: it runs inside a sandbox with
its own tools and its own turn structure, and there is no meaningful intermediate state for a caller
to observe. `reset()` is selection only and boots nothing.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation

from ..config import DataAgentConfig
from ..models import DataAgentState
from ..tasks import DataAgentTaskProvider


logger = logging.getLogger(__name__)

# A rollout is sandbox boot, data staging, the agent's whole tool loop, then grading: 60-600 s is
# normal. MCP tools default to 30 s, which would abort every healthy rollout, so `step` is shadowed
# below to raise the floor.
ROLLOUT_TIMEOUT_S = 1800.0


class DataAgentEnvironment(MCPEnvironment):
    """Per-session environment exposing `run_rollout`, `capabilities` and `list_tasks` over MCP."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Server-wide, set once by `app.py`. Class-level for the reason in the module docstring.
    _splits: list[str] = []
    _llm_url: str = ""
    _model: str = ""
    _hf_token: str | None = None
    _sandbox: str = "e2b"

    @classmethod
    def configure(
        cls,
        *,
        splits: list[str],
        llm_url: str = "",
        model: str = "",
        hf_token: str | None = None,
        sandbox: str = "e2b",
    ) -> None:
        cls._splits = list(splits)
        cls._llm_url = llm_url
        cls._model = model
        # Resolved ONCE at startup, never per rollout: it is used to stage each task's tables, and
        # re-resolving it per rollout would hit the token store thousands of times per run.
        cls._hf_token = hf_token
        cls._sandbox = sandbox

    def __init__(self) -> None:
        from fastmcp import FastMCP

        self._provider = DataAgentTaskProvider(self._splits)
        self._state = DataAgentState(episode_id=str(uuid4()))

        mcp = FastMCP("data_agent_env")

        @mcp.tool
        def run_rollout(
            split: str = "",
            index: int = 0,
            llm_url: str = "",
            model: str = "",
            sandbox: str = "",
            agent_step_limit: int = 10,
            agent_timeout_s: float = 600.0,
            require_tokens: bool = True,
        ) -> str:
            """Run one data-agent rollout and return a JSON `DataAgentRolloutResult`.

            The ENGINE is per call, not per deployment. A dataset and its prebuilt sandbox templates
            are the expensive things to host and an engine restarts every training run, so one server
            serves a training run and an evaluation run against different engines at once.

            `require_tokens` refuses an engine that cannot return token ids. Leave it on for training
            -- such a rollout looks completely normal and carries nothing to train on -- and turn it
            off for evaluation, where a text-only endpoint is a perfectly good backend.
            """
            return self._run_rollout(
                split or (self._splits[0] if self._splits else ""),
                index,
                llm_url,
                model,
                sandbox,
                agent_step_limit,
                agent_timeout_s,
                require_tokens,
            )

        @mcp.tool
        def capabilities() -> str:
            """Usable sandboxes, splits, concurrency budget, and whether rollouts can be trained on."""
            return json.dumps(self._capabilities())

        @mcp.tool
        def list_tasks(split: str = "", start: int = 0, stop: int = 20) -> str:
            """A window of tasks in a split, for browsing without pulling all of them."""
            return json.dumps(self._provider.get_task_range(split, start, stop))

        super().__init__(mcp)

    # --- Task API (OpenEnv discovers these by duck typing) --------------------------------------

    def list_splits(self) -> list[dict[str, Any]]:
        return self._provider.list_splits()

    def num_tasks(self, split: str) -> int:
        return self._provider.num_tasks(split)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self._provider.list_tasks(split)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self._provider.get_task(split, index)

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        return self._provider.get_task_range(split, start, stop)

    # --- Environment ---------------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str = "",
        index: int = 0,
        **_: Any,
    ) -> Observation:
        """Select a task. Boots nothing -- a sandbox is created per `run_rollout`.

        The selection is ECHOED BACK, and a caller should assert on it. OpenEnv's reset body is
        `extra="allow"` (`http_server.py:682-686`), so a misspelled kwarg is DROPPED rather than
        rejected: send `task_index=` where this declares `index=` and every rollout silently runs
        index 0, with nothing anywhere reporting a problem.
        """
        self._state = DataAgentState(
            episode_id=episode_id or str(uuid4()),
            split=split or (self._splits[0] if self._splits else None),
            index=index,
        )
        task_name = ""
        if self._state.split is not None:
            try:
                task_name = self._provider.get_task(self._state.split, index).get(
                    "task_id", ""
                )
            except Exception:  # noqa: BLE001 -- an out-of-range index is the caller's to see, not a crash
                logger.warning(
                    "reset could not resolve %s[%d]", self._state.split, index
                )
        self._state.task_id = task_name or None
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "split": self._state.split,
                "index": index,
                "task_id": task_name,
                "message": "Call run_rollout(split=..., index=..., llm_url=..., model=...)",
            },
        )

    def _step_impl(
        self, action: Any, timeout_s: float | None = None, **_: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action {type(action).__name__}; "
                "use CallToolAction(name='run_rollout', ...)"
            },
        )

    def step(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return super().step(action, timeout_s=timeout_s or ROLLOUT_TIMEOUT_S, **kwargs)

    async def step_async(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return await super().step_async(
            action, timeout_s=timeout_s or ROLLOUT_TIMEOUT_S, **kwargs
        )

    @property
    def state(self) -> DataAgentState:
        return self._state

    # --- internals -----------------------------------------------------------------------------

    def _capabilities(self) -> dict[str, Any]:
        from .rollout import concurrency_status
        from ..sandbox import available, BACKENDS

        return {
            "env": "data_agent_env",
            "splits": self._provider.list_splits(),
            "sandboxes": {"supported": list(BACKENDS), "usable": available()},
            "llm_url": self._llm_url,
            "model": self._model,
            "concurrency": concurrency_status(),
        }

    def _run_rollout(
        self,
        split: str,
        index: int,
        llm_url: str,
        model: str,
        sandbox: str,
        agent_step_limit: int,
        agent_timeout_s: float,
        require_tokens: bool,
    ) -> str:
        from ..tasks import task_at
        from .rollout import run_rollout

        # `task_at`, NOT `provider.get_task`. The latter returns the PUBLIC projection, which omits
        # the gold answer on purpose -- a task spec travels to whoever asks, including the agent's
        # own side of the wire. Grading needs the gold, so the rollout path takes the full task.
        task = task_at(split, index)
        config = DataAgentConfig(
            sandbox=sandbox or self._sandbox,
            agent_step_limit=agent_step_limit,
            agent_timeout_s=agent_timeout_s,
        )
        result = run_rollout(
            task,
            llm_url=llm_url or self._llm_url,
            model=model or self._model,
            hf_token=self._hf_token,
            config=config,
            require_tokens=require_tokens,
        )
        self._state.rollout_type = result.rollout_type
        self._state.split = split
        self._state.index = index
        self._state.task_id = task.instruction_id
        return result.model_dump_json()
