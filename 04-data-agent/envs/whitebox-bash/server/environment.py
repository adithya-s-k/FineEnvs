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

"""The server. Registers every tool with FastMCP and owns the sandboxes.

SESSIONS LIVE AT MODULE SCOPE, NOT ON `self`
OpenEnv's HTTP server builds a throwaway environment instance per request and closes it in a
`finally` (`http_server.py`). Anything held on `self` therefore dies with the request that created
it -- a sandbox stored there would be leaked on every single call, and the next tool call would find
no session. The registry below is module-level and guarded by a lock for exactly that reason.

TOOLS TAKE A `session_id` FOR THE SAME REASON
There is no per-connection server state to hang an episode off, so the episode id travels in the
call. `start_episode` mints it; every other tool presents it.

WHY THIS IS THE WHITE-BOX HALF
Nothing here runs an agent. The trainer generates a tool call, TRL invokes the matching client
method, that arrives here as one MCP call, this module runs it in the sandbox and returns the
result. Every action is a separate, observable, individually-scoreable event -- which is what lets
`grade` reward what the agent DID rather than only what it finally said.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shlex
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

from .. import grader, tasks
from ..tools import resolve
from .sandbox import WORKDIR, ExecResult, Sandbox

# The reproducible Harbor catalog uses the same native grader on Daytona.
# Other task sources keep the original E2B backend.
if os.environ.get("WHITE_BOX_BASH_TASK_SOURCE") == "harbor-frozen":
    from daytona_whitebox_backend import DaytonaSandbox as Sandbox
    WORKDIR = "/workdir"


logger = logging.getLogger(__name__)

# How long a sandbox may live. An episode that outlives this is not salvageable, and leaving it
# running costs money in the sandbox provider rather than in this process, so it is bounded here.
SANDBOX_TIMEOUT_S = int(os.environ.get("WHITE_BOX_BASH_SANDBOX_TIMEOUT_S", "1800"))
MAX_SESSIONS = int(os.environ.get("WHITE_BOX_BASH_MAX_SESSIONS", "200"))


@dataclass
class Session:
    """One episode: its sandbox, its task, and what the agent has done so far."""

    session_id: str
    split: str
    index: int
    task: Any
    sandbox: Sandbox
    toolsets: tuple[str, ...]
    submitted: str | None = None
    alive: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)


_LOCK = threading.Lock()
_SESSIONS: dict[str, Session] = {}


def _get(session_id: str | None) -> Session:
    with _LOCK:
        s = _SESSIONS.get(str(session_id))
    if s is None:
        raise KeyError(
            f"no such session {session_id!r}; it may have been released after grading. "
            "Call start_episode first."
        )
    return s


def _release(session_id: str) -> None:
    """Drop a session and kill its sandbox.

    Not optional bookkeeping: a sandbox held past its episode keeps costing money and counts against
    the provider's concurrency cap, and leftovers collide with the next run's claim -- which presents
    as a burst of capacity errors on a server that looks idle.
    """
    with _LOCK:
        s = _SESSIONS.pop(session_id, None)
    if s is not None:
        s.sandbox.kill()


def _run(session_id: str, tool: str, fn, **args: Any) -> dict[str, Any]:
    """Invoke one tool, record it, and render the result for the model.

    A failing tool returns its error AS TEXT rather than raising: the agent should see `No such file`
    and adapt, which is a thing to learn. Raising would abort the rollout and score a recoverable
    mistake as an infrastructure failure.
    """
    s = _get(session_id)
    try:
        result: ExecResult = fn(s)
    except Exception as exc:
        logger.warning("tool %s raised for session %s", tool, session_id, exc_info=True)
        result = ExecResult(error=f"{type(exc).__name__}: {exc}", exit_code=1)
        # A sandbox that has gone away makes the whole episode ungradeable, and `grade` must be told
        # so it returns None rather than 0.0.
        if "sandbox" in str(exc).lower() or "timeout" in str(exc).lower():
            s.alive = False
    s.calls.append({"tool": tool, "args": args, "ok": result.ok})
    return {"output": result.render(), "ok": result.ok}


class WhiteBoxBashEnvironment(MCPEnvironment):
    """Bash and the SETA file tools over one sandbox, plus the Task API."""

    # The server refuses `max_concurrent_envs > 1` unless an environment asserts this, and the
    # assertion is true here for a specific reason: NO per-episode state is held on the instance.
    # Sessions live in the module-level `_SESSIONS` registry behind `_LOCK`, each owning its own
    # sandbox, and the instance itself is stateless -- which it has to be anyway, because the HTTP
    # server builds a throwaway instance per request.
    #
    # This matters for training: TRL creates one environment client per batch slot, so at
    # `num_generations=4` a cap of 1 refuses three of four. Do not set this on an environment that
    # keeps episode state on `self`.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        mcp = FastMCP("white_box_bash")

        # --- episode lifecycle -----------------------------------------------------------------
        @mcp.tool
        def start_episode(split: str = "train", index: int | None = None,
                          toolsets: list[str] | None = None) -> dict:
            """Create a sandbox, stage the task's inputs, and return the task text."""
            with _LOCK:
                if len(_SESSIONS) >= MAX_SESSIONS:
                    raise RuntimeError(
                        f"{MAX_SESSIONS} sessions already live; refusing to start another. "
                        "Lower the trainer's concurrency or raise WHITE_BOX_BASH_MAX_SESSIONS."
                    )
            idx = 0 if index is None else int(index)
            task = tasks.task_at(split, idx)
            # The task's own environment goes into the SANDBOX, not into the setup command text:
            # data-agent tasks stage their tables from an HF bucket and need HF_TOKEN, HF_BUCKET and
            # BUCKET_PREFIX there. Credentials travel by name through the process environment and are
            # never interpolated into a command string, so a token cannot reach a log line or a trace.
            sandbox_kwargs = {"task": task} if tasks.TASK_SOURCE == "harbor-frozen" else {}
            sb = Sandbox.start(timeout_s=SANDBOX_TIMEOUT_S,
                               envs=(task.metadata or {}).get("env") or {}, **sandbox_kwargs)
            if task.setup:
                # Staging failure is fatal to the episode: an agent asked about `data.csv` that was
                # never written will look like a model that cannot read a file.
                r = sb.bash(task.setup)
                if not r.ok:
                    sb.kill()
                    raise RuntimeError(f"task setup failed: {r.render()}")
            sid = uuid.uuid4().hex
            with _LOCK:
                _SESSIONS[sid] = Session(
                    session_id=sid, split=split, index=idx, task=task, sandbox=sb,
                    toolsets=resolve(toolsets),
                )
            return {"session_id": sid, "prompt": task.instruction,
                    "task_id": task.task_id, "workdir": WORKDIR}

        @mcp.tool
        def close_episode(session_id: str) -> dict:
            """Release an abandoned episode without inventing a grade."""
            _release(session_id)
            return {"closed": True}

        @mcp.tool
        def submit_solution(session_id: str, answer: str) -> dict:
            """Record the agent's final answer. Does not grade; `grade` does."""
            s = _get(session_id)
            s.submitted = answer
            s.calls.append({"tool": "submit_solution", "args": {"answer": answer}, "ok": True})
            return {"output": "submitted", "ok": True}

        @mcp.tool
        def grade(session_id: str) -> dict:
            """Score the episode and release its sandbox."""
            s = _get(session_id)
            if (s.task.metadata or {}).get('source') == 'harbor-frozen':
                try:
                    return s.sandbox.grade(s.submitted)
                finally:
                    _release(session_id)
            check_passed: bool | None = None
            if s.alive and s.task.check:
                check_passed = s.sandbox.bash(s.task.check).ok
            # Correctness comes from the BLACK-BOX environment's grader for data-agent tasks, so
            # both environments agree on what counts as right; the shaping below stays local.
            override = None
            if (s.task.metadata or {}).get("source") == "data-agent" and s.submitted is not None:
                from .. import dataagent

                override = dataagent.grade_answer(s.task, s.submitted)
            verdict = grader.grade(
                submitted=s.submitted,
                gold=s.task.answer,
                correct_override=override,
                # Submitting is bookkeeping, not work: counting it would charge the agent for
                # finishing, which is the one thing we want it to do.
                n_tool_calls=sum(1 for c in s.calls if c["tool"] != "submit_solution"),
                check_passed=check_passed,
                sandbox_alive=s.alive,
            )
            _release(session_id)
            return verdict.as_dict()

        # --- bash ------------------------------------------------------------------------------
        @mcp.tool
        def bash(session_id: str, command: str) -> dict:
            """Run a shell command in the working directory."""
            return _run(session_id, "bash", lambda s: s.sandbox.bash(command), command=command)

        # --- seta ------------------------------------------------------------------------------
        @mcp.tool
        def read(session_id: str, path: str) -> dict:
            """Read a file."""
            return _run(session_id, "read", lambda s: s.sandbox.read_file(path), path=path)

        @mcp.tool
        def write(session_id: str, path: str, content: str) -> dict:
            """Write a file."""
            return _run(session_id, "write", lambda s: s.sandbox.write_file(path, content),
                        path=path)

        @mcp.tool
        def edit(session_id: str, path: str, old: str, new: str) -> dict:
            """Replace the first exact occurrence of `old` with `new`."""

            def do(s: Session) -> ExecResult:
                cur = s.sandbox.read_file(path)
                if not cur.ok:
                    return cur
                if old not in cur.stdout:
                    # Loudly, with a hint. A silent no-op here is the worst outcome: the agent
                    # believes it edited the file and every later step reasons from a false premise.
                    return ExecResult(
                        error=f"`old` not found in {path}; read the file and match the text exactly",
                        exit_code=1,
                    )
                return s.sandbox.write_file(path, cur.stdout.replace(old, new, 1))

            return _run(session_id, "edit", do, path=path)

        @mcp.tool
        def grep(session_id: str, pattern: str, path: str) -> dict:
            """Search for a regular expression."""

            def do(s: Session) -> ExecResult:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    return ExecResult(error=f"bad regular expression: {exc}", exit_code=1)
                # -r so a directory works, -n for line numbers, -I to skip binaries. `|| true` keeps
                # "no matches" (grep's exit 1) from reading as a tool failure -- it is a valid result.
                return s.sandbox.bash(
                    f"grep -rnI -E {shlex.quote(pattern)} {shlex.quote(path)} || true"
                )

            return _run(session_id, "grep", do, pattern=pattern, path=path)

        @mcp.tool
        def glob(session_id: str, pattern: str) -> dict:
            """List paths matching a glob."""

            def do(s: Session) -> ExecResult:
                listing = s.sandbox.bash("find . -type f")
                if not listing.ok:
                    return listing
                names = [p[2:] if p.startswith("./") else p
                         for p in listing.stdout.splitlines() if p.strip()]
                hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
                return ExecResult(stdout="\n".join(hits) if hits else "(no matches)")

            return _run(session_id, "glob", do, pattern=pattern)

        @mcp.tool
        def ls(session_id: str, path: str = ".") -> dict:
            """List a directory."""
            return _run(session_id, "ls", lambda s: s.sandbox.bash(f"ls -la {shlex.quote(path)}"),
                        path=path)

        super().__init__(mcp)


    # --- OpenEnv Environment ABC ----------------------------------------------------------------
    # The MCP tools above are how the trainer actually drives this environment. These three exist
    # because `Environment` is abstract, and because a human poking the server with the standard
    # reset/step API should get something coherent rather than an AttributeError.

    def reset(self, split: str = "train", index: int | None = None,
              seed: int | None = None, episode_id: str | None = None,
              **kwargs: Any) -> Observation:
        """Start an episode through the Gym-style API and hand back its session id.

        The session id is in `metadata` rather than on `self` deliberately: the HTTP server discards
        this instance at the end of the request, so anything kept here would be gone by the next
        call. Selection is by `split`/`index` per the Task API -- a misspelled kwarg is silently
        dropped by the server's permissive model, so the chosen split and index are ECHOED BACK for
        the caller to assert on.
        """
        result = self.get_callables()["start_episode"](
            split=split, index=index, toolsets=list(kwargs.get("toolsets") or [])or None
        )
        return Observation(
            done=False,
            reward=None,
            metadata={"session_id": result["session_id"], "prompt": result["prompt"],
                      "task_id": result["task_id"], "split": split,
                      "index": 0 if index is None else int(index)},
        )

    def _step_impl(self, action: Action, timeout_s: float | None = None,
                   **kwargs: Any) -> Observation:
        """Fallback for non-MCP actions -- point the caller at the tools."""
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction; every capability here is an MCP tool."
            },
        )

    @property
    def state(self) -> dict:
        """Live session count. Per-episode state belongs to the session, not to this instance."""
        with _LOCK:
            return {"live_sessions": len(_SESSIONS), "max_sessions": MAX_SESSIONS}

    # --- Task API, declared structurally (no base class to inherit) -----------------------------
    def list_splits(self) -> list[dict]:
        return tasks.list_splits()

    def num_tasks(self, split: str) -> int:
        return tasks.num_tasks(split)

    def list_tasks(self, split: str) -> list[dict]:
        return tasks.list_tasks(split)

    def get_task(self, split: str, index: int) -> dict:
        return tasks.get_task(split, index)

    def get_task_range(self, split: str, start: int | None = None,
                       stop: int | None = None) -> list[dict]:
        return tasks.get_task_range(split, start, stop)
