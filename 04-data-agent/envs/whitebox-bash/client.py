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

"""The trainer-side class. Its methods ARE the model's tools.

HOW TRL SEES THIS
`GRPOTrainer(environment_factory=...)` calls the factory once per rollout and then introspects the
instance (`trl/trainer/grpo_trainer.py`):

    for member_name, member in inspect.getmembers(instance, predicate=inspect.ismethod):
        if member_name == "reset":        has_reset = True
        elif member_name == "get_reward": has_reward = True
        elif not member_name.startswith("_"): methods.append(member)

So `reset` and `get_reward` are special, everything else public becomes a tool, and anything private
is invisible. Three consequences shape this file:

  * TOOL SELECTION CANNOT BE A RUNTIME FLAG. If every tool were a method on one class, TRL would put
    all of them in the schema no matter which toolsets were requested, and the model would call a
    tool the server does not serve. The surface therefore has to vary by TYPE, which is why the tools
    live on mixins and `white_box_bash_env()` composes a class from the selection.
  * EVERY HELPER MUST BE `_`-PREFIXED, or it silently becomes a tool the model can call.
  * Type hints and docstrings are not documentation here, they are the tool schema the model reads.
    Write them for the model.

WHY THIS IS WHITE BOX
The trainer owns the loop: it generates, parses the tool call, invokes one of these methods, appends
the result, and generates again. TRL masks the tool-result tokens itself, so the policy is never
trained on text it did not write. That is the thing the black-box path had to reconstruct from a
capture proxy, and it is why this environment needs no proxy, no tunnel and no token-id contract.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import os
import threading
from typing import Any

from .models import EpisodeStart, Grade, ToolResult
from .tools import SUBMIT, TOOLSETS, resolve


logger = logging.getLogger(__name__)

DEFAULT_SERVER = os.environ.get("WHITE_BOX_BASH_URL", "http://127.0.0.1:8000")
ENV_NAME = "white_box_bash"


class _SyncMCP:
    """Blocking wrapper over OpenEnv's async MCP client.

    TRL calls tools synchronously, from worker threads, and may itself be inside a running event
    loop. `asyncio.run` would raise in that case, so the loop lives on its own daemon thread and every
    call is handed to it with `run_coroutine_threadsafe`. One loop per instance: instances are pooled
    per rollout and must not share a session.

    TRANSPORT IS HTTP `/mcp`, NOT THE WEBSOCKET -- see `_ensure`. The rest of this note explains why
    the WebSocket knobs are still set: they apply if that transport is ever re-enabled.

    KEEPALIVE PINGS ARE DISABLED, DELIBERATELY
    The transport is a WebSocket whose default keepalive is a 20 s ping with a 20 s timeout. That
    event loop is a Python thread inside the TRAINING process, so it cannot answer a ping while the
    main thread holds the GIL through CUDA-graph capture, compilation or a long generation -- all of
    which routinely exceed 20 s. The server then closes the connection cleanly and the next call dies
    with `ConnectionClosedOK: received 1000 (OK)`, which is what killed the first smoke at step 0.

    Liveness is not lost by turning pings off: every call carries its own timeout, so a genuinely
    dead server surfaces there instead. What is lost is *early* detection of a dead peer while idle,
    which is worth trading away -- the alternative is a healthy server being declared dead because the
    trainer was busy, and this project has already lost a night to exactly that inversion.

    Calls also retry ONCE on a closed connection, because a connection can still be dropped for
    reasons unrelated to pings and a fresh session is cheap.
    """

    def __init__(self, base_url: str, timeout_s: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _ensure(self) -> None:
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=self._run_loop, args=(loop,), daemon=True,
                                      name="white-box-bash-mcp")
            thread.start()
            self._loop, self._thread = loop, thread
        if self._client is None:
            from openenv.core.mcp_client import MCPToolClient

            client = MCPToolClient(
                base_url=self._base_url,
                websocket_ping_interval_s=None,   # see the class docstring
                websocket_ping_timeout_s=None,
            )
            # HTTP `/mcp`, NOT the WebSocket. Measured: the server accepts a second concurrent
            # WebSocket and then immediately closes it (`ConnectionClosedOK: received 1000 (OK)`),
            # so the first client keeps working and every later one dies on its first call. TRL
            # builds one environment instance per batch slot, so with `num_generations=4` that is
            # three dead clients out of four -- it killed the smoke at step 0, twice.
            #
            # HTTP suits this environment better anyway: the episode id already travels in the
            # payload as `session_id`, so nothing needs a per-connection session, and there is no
            # long-lived socket for a GIL-blocked event loop to fail to keep alive.
            client.use_production_mode = True
            self._client = client

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def call(self, name: str, **kwargs: Any) -> Any:
        for attempt in (0, 1):
            self._ensure()
            try:
                fut: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
                    self._client.call_tool(name, **kwargs), self._loop  # type: ignore[arg-type]
                )
                return fut.result(timeout=self._timeout_s)
            except Exception as exc:
                closed = "ConnectionClosed" in type(exc).__name__ or "closed" in str(exc).lower()
                if attempt == 0 and closed:
                    logger.warning("MCP connection closed on %s; reconnecting once", name)
                    self._client = None      # force a fresh session on the retry
                    continue
                raise

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
            self._client = None


class _Base:
    """Lifecycle and transport. Deliberately holds no tool methods."""

    def __init__(
        self,
        base_url: str = DEFAULT_SERVER,
        *,
        split: str = "train",
        index: int | None = None,
        toolsets: str | list[str] | None = None,
        step_limit: int = 20,
        timeout_s: float = 600.0,
    ) -> None:
        self._mcp = _SyncMCP(base_url, timeout_s=timeout_s)
        self._split = split
        self._index = index
        self._toolsets = resolve(toolsets)
        self._step_limit = step_limit
        self._session: str | None = None
        self._steps = 0
        self._submitted: str | None = None
        self._reward: float | None = None

    # --- TRL lifecycle -------------------------------------------------------------------------
    def reset(self, **kwargs: Any) -> str | None:
        """Start an episode and return the task text.

        TRL appends the returned string to the last user message, so this is where the task statement
        reaches the model. Returning `None` would leave the model with the system prompt alone and no
        task -- which reads as a policy that will not engage, so an empty task is an error here, not a
        silent no-op.
        """
        split = kwargs.pop("split", self._split)
        index = kwargs.pop("index", self._index)
        self._steps = 0
        self._submitted = None
        self._reward = None
        payload = self._mcp.call(
            "start_episode", split=split, index=index, toolsets=list(self._toolsets)
        )
        start = EpisodeStart.from_payload(_as_dict(payload))
        self._session = start.session_id
        prompt = start.prompt
        if not prompt:
            raise RuntimeError(
                f"environment returned no task text for split={split!r} index={index!r}; "
                "training on this would be training on an empty task"
            )
        return prompt

    def get_reward(self) -> float:
        """Score the episode from what the agent DID, not only from what it said.

        Called once per completed rollout. Grading happens server-side, where the sandbox and the
        gold answer live; the client only carries the number back.
        """
        if self._reward is not None:
            return self._reward
        verdict = Grade.from_payload(_as_dict(self._mcp.call("grade", session_id=self._session)))
        # TRL's reward column is a float, so an UNGRADED episode has to become a number here. 0.0 is
        # the honest choice at this boundary -- but note it is a LOSS of information the server had,
        # and `verdict.note` says which episodes it happened to.
        if verdict.ungraded:
            logger.warning("episode ungraded (%s); reporting 0.0 to the trainer", verdict.note)
        self._reward = 0.0 if verdict.reward is None else verdict.reward
        return self._reward

    # --- internals (must stay `_`-prefixed or they become tools) -------------------------------
    def _invoke(self, tool: str, _counts: bool = True, **kwargs: Any) -> str:
        """Call one tool.

        `_counts=False` exempts a tool from the step budget. Only the terminator uses it, and it must:
        the budget message tells the agent to submit, so if submitting were itself blocked the advice
        would be impossible to follow and every capped episode would score 0.0 -- indistinguishable
        from a policy that cannot solve the task. Caught by a live rollout, not by any unit test.
        """
        if self._session is None:
            return "[error] no episode; the trainer must call reset() first"
        if _counts and self._steps >= self._step_limit:
            # A budget the MODEL can see. Returning an error string rather than raising keeps the
            # rollout alive and lets the policy learn to submit before it runs out, which is the
            # behaviour we actually want to reinforce.
            return (f"[error] step budget of {self._step_limit} exhausted; "
                    f"call submit_solution with your best answer")
        if _counts:
            self._steps += 1
        try:
            payload = self._mcp.call(tool, session_id=self._session, **kwargs)
        except Exception as exc:  # a dead sandbox must not kill the whole batch
            logger.warning("tool %s failed: %s", tool, exc)
            return f"[error] {type(exc).__name__}: {exc}"
        return ToolResult.from_payload(_as_dict(payload)).output


def _as_dict(payload: Any) -> dict[str, Any]:
    """Unwrap an MCP result into a plain dict.

    MCP nests the payload twice -- `structured_content.result`, or a JSON string in `content[0].text`
    -- and which one arrives depends on the server's return annotation. Handling only one layer is a
    real bug: the call succeeds, the dict comes back empty, and the rollout reads as a tool that
    returned nothing.
    """
    import json

    if payload is None:
        return {}
    if isinstance(payload, dict):
        for key in ("structured_content", "structuredContent"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return inner.get("result", inner) if "result" in inner else inner
        content = payload.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("text") if isinstance(content[0], dict) else None
            if text:
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    return {"output": text}
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except (ValueError, TypeError):
            return {"output": payload}
    return {"output": str(payload)}


# --- the tool mixins ---------------------------------------------------------------------------
# One class per toolset. `white_box_bash_env` composes only the selected ones, so the model's schema
# and the server's registered tools are the same set by construction.

class _BashTools:
    def bash(self, command: str) -> str:
        """Run a shell command in the working directory and return its output.

        Each call is a fresh shell, so `cd` does not persist between calls; use absolute paths or
        chain with `&&`.

        Args:
            command: The shell command to run.
        """
        return self._invoke("bash", command=command)


class _SetaTools:
    def read(self, path: str) -> str:
        """Read a file and return its contents.

        Args:
            path: Path to the file, absolute or relative to the working directory.
        """
        return self._invoke("read", path=path)

    def write(self, path: str, content: str) -> str:
        """Write content to a file, creating or overwriting it.

        Args:
            path: Path to write to.
            content: The full contents to write.
        """
        return self._invoke("write", path=path, content=content)

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace the first exact occurrence of `old` with `new` in a file.

        Fails if `old` does not appear, so read the file first to get the text exactly right.

        Args:
            path: Path to the file to edit.
            old: The exact text to replace.
            new: The replacement text.
        """
        return self._invoke("edit", path=path, old=old, new=new)

    def grep(self, pattern: str, path: str) -> str:
        """Search files for a regular expression and return matching lines with their paths.

        Args:
            pattern: A regular expression.
            path: File or directory to search.
        """
        return self._invoke("grep", pattern=pattern, path=path)

    def glob(self, pattern: str) -> str:
        """List paths matching a glob pattern, one per line.

        Args:
            pattern: A glob such as `**/*.py`.
        """
        return self._invoke("glob", pattern=pattern)

    def ls(self, path: str) -> str:
        """List the entries of a directory.

        Args:
            path: Directory to list.
        """
        return self._invoke("ls", path=path)


class _SubmitTool:
    # `answer` accepts a number as well as a string, and that is not laziness.
    # The type hint IS the schema the model sees, and pydantic validates the model's tool call
    # against it. Many tasks end in "submit just the number", so the model emits a bare `3` -- and an
    # `answer: str` annotation rejects it with
    #   Input should be a valid string [input_value=3, input_type=int]
    # Every numeric answer then fails to submit and the episode scores 0.0, which is indistinguishable
    # from a model that could not finish. Observed on the very first smoke step. Coerced to `str`
    # immediately so the grader still sees one type.
    def submit_solution(self, answer: str | int | float) -> str:
        """Submit your final answer and end the episode.

        Call this once you are confident. The episode is graded on this answer together with what you
        did to reach it.

        Args:
            answer: The final answer, as a string or a number.
        """
        answer = str(answer)
        self._submitted = answer
        # Exempt from the budget on purpose -- see `_invoke`.
        return self._invoke("submit_solution", _counts=False, answer=answer)


_MIXINS: dict[str, type] = {
    "bash": _BashTools,
    "seta": _SetaTools,
}


def white_box_bash_env(
    base_url: str = DEFAULT_SERVER,
    *,
    split: str = "train",
    index: int | None = None,
    toolsets: str | list[str] | None = None,
    step_limit: int = 20,
    timeout_s: float = 600.0,
):
    """Build the `environment_factory` callable TRL expects.

    Returns a zero-argument callable producing instances whose public methods are exactly the tools
    for `toolsets` -- so the model's schema cannot drift from what the server serves.

    Args:
        base_url (`str`, *optional*):
            The hosted environment server, e.g. an HF Space URL.
        toolsets (`str` or `list[str]`, *optional*):
            `"bash"`, `"seta"`, a comma-separated string, or `"all"`. `"bash"` is always
            included. Defaults to `("bash", "seta")` -- full SETA parity.
        step_limit (`int`, *optional*, defaults to `20`):
            Tool calls per episode. Enforced client-side and reported to the model as an error string
            rather than an exception, so the policy can learn to submit before exhausting it.

    Returns:
        `Callable[[], object]`: pass directly as `environment_factory`.

    Examples:

    ```python
    from trl import GRPOTrainer
    from white_box_bash import white_box_bash_env

    trainer = GRPOTrainer(
        model="Qwen/Qwen3.5-2B",
        args=config,
        train_dataset=dataset,
        environment_factory=white_box_bash_env("https://my-space.hf.space"),
    )
    ```
    """
    names = resolve(toolsets)
    bases = tuple(_MIXINS[n] for n in names) + (_SubmitTool, _Base)
    cls = type("WhiteBoxBashEnv", bases, {"__doc__": _Base.__doc__})

    def factory() -> Any:
        return cls(
            base_url,
            split=split,
            index=index,
            toolsets=names,
            step_limit=step_limit,
            timeout_s=timeout_s,
        )

    return factory


def exposed_tool_names(toolsets: str | list[str] | None = None) -> tuple[str, ...]:
    """The tool names TRL will actually expose, derived the same way TRL derives them.

    Used by the consistency check in `tests/` and useful in a smoke script: if this does not match the
    server's registered tools, the run is already broken and nothing downstream will say so.
    """
    factory = white_box_bash_env(toolsets=toolsets)
    instance = factory()
    return tuple(
        sorted(
            name
            for name, member in inspect.getmembers(instance, predicate=inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        )
    )
