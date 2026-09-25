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

"""One sandbox per episode, shared by every toolset.

WHY ONE SANDBOX AND NOT THREE
`bash` and the SETA file tools are two views of the SAME machine, and the agent expects them to
behave that way: a file written by `write` has to be visible to `bash` in the very next call. E2B
gives both off a single sandbox -- `commands.run` is a shell and `files.*` is the filesystem it sees
-- so keeping them together is not a convenience, it is the semantics.

Splitting them across sandboxes (or servers) would mean replicating state between them, and every
divergence would surface as an agent that wrote a file and then could not find it. That reads as a
model failure and is not one.

THE SHELL IS NOT PERSISTENT
Each `commands.run` is a fresh process, so `cd` does not carry between calls; the tool docstring says
so, because an agent that assumes otherwise will `cd` and then be baffled. A durable working
directory is carried explicitly in `cwd`.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

# Everything the agent does happens under here. Kept off `/` so a stray `rm -rf` in the workdir
# cannot take the interpreter with it, and so `glob`/`grep` have a bounded root to walk.
WORKDIR = "/home/user/work"

# Hard ceiling on any single tool result fed back to the model.
#
# 2,000 characters, NOT 8,000, and the difference decided whether the agent could take a second turn
# at all. Tool-result tokens live in TRL's `completion_ids` (masked out of the loss, and out of the
# `completions/mean_length` metric, but still occupying the budget), so they are bounded by
# `max_completion_length`. A `read` of a real data-agent CSV clipped at 8,000 chars is roughly 3,000
# tokens and exhausted a 1,024-token completion budget on the FIRST tool result -- leaving no room to
# generate turn two. Measured: `tools/call_frequency` pinned at 1.0 and reward 0 across every step,
# which reads exactly like a model that will not engage.
#
# Dumping raw CSV was never useful anyway: the agent should compute with pandas rather than read a
# 9.8 MB file into its context, and a tighter clip pushes it that way.
MAX_OUTPUT_CHARS = int(os.environ.get("WHITE_BOX_BASH_MAX_OUTPUT_CHARS", "1200"))


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Trim to `limit`, saying so, and keep BOTH ends.

    The head carries the command that ran and the tail carries the error; truncating either way round
    loses the half that diagnoses the failure.
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    dropped = len(text) - limit
    return f"{head}\n\n... [{dropped} characters omitted] ...\n\n{tail}"


@dataclass
class ExecResult:
    """What a tool call actually did. `ok` is derived, never guessed."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.error

    def render(self) -> str:
        """The string the model sees. Empty success is reported as such rather than as nothing."""
        parts = []
        if self.stdout.strip():
            parts.append(_clip(self.stdout.rstrip()))
        if self.stderr.strip():
            parts.append(f"[stderr]\n{_clip(self.stderr.rstrip())}")
        if self.error:
            parts.append(f"[error] {_clip(self.error, 1000)}")
        if not self.ok and self.exit_code:
            parts.append(f"[exit code {self.exit_code}]")
        return "\n".join(parts) if parts else "(no output)"


@dataclass
class Sandbox:
    """A live E2B sandbox plus the per-episode bookkeeping the grader needs.

    `calls` is the audit trail. A white-box environment's whole advantage over a black-box one is that
    every tool call is observable here rather than inferred from free text afterwards, so the reward
    can be a function of what the agent DID, not only of what it finally said.
    """

    handle: Any
    cwd: str = WORKDIR
    calls: list[dict[str, Any]] = field(default_factory=list)
    submitted: str | None = None

    # --- lifecycle -----------------------------------------------------------------------------
    # The SAME prebuilt E2B template the black-box environment uses. Not optional for data-agent
    # tasks: the default E2B base has no `huggingface_hub`, so the bucket staging cannot run, and
    # `/workdir` is not writable by `user`, so the setup dies on `mkdir: Permission denied` before
    # the agent starts. Both were measured. Sizing (cpu=2, mem=4096) is baked in at BUILD time and
    # cannot be set per sandbox, which is why this is a template name and not a set of kwargs.
    TEMPLATE = os.environ.get("E2B_TEMPLATE", "data-agent-opencode")

    @classmethod
    def start(cls, *, timeout_s: int = 900, envs: dict[str, str] | None = None) -> "Sandbox":
        # Plain `e2b`, not `e2b_code_interpreter`: with no Jupyter toolset there is no kernel to
        # drive, so the code-interpreter variant would be a heavier dependency for nothing.
        from e2b import Sandbox as E2BSandbox

        handle = E2BSandbox.create(template=cls.TEMPLATE, timeout=timeout_s, envs=envs or {})
        sb = cls(handle=handle)
        # `-p` so a re-reset against a warm template is not an error.
        sb.handle.commands.run(f"mkdir -p {shlex.quote(WORKDIR)}")
        logger.info("sandbox %s up (template %s), workdir %s",
                    getattr(handle, "sandbox_id", "?"), cls.TEMPLATE, WORKDIR)
        return sb

    def kill(self) -> None:
        try:
            self.handle.kill()
        except Exception:
            logger.warning("sandbox did not stop cleanly", exc_info=True)

    @property
    def sandbox_id(self) -> str:
        return str(getattr(self.handle, "sandbox_id", ""))

    # --- the three backends --------------------------------------------------------------------
    def bash(self, command: str, timeout_s: int = 120) -> ExecResult:
        """A fresh shell, rooted at `self.cwd`. Not persistent -- see the module docstring."""
        try:
            r = self.handle.commands.run(
                f"cd {shlex.quote(self.cwd)} && {command}", timeout=timeout_s
            )
        except Exception as exc:
            return ExecResult(error=f"{type(exc).__name__}: {exc}", exit_code=1)
        # `exit_code` of 0 is FALSY -- `getattr(r, "exit_code", 1) or 1` would read every success as a
        # failure. This bug cost a full eval on the data-agent env; check for None explicitly.
        code = getattr(r, "exit_code", None)
        return ExecResult(
            stdout=getattr(r, "stdout", "") or "",
            stderr=getattr(r, "stderr", "") or "",
            exit_code=0 if code is None else int(code),
        )

    def read_file(self, path: str) -> ExecResult:
        try:
            return ExecResult(stdout=self.handle.files.read(self._abs(path)))
        except Exception as exc:
            return ExecResult(error=f"{type(exc).__name__}: {exc}", exit_code=1)

    def write_file(self, path: str, content: str) -> ExecResult:
        try:
            self.handle.files.write(self._abs(path), content)
            return ExecResult(stdout=f"wrote {len(content)} bytes to {path}")
        except Exception as exc:
            return ExecResult(error=f"{type(exc).__name__}: {exc}", exit_code=1)

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else f"{self.cwd.rstrip('/')}/{path}"

    # --- audit ---------------------------------------------------------------------------------
    def record(self, tool: str, args: dict[str, Any], result: ExecResult) -> None:
        self.calls.append(
            {"tool": tool, "args": args, "ok": result.ok, "exit_code": result.exit_code}
        )
