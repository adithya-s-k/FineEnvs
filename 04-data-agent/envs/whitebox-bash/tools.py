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

"""The tool surface, declared ONCE.

Two consumers need to agree on exactly this list: the server, which registers each tool with FastMCP
so remote callers can invoke it, and the client, whose Python methods are what TRL's
`environment_factory` introspects into the model's tool schema. A surface duplicated across those two
files would drift, and the drift is silent -- the model is offered a tool the server does not
implement, calls it, gets an error, and the run reads as a policy that cannot use tools.

So this module is the single source of truth, and `client.py` asserts against it at import time.

WHY TOOLSETS RATHER THAN ONE FLAT LIST
`bash` alone is enough for most terminal work. Adding the Jupyter kernel matters when the task is
analysis and the agent wants names to persist between steps. The file tools (`read`/`write`/`edit`/
`grep`/`glob`/`ls`) are the SETA surface, and they are worth having as a group because an agent that
can `grep` a repository behaves very differently from one reduced to `bash` heredocs.

They all run on ONE sandbox and share its filesystem, which is why they can be mixed freely -- see
`server/sandbox.py` for why that is the semantics and not just a convenience.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    """One tool, as both sides must see it.

    Attributes:
        name (`str`):
            The tool name the model calls. Must match the client method name exactly.
        params (`tuple[str, ...]`):
            Parameter names, in order. Used only to check the client against this registry; the
            authoritative signature is the client method, because that is what TRL introspects.
        summary (`str`):
            One line, shown in the tool schema. Written for the MODEL, not for us.
    """

    name: str
    params: tuple[str, ...]
    summary: str


# --- the toolsets ------------------------------------------------------------------------------
# `bash` is always on: every toolset combination includes it, because an agent with file tools but no
# shell cannot run anything it writes, and that is never what the caller meant.
BASH: tuple[ToolSpec, ...] = (
    ToolSpec("bash", ("command",), "Run a shell command in the working directory."),
)

JUPYTER: tuple[ToolSpec, ...] = (
    ToolSpec("run_python", ("code",), "Run Python in a persistent Jupyter kernel; names persist."),
    ToolSpec("reset_kernel", (), "Restart the Python kernel, discarding all defined names."),
)

# The SETA surface. Deliberately the same names SETA uses, so a task written against SETA reads the
# same here -- that is what makes its suite portable later without rewriting every task prompt.
FILES: tuple[ToolSpec, ...] = (
    ToolSpec("read", ("path",), "Read a file and return its contents."),
    ToolSpec("write", ("path", "content"), "Write content to a file, creating or overwriting it."),
    ToolSpec("edit", ("path", "old", "new"), "Replace the first exact occurrence of `old` with `new`."),
    ToolSpec("grep", ("pattern", "path"), "Search files for a regular expression."),
    ToolSpec("glob", ("pattern",), "List paths matching a glob pattern."),
    ToolSpec("ls", ("path",), "List a directory."),
)

# Always present, whatever the toolset: the episode needs a way to end deliberately. Without it the
# only terminator is the step cap, and a capped episode is indistinguishable from a stuck one.
SUBMIT: tuple[ToolSpec, ...] = (
    ToolSpec("submit", ("answer",), "Submit the final answer and end the episode."),
)

TOOLSETS: dict[str, tuple[ToolSpec, ...]] = {
    "bash": BASH,
    "jupyter": JUPYTER,
    "files": FILES,
}

# What a caller gets by asking for nothing. `bash` + `files` is the terminal-agent shape SETA
# evaluates; the Jupyter kernel is opt-in because it only pays off on analysis tasks and it costs a
# kernel process per sandbox.
DEFAULT_TOOLSETS: tuple[str, ...] = ("bash", "files")


def resolve(toolsets: str | list[str] | None) -> tuple[str, ...]:
    """Normalise a toolset selection, failing loudly on an unknown name.

    A typo'd toolset must not silently degrade the agent to a smaller surface: that shows up as a
    policy that "stopped using grep", which is a very expensive thing to debug from metrics alone.

    Args:
        toolsets (`str` or `list[str]`, *optional*):
            Comma-separated string or list. `None` selects `DEFAULT_TOOLSETS`. `"all"` selects
            everything.

    Returns:
        `tuple[str, ...]`: Selected toolset names, always including `"bash"`.
    """
    if toolsets is None:
        names = list(DEFAULT_TOOLSETS)
    elif isinstance(toolsets, str):
        names = ["all"] if toolsets.strip() == "all" else [t.strip() for t in toolsets.split(",") if t.strip()]
    else:
        names = [str(t).strip() for t in toolsets if str(t).strip()]
    if names == ["all"]:
        names = list(TOOLSETS)
    unknown = [n for n in names if n not in TOOLSETS]
    if unknown:
        raise ValueError(f"unknown toolset(s) {unknown}; known: {sorted(TOOLSETS)}")
    if "bash" not in names:
        names.insert(0, "bash")
    return tuple(dict.fromkeys(names))


def specs_for(toolsets: str | list[str] | None) -> tuple[ToolSpec, ...]:
    """Every `ToolSpec` a selection exposes, `submit` included."""
    out: list[ToolSpec] = []
    for name in resolve(toolsets):
        out.extend(TOOLSETS[name])
    out.extend(SUBMIT)
    return tuple(out)


def tool_names(toolsets: str | list[str] | None) -> tuple[str, ...]:
    return tuple(s.name for s in specs_for(toolsets))
