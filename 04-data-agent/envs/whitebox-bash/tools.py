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

WHY TWO TOOLSETS
`bash` alone is a minimal terminal agent: one tool, one way to do anything. `seta` adds the surface
SETA evaluates against -- `read`/`write`/`edit`/`grep`/`glob`/`ls` -- and an agent that can `grep` a
repository behaves very differently from one reduced to `bash` heredocs.

There is deliberately NO second execution model. An earlier revision also offered a persistent
Jupyter kernel, which meant two tools could do the same job under different state semantics (kernel
names persist, shell state does not) -- an easy thing for a small model to conflate, and one more
unvalidated variable in an environment that has not trained yet.

Both toolsets run on ONE sandbox and share its filesystem, which is why they mix freely -- see
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

# The SETA surface. Deliberately the same names SETA uses, so a task written against SETA reads the
# same here -- that is what makes its suite portable later without rewriting every task prompt.
SETA: tuple[ToolSpec, ...] = (
    ToolSpec("read", ("path",), "Read a file and return its contents."),
    ToolSpec("write", ("path", "content"), "Write content to a file, creating or overwriting it."),
    ToolSpec("edit", ("path", "old", "new"), "Replace the first exact occurrence of `old` with `new`."),
    ToolSpec("grep", ("pattern", "path"), "Search files for a regular expression."),
    ToolSpec("glob", ("pattern",), "List paths matching a glob pattern."),
    ToolSpec("ls", ("path",), "List a directory."),
)

# Always present, whatever the toolset, and named as SETA names it.
#
# Always present, because the episode needs a way to end deliberately -- without it the only
# terminator is the step cap, and a capped episode is indistinguishable from a stuck one. It is NOT
# folded into `seta` for that reason: a `bash`-only agent would otherwise have no way to finish.
#
# Named `submit_solution` rather than `submit` for SETA parity, so a task written against SETA reads
# unchanged here. The alternative -- `submit` for bash-only and `submit_solution` with `seta` -- would
# make the terminator's NAME depend on the selection, which is worse than either name alone.
SUBMIT: tuple[ToolSpec, ...] = (
    ToolSpec("submit_solution", ("answer",), "Submit the final answer and end the episode."),
)

TOOLSETS: dict[str, tuple[ToolSpec, ...]] = {
    "bash": BASH,
    "seta": SETA,
}

# What a caller gets by asking for nothing: full SETA parity.
DEFAULT_TOOLSETS: tuple[str, ...] = ("bash", "seta")


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
