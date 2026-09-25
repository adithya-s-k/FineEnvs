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

"""How the agent is configured inside the sandbox. Several of these are not preferences."""

from __future__ import annotations

from dataclasses import dataclass, field

from .sandbox import DEFAULT_IMAGE, sandbox_home


@dataclass
class DataAgentConfig:
    """Per-rollout agent configuration.

    Attributes:
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Backend name. `"e2b"`, `"hf"` or `"daytona"`; see `sandbox/__init__.py`.
        image (`str`, *optional*):
            Container image carrying pandas/numpy/scipy and friends, which the instruction
            promises. USED BY THE `hf` AND `daytona` BACKENDS -- E2B has no image parameter; it carries the
            equivalent inside its prebuilt template (see `sandbox/__init__.py`).
        agent_timeout_s (`float`, *optional*, defaults to `600.0`):
            Wall clock for one rollout before it is abandoned and scored on whatever it filed.
        agent_step_limit (`int`, *optional*, defaults to `10`):
            Hard cap on MODEL CALLS, enforced host-side in the capture proxy.
        max_output_tokens (`int`, *optional*, defaults to `4096`):
            Output cap per model call. The hosted Task API uses4096for test and16384for train.
        setup_timeout_s (`float`, *optional*, defaults to `600.0`):
            Wall clock for staging the task's tables, separate from the agent's own budget.
        install_timeout_s (`float`, *optional*, defaults to `300.0`):
            Wall clock for installing opencode when the image does not ship it (`hf`).
        disabled_tools (`list[str]`, *optional*):
            Tools removed from the agent. Web access and sub-agents make a rollout unreproducible
            and are off by default.
    """

    sandbox: str = "e2b"
    image: str = DEFAULT_IMAGE
    agent_timeout_s: float = 600.0
    agent_step_limit: int = 10
    max_output_tokens: int = 4096
    # Staging pulls a bucket that can reach gigabytes; the agent's own budget is a different
    # clock and must not be spent on it.
    setup_timeout_s: float = 600.0
    # nvm + a ~50 MB Node tarball + `npm i -g`. Only the E2B template skips it.
    install_timeout_s: float = 300.0
    disabled_tools: list[str] = field(
        default_factory=lambda: ["webfetch", "question", "task"]
    )

    @property
    def home(self) -> str:
        """Sandbox home for the chosen backend. Never hardcode this; see `sandbox/__init__.py`."""
        return sandbox_home(self.sandbox)

    def opencode_settings(self) -> dict:
        """Settings written into the agent's config inside the sandbox.

        `permission.external_directory = "allow"` IS NOT OPTIONAL. opencode runs with its cwd at
        `{home}/workdir` and treats `/home/user/input/*` and `/workdir/answer.txt` as external, so
        without it the agent auto-rejects reading its own data and writing its own answer. That one
        block is the entire explanation for a measured train/eval gap on the same model: 12.7 tool
        calls and pass@1 0.514 under eval, against 1.3 calls and zero solves under training, with 79
        of 95 rollouts making exactly one model call before giving up.

        `agent.build.steps` is deliberately absent. It caps nothing -- measured on opencode 1.18.30
        against a fake engine, `steps=3`, `maxSteps=3` and no cap all produced 61 model calls. The
        only component that sees every call is the capture proxy, which is where `agent_step_limit`
        is enforced instead.

        TOOLS ARE DISABLED VIA THE `tools` MAP, NOT A `disabled_tools` LIST. opencode's schema has no
        `disabled_tools` key, so an earlier revision that emitted one had every tool ENABLED while the
        config read as if three were off. `task` spawns subagents, and a subagent is a separate
        conversation: capture flagged `multiple_roots` on 136 of 576 rollouts (24%) naming "subagent"
        first, which breaks the prefix chain the trainer needs (`rollout/fork_frac` 0.02-0.06 against
        a reference's flat 0, `drift_tokens_max` 32,770). `webfetch` additionally gives a sandbox with
        no egress a tool that can only fail, and every failure is error text in a tool result.
        """
        return {
            "permission": {
                "external_directory": "allow",
                "bash": "allow",
                "edit": "allow",
            },
            # {name: False} is the shape opencode actually reads; see the docstring.
            "tools": {tool: False for tool in self.disabled_tools},
        }
