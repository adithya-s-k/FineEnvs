"""
Jupyter Agent SkyRL Gym Environment.

Properly uses the SkyRL Gym library:
- Subclasses skyrl_gym.envs.base_text_env.BaseTextEnv
- Implements init(), step(), close() following the SkyRL pattern
- Returns BaseTextEnvStepOutput from step()
- Registered as "jupyter:JupyterAgent-v0" in SkyRL's registry

This module provides:
1. JupyterSkyRLEnv — proper BaseTextEnv subclass (for native SkyRL training)
2. JupyterToolkit is in verifiers/env.py (shared, used by TRL adapter)

Usage with native SkyRL:
    import skyrl_gym
    env = skyrl_gym.make("jupyter:JupyterAgent-v0")
    obs, info = env.init([{"role": "user", "content": "Print 42"}])
    result = env.step("<code>print(42)</code>")

Usage with TRL (via adapter):
    # Adapter uses JupyterToolkit from verifiers/env.py (shared)
    trainer = GRPOTrainer(
        environment_factory=SkyRLEnvironment,
        environment_config={},
    )
"""

import sys
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:
    # Fallback for when skyrl-gym isn't installed
    from dataclasses import dataclass

    class BaseTextEnv:
        def init(self, prompt): return prompt, {}
        def step(self, action): raise NotImplementedError
        def close(self): pass

    @dataclass
    class BaseTextEnvStepOutput:
        observations: list
        reward: float
        done: bool
        metadata: dict
        postprocessed_action: str = None

_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from core.e2b_sandbox import E2BSandbox, CellResult  # noqa: E402
from core.notebook_tracker import NotebookTracker  # noqa: E402
from core.tasks import TASKS, matches_expected  # noqa: E402

load_dotenv()


class JupyterSkyRLEnv(BaseTextEnv):
    """SkyRL Gym BaseTextEnv for Jupyter notebook interaction.

    The model sends free text via step(). The env parses tool calls
    from <code>/<shell>/<edit> tags and executes in E2B sandbox.

    This is the native SkyRL pattern — for TRL, the adapter uses
    JupyterToolkit with structured tool calling instead.
    """

    def __init__(self, expected_output: str = "", max_turns: int = 10):
        super().__init__()
        self._api_key = os.environ.get("E2B_API_KEY", "")
        self._expected_output = expected_output
        self.max_turns = max_turns
        self._sandbox: Optional[E2BSandbox] = None
        self._tracker = NotebookTracker()
        self.last_output = ""
        self._last_code_success = False
        self.error_count = 0

    def init(self, prompt) -> Tuple[Any, Dict]:
        """Initialize episode. Creates fresh E2B sandbox."""
        if self._sandbox:
            self._sandbox.kill()
        self._sandbox = E2BSandbox(api_key=self._api_key)
        self._tracker.reset()
        self.last_output = ""
        self._last_code_success = False
        self.turns = 0
        self.error_count = 0
        return prompt, {"max_turns": self.max_turns}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process model text — parse <code>/<shell>/<edit> tags, execute in sandbox."""
        self.turns += 1
        results = []

        # Parse tool calls
        code_matches = re.findall(r"<code>(.*?)</code>", action, re.DOTALL)
        shell_matches = re.findall(r"<shell>(.*?)</shell>", action, re.DOTALL)
        edit_matches = re.findall(r"<edit>(.*?)</edit>", action, re.DOTALL)

        for code in code_matches:
            result = self._sandbox.run_code(code.strip())
            self._tracker.add_code_cell(code.strip(), result)
            text = _format_for_llm(result)
            self.last_output = text
            self._last_code_success = result.success
            if not result.success:
                self.error_count += 1
            results.append(f"[Code]: {text}")

        for cmd in shell_matches:
            result = self._sandbox.run_shell(cmd.strip())
            self._tracker.add_shell_cell(cmd.strip(), result)
            shell_text = _format_for_llm(result)
            results.append(f"[Shell]: {shell_text}")

        for code in edit_matches:
            result = self._sandbox.run_code(code.strip())
            self._tracker.update_last_code_cell(code.strip(), result)
            self.last_output = _format_for_llm(result)
            self._last_code_success = result.success
            if not result.success:
                self.error_count += 1
            results.append(f"[Edit]: {self.last_output}")

        # Fallback: raw code
        if not results:
            code_block = re.search(r"```(?:python)?\s*\n(.*?)```", action, re.DOTALL)
            code = code_block.group(1).strip() if code_block else action.strip()
            if code and not code.startswith(("The ", "I ", "Based ", "Let me")):
                result = self._sandbox.run_code(code)
                self._tracker.add_code_cell(code, result)
                self.last_output = _format_for_llm(result)
                self._last_code_success = result.success
                if not result.success:
                    self.error_count += 1
                results.append(f"[Code]: {self.last_output}")

        reward = self._compute_reward()
        done = reward > 0 or self.turns >= self.max_turns

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": "\n".join(results) or "(no output)"}],
            reward=reward,
            done=done,
            metadata={"turns": self.turns, "errors": self.error_count},
        )

    def close(self):
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _compute_reward(self) -> float:
        if not self._expected_output or not self._last_code_success:
            return 0.0
        if matches_expected(self._expected_output, self.last_output):
            return 1.0
        return 0.0


# Register with SkyRL Gym registry
try:
    import skyrl_gym
    skyrl_gym.register(
        "jupyter:JupyterAgent-v0",
        JupyterSkyRLEnv,
    )
except (ImportError, Exception):
    pass


def _format_for_llm(result: CellResult) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.strip())
    if result.text_results:
        parts.extend(result.text_results)
    if result.images:
        parts.append(f"[Image: {len(result.images)} image(s)]")
    if result.error:
        parts.append(f"ERROR:\n{result.error}")
    return "\n".join(parts) if parts else "(no output)"
