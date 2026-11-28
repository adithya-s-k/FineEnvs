"""
Jupyter Agent GEM Environment.

A proper GEM (gem.Env) subclass implementing the Gymnasium-style
reset()/step() interface for Jupyter notebook interaction.

The model sends text actions which the environment parses for tool calls
using structured tags:
    <code>python code</code>     → add_and_execute_code_cell
    <shell>command</shell>        → execute_shell_command
    <edit>python code</edit>      → edit_and_execute_current_cell
    <state/>                      → get_notebook_state

If no tags found, the text is treated as raw Python code.

Usage:
    import gem
    from env import JupyterGemEnv

    # Register
    gem.register("jupyter:JupyterAgent-v0", JupyterGemEnv)

    # Use
    env = gem.make("jupyter:JupyterAgent-v0", task="Print 42", expected_output="42")
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step("<code>print(42)</code>")
"""

import sys
from pathlib import Path
import os
import re
import random
from typing import Any, Dict, Optional, SupportsFloat, Tuple

from dotenv import load_dotenv
try:
    from gem import Env
except ImportError:
    # Fallback: define a minimal Env base class for when gem-llm isn't installed
    import abc
    class Env(abc.ABC):
        @abc.abstractmethod
        def step(self, action): ...
        @abc.abstractmethod
        def reset(self, seed=None): ...
        def close(self): pass

_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from core.e2b_sandbox import E2BSandbox, CellResult  # noqa: E402
from core.notebook_tracker import NotebookTracker  # noqa: E402
from core.tasks import TASKS, matches_expected  # noqa: E402

load_dotenv()


class JupyterGemEnv(Env):
    """GEM environment for Jupyter notebook interaction via E2B sandbox.

    Follows the standard GEM Env API: reset() → step() → (obs, reward, done, truncated, info).
    Actions are free-form text with embedded tool call tags.
    """

    def __init__(
        self,
        task: str = "",
        expected_output: str = "",
        max_turns: int = 10,
        api_key: str = "",
        task_index: int = -1,
        **kwargs,
    ):
        super().__init__()
        self._api_key = api_key or os.environ.get("E2B_API_KEY", "")
        self._task = task
        self._expected_output = expected_output
        self._max_turns = max_turns
        self._task_index = task_index

        self._sandbox: Optional[E2BSandbox] = None
        self._tracker = NotebookTracker()
        self.last_output = ""
        self._last_code_success = False
        self.step_count = 0
        self.error_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset: kill old sandbox, pick task, return instruction."""
        super().reset(seed)

        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

        self._tracker = NotebookTracker()
        self.last_output = ""
        self._last_code_success = False
        self.step_count = 0
        self.error_count = 0

        # Pick task from TASKS if not provided
        if not self._task and self._task_index >= 0:
            t = TASKS[self._task_index % len(TASKS)]
            self._task = t["task"]
            self._expected_output = t["expected_output"]
        elif not self._task:
            t = random.choice(TASKS)
            self._task = t["task"]
            self._expected_output = t["expected_output"]

        instruction = (
            f"Solve this task by writing code.\n\n{self._task}\n\n"
            f"Use <code>...</code> tags for Python code, <shell>...</shell> for shell commands.\n"
            f"Print the final answer as the last output."
        )

        info = {
            "task": self._task,
            "expected_output": self._expected_output,
            "suffix": "Write your solution.",
        }

        return instruction, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Parse action for tool calls, execute in sandbox, return observation + reward."""
        # Lazy sandbox init
        if self._sandbox is None:
            self._sandbox = E2BSandbox(api_key=self._api_key)

        self.step_count += 1
        results = []

        # Parse tool calls from tags
        code_matches = re.findall(r"<code>(.*?)</code>", action, re.DOTALL)
        shell_matches = re.findall(r"<shell>(.*?)</shell>", action, re.DOTALL)
        edit_matches = re.findall(r"<edit>(.*?)</edit>", action, re.DOTALL)
        state_matches = re.findall(r"<state\s*/>", action)

        executed = False

        for code in code_matches:
            result = self._sandbox.run_code(code.strip())
            self._tracker.add_code_cell(code.strip(), result)
            text = _format_for_llm(result)
            self.last_output = text
            self._last_code_success = result.success
            if not result.success:
                self.error_count += 1
            results.append(f"[Code output]: {text}")
            executed = True

        for cmd in shell_matches:
            result = self._sandbox.run_shell(cmd.strip())
            self._tracker.add_shell_cell(cmd.strip(), result)
            text = _format_for_llm(result)
            results.append(f"[Shell output]: {text}")
            executed = True

        for code in edit_matches:
            result = self._sandbox.run_code(code.strip())
            self._tracker.update_last_code_cell(code.strip(), result)
            text = _format_for_llm(result)
            self.last_output = text
            self._last_code_success = result.success
            if not result.success:
                self.error_count += 1
            results.append(f"[Edit output]: {text}")
            executed = True

        for _ in state_matches:
            text = self._tracker.get_state_summary()
            results.append(f"[Notebook state]: {text}")
            executed = True

        # If no tags, treat as raw code
        if not executed:
            code_block = re.search(r"```(?:python)?\s*\n(.*?)```", action, re.DOTALL)
            if code_block:
                code = code_block.group(1).strip()
            else:
                code = action.strip()

            if code and not code.startswith(("The ", "I ", "Based ", "Let me", "Here")):
                result = self._sandbox.run_code(code)
                self._tracker.add_code_cell(code, result)
                text = _format_for_llm(result)
                self.last_output = text
                self._last_code_success = result.success
                if not result.success:
                    self.error_count += 1
                results.append(f"[Code output]: {text}")

        # Compute reward
        reward = self._compute_reward()
        terminated = reward > 0
        truncated = self.step_count >= self._max_turns

        observation = "\n".join(results) if results else "(no tool calls detected)"
        info = {
            "step_count": self.step_count,
            "error_count": self.error_count,
            "last_output": self.last_output,
            "suffix": "Continue solving." if not terminated else "Task completed.",
        }

        return observation, reward, terminated, truncated, info

    def spawn(self, same_state: bool = False, **kwargs) -> "JupyterGemEnv":
        """Spawn a new instance with same config."""
        return JupyterGemEnv(
            task=self._task if same_state else "",
            expected_output=self._expected_output if same_state else "",
            max_turns=self._max_turns,
            api_key=self._api_key,
            task_index=self._task_index,
            **kwargs,
        )

    def close(self):
        """Cleanup sandbox."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _compute_reward(self) -> float:
        if not self._expected_output or not self._last_code_success:
            return 0.0
        if matches_expected(self._expected_output, self.last_output):
            r = 1.0
            r += max(0.0, 0.2 * (1.0 - self.step_count / 10.0))
            r -= 0.05 * self.error_count
            return max(r, 0.0)
        return 0.0


def _format_for_llm(result: CellResult) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.strip())
    if result.text_results:
        parts.extend(result.text_results)
    if result.images:
        parts.append(f"[Image output: {len(result.images)} image(s) generated]")
    if result.error:
        parts.append(f"ERROR:\n{result.error}")
    return "\n".join(parts) if parts else "(no output)"
