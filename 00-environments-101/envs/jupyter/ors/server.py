"""
Jupyter Agent ORS Environment Server.

Exposes 4 notebook interaction tools via the Open Reward Standard (ORS):
  1. add_and_execute_code_cell    — primary execution tool
  2. edit_and_execute_current_cell — error-recovery (replace last cell)
  3. execute_shell_command         — shell access inside sandbox
  4. get_notebook_state            — compact history for agent memory

Each episode maps to one E2B sandbox. Rewards are embedded in every
ToolOutput — the server computes whether the expected output was found.

Usage:
    E2B_API_KEY=... python server.py                     # localhost:8080
    E2B_API_KEY=... python server.py --port 9090         # custom port

Deploy:
    docker build -t jupyter-agent-ors .
    docker run -p 8080:8080 -e E2B_API_KEY=... jupyter-agent-ors
"""

import sys
from pathlib import Path
import os
import argparse
from typing import Optional

from pydantic import BaseModel

from openreward.environments import Environment, Server, tool, ToolOutput, TextBlock, Split

_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from core.e2b_sandbox import E2BSandbox, CellResult  # noqa: E402
from core.notebook_tracker import NotebookTracker  # noqa: E402
from core.tasks import TASKS, matches_expected  # noqa: E402



# TASKS imported from tasks.py (same 46 tasks as openenv/dataset.py)


# ---------------------------------------------------------------------------
# Pydantic input models for @tool methods
# ---------------------------------------------------------------------------

class CodeInput(BaseModel):
    code: str

class CommandInput(BaseModel):
    command: str

class NotebookStateInput(BaseModel):
    include_images: bool = False

class FinalAnswerInput(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# ORS Environment
# ---------------------------------------------------------------------------

class JupyterAgentORS(Environment):
    """Jupyter notebook environment via ORS protocol.

    Each session gets an isolated E2B sandbox. Tools execute code, shell
    commands, and inspect notebook state. Rewards are computed inline by
    checking if the expected output appears in the execution result.
    """

    def __init__(self, task_spec=None, secrets=None, **kwargs):
        # ORS passes task_spec and secrets to __init__
        if task_spec is None:
            task_spec = {}
        if secrets is None:
            secrets = {}
        super().__init__(task_spec=task_spec, secrets=secrets)
        self._api_key = secrets.get("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
        self._sandbox: Optional[E2BSandbox] = None
        self._tracker = NotebookTracker()
        self._step_count = 0
        self._error_count = 0
        self._last_output = ""
        self._code_executed_ok = False

    # ── ORS lifecycle ──────────────────────────────────────────────────────

    def setup(self):
        """Called on first tool invocation — create E2B sandbox."""
        self._sandbox = E2BSandbox(api_key=self._api_key)
        self._tracker.reset()
        self._step_count = 0
        self._error_count = 0
        self._last_output = ""
        self._submitted_answer = None
        self._code_executed_ok = False

    def teardown(self):
        """Called on session delete — kill sandbox."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    # ── ORS metadata ───────────────────────────────────────────────────────

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split: str):
        return TASKS

    def get_prompt(self):
        task = self.task_spec.get("task", "")
        return [TextBlock(
            text=f"Solve this task using the Jupyter notebook tools:\n\n{task}\n\nPrint the final answer as the last output."
        )]

    # ── Tools ──────────────────────────────────────────────────────────────

    @tool
    def add_and_execute_code_cell(self, params: CodeInput) -> ToolOutput:
        """Execute Python code in the stateful Jupyter notebook.

        Variables, imports, and side-effects persist between calls within
        the same episode. Use this as the primary tool for all computation.
        """
        if not self._sandbox:
            return ToolOutput(
                blocks=[TextBlock(text="Error: environment not initialized.")],
                reward=0.0,
                finished=False,
            )
        self._step_count += 1
        result = self._sandbox.run_code(params.code)
        self._tracker.add_code_cell(params.code, result)
        text = _format_for_llm(result)
        self._last_output = text
        if not result.success:
            self._error_count += 1
            reward = 0.0
        else:
            self._code_executed_ok = True
            reward = self._compute_reward(text)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=reward,
            finished=reward > 0,
        )

    @tool
    def edit_and_execute_current_cell(self, params: CodeInput) -> ToolOutput:
        """Replace the last code cell with new code and re-execute it.

        Use this to fix errors in the previous cell instead of creating a
        new cell. This keeps the notebook clean.
        """
        if not self._sandbox:
            return ToolOutput(
                blocks=[TextBlock(text="Error: environment not initialized.")],
                reward=0.0,
                finished=False,
            )
        self._step_count += 1
        result = self._sandbox.run_code(params.code)
        self._tracker.update_last_code_cell(params.code, result)
        text = _format_for_llm(result)
        self._last_output = text
        if not result.success:
            self._error_count += 1
            reward = 0.0
        else:
            self._code_executed_ok = True
            reward = self._compute_reward(text)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=reward,
            finished=reward > 0,
        )

    @tool
    def execute_shell_command(self, params: CommandInput) -> ToolOutput:
        """Run a shell command inside the sandbox.

        Useful for package installation, file system inspection, or
        running scripts. Examples: 'pip install polars', 'ls -la'.
        """
        if not self._sandbox:
            return ToolOutput(
                blocks=[TextBlock(text="Error: environment not initialized.")],
                reward=0.0,
                finished=False,
            )
        self._step_count += 1
        result = self._sandbox.run_shell(params.command)
        self._tracker.add_shell_cell(params.command, result)
        text = _format_for_llm(result)
        self._last_output = text
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=0.0,
            finished=False,
        )

    @tool
    def get_notebook_state(self, params: NotebookStateInput) -> ToolOutput:
        """Return a compact summary of all executed cells and their outputs.

        Useful to check what has already been computed or to review results.
        """
        text = self._tracker.get_state_summary(include_images=params.include_images)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=None,  # No reward for state inspection
            finished=False,
        )

    @tool
    def final_answer(self, params: FinalAnswerInput) -> ToolOutput:
        """Submit your final answer to the question.

        Call this when you have computed the answer and are ready to submit.
        This ends the current task.
        """
        self._submitted_answer = params.answer
        # Require at least one successful code execution so agents cannot guess on Turn 1
        reward = self._compute_reward(params.answer) if self._code_executed_ok else 0.0
        return ToolOutput(
            blocks=[TextBlock(text=f"Answer submitted: {params.answer}")],
            reward=reward,
            finished=True,
        )

    # ── Reward computation ─────────────────────────────────────────────────

    def _compute_reward(self, output: str) -> float:
        """Check if expected output appears in the execution result.

        Reward components:
        - Task completion (1.0): expected output found in output
        - Efficiency bonus (up to 0.2): fewer steps = higher bonus
        - Error penalty (-0.05 per error)
        """
        expected = self.task_spec.get("expected_output", "").strip()
        if not expected:
            return 0.0

        if matches_expected(expected, output):
            r = 1.0
            r += max(0.0, 0.2 * (1.0 - self._step_count / 10.0))
            r -= 0.05 * self._error_count
            return max(r, 0.0)

        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_for_llm(result: CellResult) -> str:
    """Format a CellResult as a concise string for the LLM."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Jupyter Agent ORS Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Jupyter Agent ORS server on {args.host}:{args.port}")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Tools: add_and_execute_code_cell, edit_and_execute_current_cell, execute_shell_command, get_notebook_state")
    Server([JupyterAgentORS]).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
