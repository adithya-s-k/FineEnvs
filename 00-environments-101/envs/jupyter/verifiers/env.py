"""
Jupyter Agent Verifiers Environment.

Properly uses the PrimeIntellect Verifiers library:
- Tools are plain Python functions (verifiers' native tool format)
- vf.ToolEnv wraps them with multi-turn support
- vf.Rubric defines reward scoring
- JupyterToolkit manages E2B sandbox state

This module provides:
1. JupyterToolkit — stateful E2B sandbox manager (used by TRL adapter)
2. create_verifiers_env() — creates a proper vf.ToolEnv (for native verifiers use)

Usage with TRL (via adapter):
    # Adapter uses JupyterToolkit directly for tool discovery
    trainer = GRPOTrainer(
        environment_factory=VerifiersEnvironment,
        environment_config={},
    )

Usage with native verifiers:
    env = create_verifiers_env()
    results = await env.evaluate(client=AsyncOpenAI(...), model="gpt-4")
"""

import sys
from pathlib import Path
import os
from typing import Optional

from dotenv import load_dotenv

_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from core.e2b_sandbox import E2BSandbox, CellResult  # noqa: E402
from core.notebook_tracker import NotebookTracker  # noqa: E402
from core.tasks import TASKS, matches_expected  # noqa: E402

load_dotenv()


# ---------------------------------------------------------------------------
# JupyterToolkit — stateful E2B sandbox manager
# Used by the TRL InProcessEnvironment adapter for tool discovery
# ---------------------------------------------------------------------------

class JupyterToolkit:
    """Stateful toolkit wrapping an E2B sandbox with 4 Jupyter tools.

    Each instance manages one sandbox (one episode). The TRL adapter
    introspects public methods to discover tools automatically.
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("E2B_API_KEY", "")
        self._sandbox: Optional[E2BSandbox] = None
        self._tracker = NotebookTracker()
        self.last_output = ""
        self.step_count = 0
        self.error_count = 0
        self._submitted_answer = None

    def initialize(self):
        """Create sandbox. Called lazily on first tool call."""
        if self._sandbox is None:
            self._sandbox = E2BSandbox(api_key=self._api_key)
            self._tracker.reset()

    def cleanup(self):
        """Kill sandbox."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def reset(self):
        """Reset for new episode."""
        self.cleanup()
        self.last_output = ""
        self.step_count = 0
        self.error_count = 0
        self._submitted_answer = None
        self._tracker = NotebookTracker()

    def add_and_execute_code_cell(self, code: str) -> str:
        """Execute Python code in the stateful Jupyter notebook.

        Variables, imports, and side-effects persist between calls within
        the same episode. Use this as the primary tool for all computation.

        Args:
            code: Python code to execute. Can span multiple lines.
        """
        self.initialize()
        self.step_count += 1
        result = self._sandbox.run_code(code)
        self._tracker.add_code_cell(code, result)
        text = _format_for_llm(result)
        self.last_output = text
        if not result.success:
            self.error_count += 1
        return text

    def edit_and_execute_current_cell(self, code: str) -> str:
        """Replace the last code cell with new code and re-execute it.

        Use this to fix errors in the previous cell instead of creating a
        new cell. This keeps the notebook clean.

        Args:
            code: Replacement Python code for the current cell.
        """
        self.initialize()
        self.step_count += 1
        result = self._sandbox.run_code(code)
        self._tracker.update_last_code_cell(code, result)
        text = _format_for_llm(result)
        self.last_output = text
        if not result.success:
            self.error_count += 1
        return text

    def execute_shell_command(self, command: str) -> str:
        """Run a shell command inside the sandbox.

        Useful for package installation, file system inspection, or
        running scripts. Examples: 'pip install polars', 'ls -la'.

        Args:
            command: Shell command string to execute.
        """
        self.initialize()
        self.step_count += 1
        result = self._sandbox.run_shell(command)
        self._tracker.add_shell_cell(command, result)
        text = _format_for_llm(result)
        self.last_output = text
        return text

    def get_notebook_state(self, include_images: bool = False) -> str:
        """Return a compact summary of all executed cells and their outputs.

        Useful to check what has already been computed or to review results.

        Args:
            include_images: Whether to include base64-encoded images.
        """
        return self._tracker.get_state_summary(include_images=include_images)

    def final_answer(self, answer: str) -> str:
        """Submit your final answer to the question.

        Call this when you have computed the answer and are ready to submit.
        This ends the current task.

        Args:
            answer: Your final answer as a string.
        """
        self._submitted_answer = answer
        self.step_count += 1
        self.last_output = f"Final answer submitted: {answer}"
        return f"Answer submitted: {answer}"

    @property
    def submitted_answer(self) -> Optional[str]:
        return getattr(self, "_submitted_answer", None)


# ---------------------------------------------------------------------------
# Standalone tool functions for native verifiers vf.ToolEnv
# These are stateless wrappers — each call creates/reuses a shared sandbox
# ---------------------------------------------------------------------------

# Shared sandbox for verifiers' native tool functions
_shared_sandbox: Optional[E2BSandbox] = None
_shared_tracker: Optional[NotebookTracker] = None


def _get_sandbox() -> E2BSandbox:
    global _shared_sandbox
    if _shared_sandbox is None:
        api_key = os.environ.get("E2B_API_KEY", "")
        _shared_sandbox = E2BSandbox(api_key=api_key)
    return _shared_sandbox


def _get_tracker() -> NotebookTracker:
    global _shared_tracker
    if _shared_tracker is None:
        _shared_tracker = NotebookTracker()
    return _shared_tracker


def add_and_execute_code_cell(code: str) -> str:
    """Execute Python code in the stateful Jupyter notebook.

    Variables, imports, and side-effects persist between calls.

    Args:
        code: Python code to execute.
    """
    result = _get_sandbox().run_code(code)
    _get_tracker().add_code_cell(code, result)
    return _format_for_llm(result)


def edit_and_execute_current_cell(code: str) -> str:
    """Replace the last code cell with new code and re-execute.

    Args:
        code: Replacement Python code.
    """
    result = _get_sandbox().run_code(code)
    _get_tracker().update_last_code_cell(code, result)
    return _format_for_llm(result)


def execute_shell_command(command: str) -> str:
    """Run a shell command inside the sandbox.

    Args:
        command: Shell command to execute.
    """
    result = _get_sandbox().run_shell(command)
    _get_tracker().add_shell_cell(command, result)
    return _format_for_llm(result)


def get_notebook_state(include_images: bool = False) -> str:
    """Return summary of executed cells and outputs.

    Args:
        include_images: Include base64 images.
    """
    return _get_tracker().get_state_summary(include_images=include_images)


# The 4 tool functions as a list (verifiers' expected format)
TOOL_FUNCTIONS = [
    add_and_execute_code_cell,
    edit_and_execute_current_cell,
    execute_shell_command,
    get_notebook_state,
]


# ---------------------------------------------------------------------------
# Create a proper verifiers ToolEnv + Rubric
# ---------------------------------------------------------------------------

def create_verifiers_env():
    """Create a proper vf.ToolEnv for native verifiers usage.

    Returns a vf.ToolEnv that can be used with:
        results = await env.evaluate(client=AsyncOpenAI(...), model="gpt-4")

    Requires: pip install verifiers
    """
    import verifiers as vf
    from datasets import Dataset

    # Build dataset in verifiers format
    dataset = Dataset.from_list([
        {"question": t["task"], "answer": t["expected_output"]}
        for t in TASKS
    ])

    # Reward rubric: verify expected output in successful tool execution outputs
    # rather than ungrounded assistant text (prevents guessing without running code)
    async def correctness(completion, answer, **kwargs) -> float:
        if not completion:
            return 0.0
        if isinstance(completion, list):
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    content = str(msg.get("content", ""))
                    if not content.startswith("ERROR:") and matches_expected(answer, content):
                        return 1.0
            return 0.0
        return 1.0 if matches_expected(answer, str(completion)) else 0.0

    rubric = vf.Rubric(funcs=[correctness])

    # Create ToolEnv with our 4 tools
    env = vf.ToolEnv(
        tools=TOOL_FUNCTIONS,
        max_turns=10,
        dataset=dataset,
        rubric=rubric,
        system_prompt=(
            "You are an intelligent data science assistant operating inside a "
            "stateful Jupyter notebook environment. Use the available tools to "
            "solve tasks through code execution."
        ),
    )

    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_for_llm(result: CellResult) -> str:
    """Format a CellResult as a concise string."""
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
