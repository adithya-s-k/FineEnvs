"""
Jupyter Agent NeMo Gym Resources Server.

Exposes 4 notebook tools as NeMo Gym tool endpoints:
  POST /add_and_execute_code_cell
  POST /edit_and_execute_current_cell
  POST /execute_shell_command
  POST /get_notebook_state

Plus the standard NeMo Gym endpoints:
  POST /seed_session   — initialize E2B sandbox per session
  POST /verify         — compute reward after episode

Usage:
    python server.py

With NeMo Gym CLI:
    ng_run "+config_paths=[configs/jupyter_agent.yaml]"

Docker:
    docker build -t jupyter-agent-nemo-gym .
    docker run -p 11000:11000 -e E2B_API_KEY=... jupyter-agent-nemo-gym
"""

import sys
from pathlib import Path
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

_ENV_ROOT = str(Path(__file__).resolve().parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from core.e2b_sandbox import E2BSandbox, CellResult  # noqa: E402
from core.notebook_tracker import NotebookTracker  # noqa: E402
from core.tasks import TASKS, matches_expected  # noqa: E402

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class JupyterAgentConfig(BaseResourcesServerConfig):
    """Configuration for the Jupyter Agent NeMo Gym Resources Server."""
    pass


# ---------------------------------------------------------------------------
# Request / Response models for tool endpoints
# ---------------------------------------------------------------------------

class CodeRequest(BaseModel):
    code: str

class CommandRequest(BaseModel):
    command: str

class FinalAnswerRequest(BaseModel):
    answer: str

class NotebookStateRequest(BaseModel):
    include_images: bool = False

class ToolResponse(BaseModel):
    output: str

class JupyterVerifyRequest(BaseVerifyRequest):
    """Extended verify request with ground_truth for expected output checking."""
    ground_truth: list = []


# ---------------------------------------------------------------------------
# Resources Server
# ---------------------------------------------------------------------------

class JupyterAgentResourcesServer(SimpleResourcesServer):
    """NeMo Gym Resources Server for the Jupyter Agent environment.

    Each session gets an isolated E2B sandbox. The 4 tool endpoints execute
    code/commands in the sandbox. verify() checks if the expected output
    appears in the last execution result.
    """

    config: JupyterAgentConfig

    # Per-session state: sandbox + tracker + metadata
    sessions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        """Register tool endpoints on top of the base /seed_session and /verify."""
        app = super().setup_webserver()

        # Register 5 tool endpoints
        app.post("/add_and_execute_code_cell")(self.add_and_execute_code_cell)
        app.post("/edit_and_execute_current_cell")(self.edit_and_execute_current_cell)
        app.post("/execute_shell_command")(self.execute_shell_command)
        app.post("/get_notebook_state")(self.get_notebook_state)
        app.post("/final_answer")(self.final_answer)

        return app

    # ── Session lifecycle ──────────────────────────────────────────────────

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        """Acknowledge session creation. Sandbox created lazily on first tool call."""
        return BaseSeedSessionResponse()

    def _get_or_create_session(self, request: Request) -> Dict[str, Any]:
        """Get or create session state (lazy sandbox initialization)."""
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.sessions:
            api_key = os.environ.get("E2B_API_KEY", "")
            sandbox = E2BSandbox(api_key=api_key)
            tracker = NotebookTracker()
            self.sessions[session_id] = {
                "sandbox": sandbox,
                "tracker": tracker,
                "step_count": 0,
                "error_count": 0,
                "last_output": "",
            }
        return self.sessions[session_id]

    # ── Tool endpoints ─────────────────────────────────────────────────────

    async def add_and_execute_code_cell(
        self, body: CodeRequest, request: Request
    ) -> ToolResponse:
        """Execute Python code in the stateful Jupyter notebook."""
        sess = self._get_or_create_session(request)
        result = sess["sandbox"].run_code(body.code)
        sess["tracker"].add_code_cell(body.code, result)
        text = _format_for_llm(result)
        sess["last_output"] = text
        sess["step_count"] += 1
        if not result.success:
            sess["error_count"] += 1
        return ToolResponse(output=text)

    async def edit_and_execute_current_cell(
        self, body: CodeRequest, request: Request
    ) -> ToolResponse:
        """Replace the last code cell with new code and re-execute it."""
        sess = self._get_or_create_session(request)
        result = sess["sandbox"].run_code(body.code)
        sess["tracker"].update_last_code_cell(body.code, result)
        text = _format_for_llm(result)
        sess["last_output"] = text
        sess["step_count"] += 1
        if not result.success:
            sess["error_count"] += 1
        return ToolResponse(output=text)

    async def execute_shell_command(
        self, body: CommandRequest, request: Request
    ) -> ToolResponse:
        """Run a shell command inside the sandbox."""
        sess = self._get_or_create_session(request)
        result = sess["sandbox"].run_shell(body.command)
        sess["tracker"].add_shell_cell(body.command, result)
        text = _format_for_llm(result)
        sess["last_output"] = text
        sess["step_count"] += 1
        return ToolResponse(output=text)

    async def get_notebook_state(
        self, body: NotebookStateRequest, request: Request
    ) -> ToolResponse:
        """Return a compact summary of all executed cells and their outputs."""
        sess = self._get_or_create_session(request)
        text = sess["tracker"].get_state_summary(include_images=body.include_images)
        return ToolResponse(output=text)

    async def final_answer(
        self, body: FinalAnswerRequest, request: Request
    ) -> ToolResponse:
        """Submit your final answer to the question."""
        sess = self._get_or_create_session(request)
        sess["submitted_answer"] = body.answer
        return ToolResponse(output=f"Answer submitted: {body.answer}")

    # ── Verification ───────────────────────────────────────────────────────

    async def verify(self, body: JupyterVerifyRequest) -> BaseVerifyResponse:
        """Evaluate the episode and compute reward.

        Checks if any function_result in the response output contains
        the expected output string from the task's ground_truth.
        """
        # Extract expected output from the task metadata
        expected = ""
        if hasattr(body, "ground_truth") and body.ground_truth:
            if isinstance(body.ground_truth, list) and body.ground_truth:
                gt = body.ground_truth[0]
                expected = gt.get("expected_output", "") if isinstance(gt, dict) else str(gt)
            elif isinstance(body.ground_truth, dict):
                expected = body.ground_truth.get("expected_output", "")
            elif isinstance(body.ground_truth, str):
                expected = body.ground_truth

        # Also check responses_create_params for expected_output in metadata
        if not expected:
            params = body.responses_create_params
            if hasattr(params, "metadata") and params.metadata:
                expected = params.metadata.get("expected_output", "")

        # Scan executed tool outputs for the expected string using boundary matching
        # (do not reward ungrounded assistant message text without sandbox execution)
        reward = 0.0
        for item in body.response.output:
            if hasattr(item, "type") and item.type == "function_call_output":
                output_text = getattr(item, "output", "")
                if (
                    isinstance(output_text, str)
                    and not output_text.startswith("ERROR:")
                    and matches_expected(expected, output_text)
                ):
                    reward = 1.0
                    break

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    JupyterAgentResourcesServer.run_webserver()
