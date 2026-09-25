"""
Wordle NeMo Gym Resources Server.

Exposes 2 tool endpoints:
  POST /guess        — submit a 5-letter word guess
  POST /get_history  — view all previous guesses and feedback

Plus the standard NeMo Gym endpoints:
  POST /seed_session   — initialize game per session
  POST /verify         — compute reward after episode

Usage:
    python server.py

With NeMo Gym CLI:
    ng_run "+config_paths=[configs/wordle.yaml]"

Docker (from wordle/ directory):
    docker build -f nemo_gym/Dockerfile -t wordle-nemo-gym .
    docker run -p 11000:11000 wordle-nemo-gym
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

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

# Add parent to path for game import
_parent = str(Path(__file__).resolve().parents[1])
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from core.game import WordleGame, TASKS

# What WordleGame.guess returns on a win, and nothing else: five green squares, then the
# announcement. An invalid guess echoes the model's word back, so matching the whole line (not a
# substring) is what keeps a crafted guess from producing something that reads as a win.
WIN_LINE = re.compile(r"🟩{5} — Correct! The word was '[a-z]{5}'\. Solved in \d+ guesses\.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class WordleConfig(BaseResourcesServerConfig):
    """Configuration for the Wordle NeMo Gym Resources Server."""
    pass


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GuessRequest(BaseModel):
    word: str

class HistoryRequest(BaseModel):
    pass

class ToolResponse(BaseModel):
    output: str

class WordleVerifyRequest(BaseVerifyRequest):
    """Extended verify request with ground_truth."""
    ground_truth: list = []


# ---------------------------------------------------------------------------
# Resources Server
# ---------------------------------------------------------------------------

class WordleResourcesServer(SimpleResourcesServer):
    """NeMo Gym Resources Server for Wordle.

    Each session gets a WordleGame. The 2 tool endpoints submit guesses
    and inspect history. verify() checks if the game was won.
    """

    config: WordleConfig

    sessions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        """Register tool endpoints on top of base /seed_session and /verify."""
        app = super().setup_webserver()

        app.post("/guess")(self.guess)
        app.post("/get_history")(self.get_history)

        return app

    # -- Session lifecycle --------------------------------------------------

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        """Acknowledge session creation. Game created lazily on first tool call."""
        return BaseSeedSessionResponse()

    def _get_or_create_session(self, request: Request) -> Dict[str, Any]:
        """Get or create session state (lazy game initialization)."""
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "game": WordleGame(),
                "step_count": 0,
            }
        return self.sessions[session_id]

    # -- Tool endpoints -----------------------------------------------------

    async def guess(self, body: GuessRequest, request: Request) -> ToolResponse:
        """Submit a 5-letter word guess to the Wordle game."""
        sess = self._get_or_create_session(request)
        result = sess["game"].guess(body.word)
        sess["step_count"] += 1
        return ToolResponse(output=result)

    async def get_history(self, body: HistoryRequest, request: Request) -> ToolResponse:
        """View all previous guesses and their feedback."""
        sess = self._get_or_create_session(request)
        result = sess["game"].get_history()
        return ToolResponse(output=result)

    # -- Verification -------------------------------------------------------

    async def verify(self, body: WordleVerifyRequest) -> BaseVerifyResponse:
        """Evaluate the episode: 1.0 if the game itself reported a win, else 0.0.

        Only tool outputs count, and only the exact line WordleGame.guess returns on a win. The
        model's own messages are ignored: scanning them for "Correct" paid 1.0 to a model that
        simply wrote the word, without winning.
        """
        reward = 0.0
        for item in body.response.output:
            if getattr(item, "type", None) != "function_call_output":
                continue
            output = getattr(item, "output", "")
            if isinstance(output, str) and WIN_LINE.fullmatch(output.strip()):
                reward = 1.0
                break

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


# ---------------------------------------------------------------------------
# Standalone entry point (for Docker / HF Spaces deployment)
#
# run_webserver() requires the full NeMo Gym orchestrator (Ray + OmegaConf
# config). For standalone deployment we instantiate the server directly
# with uvicorn, while still using all the SDK classes above.
#
# For full NeMo Gym orchestration use:
#   ng_run "+config_paths=[configs/wordle.yaml]"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from uuid import uuid4
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi import Request, Response

    port = int(os.environ.get("PORT", "11000"))

    # Instantiate server with minimal config (no full orchestrator)
    config = WordleConfig.model_construct(
        entrypoint="server.py",
        domain="agent",
        name="wordle",
        host="0.0.0.0",
        port=port,
    )
    server = WordleResourcesServer.model_construct(
        config=config,
        server_client=None,
        sessions={},
    )
    app = FastAPI(title="Wordle NeMo Gym Resources Server")

    # Register endpoints
    app.post("/seed_session")(server.seed_session)
    app.post("/verify")(server.verify)
    app.post("/guess")(server.guess)
    app.post("/get_history")(server.get_history)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Session middleware (same pattern as SDK)
    @app.middleware("http")
    async def add_session_id(request: Request, call_next):
        request.session[SESSION_ID_KEY] = request.session.get(
            SESSION_ID_KEY, str(uuid4())
        )
        response: Response = await call_next(request)
        return response

    app.add_middleware(SessionMiddleware, secret_key="wordle-nemo-gym")

    print(f"Starting Wordle NeMo Gym server on 0.0.0.0:{port}")
    print(f"  Tasks: {len(TASKS)}, Tools: guess, get_history")
    uvicorn.run(app, host="0.0.0.0", port=port)
