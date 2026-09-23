"""
Wordle MCP Environment.

Exposes 3 tools via FastMCP on MCPEnvironment:
  1. guess        — submit a 5-letter word guess
  2. get_history  — view all previous guesses and feedback
  3. reset_game   — start a new game with a random word

Each episode (reset → guess* → [reset]) maps to one WordleGame instance. When a guess ends the game,
the step observation has done=True and reward set to WordleGame.reward.
"""

import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

# Add parent dirs for game import
_wordle_root = str(Path(__file__).resolve().parents[2])
if _wordle_root not in sys.path:
    sys.path.insert(0, _wordle_root)

try:
    from core.game import WordleGame
except ImportError:  # Space image flattens core/ into the app root
    from game import WordleGame


class WordleEnvironment(MCPEnvironment):
    """
    Stateful Wordle environment backed by a WordleGame instance.

    Inherits from MCPEnvironment which auto-routes ListToolsAction and
    CallToolAction to the registered FastMCP tools.

    Concurrent sessions: each WebSocket connection gets its own instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._game: Optional[WordleGame] = None
        self._episode_id = str(uuid4())
        self._step_count = 0

        # Register MCP tools
        mcp = FastMCP("wordle_env")

        @mcp.tool
        def guess(word: str) -> str:
            """Submit a 5-letter word guess to the Wordle game.

            The game will respond with colored feedback:
            🟩 = correct letter in correct position
            🟨 = correct letter in wrong position
            ⬛ = letter not in the word

            Args:
                word: A 5-letter English word guess.

            Returns:
                Feedback string with emoji indicators and game status.
            """
            if not self._game:
                return "Error: game not started. Call reset first."
            return self._game.guess(word)

        @mcp.tool
        def get_history() -> str:
            """View all previous guesses and their feedback.

            Shows each guess with its colored feedback, how many guesses
            have been used, and whether the game is over.

            Returns:
                Formatted history of all guesses and their feedback.
            """
            if not self._game:
                return "No game in progress."
            return self._game.get_history()

        @mcp.tool
        def reset_game() -> str:
            """Start a new Wordle game with a random word.

            Returns:
                Confirmation that a new game has started.
            """
            self._game = WordleGame()
            return "New game started! Guess the 5-letter word. You have 6 attempts."

        super().__init__(mcp)

    # -- OpenEnv lifecycle --------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Start a new episode with a fresh WordleGame."""
        answer = kwargs.get("answer", "")
        self._game = WordleGame(answer=answer) if answer else WordleGame()
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0

        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": "Wordle game ready. Use the guess tool to submit 5-letter words.",
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Fallback for non-MCP actions."""
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1
        return self._with_game_outcome(super().step(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1
        return self._with_game_outcome(await super().step_async(action, timeout_s=timeout_s, **kwargs))

    def _with_game_outcome(self, observation: Observation) -> Observation:
        """Tell OpenEnv clients when the game is over: done=True and the game's reward."""
        if self._game is not None and self._game.done:
            observation.done = True
            observation.reward = float(self._game.reward)
        return observation

    @property
    def state(self):
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "game_done": self._game.done if self._game else False,
            "game_won": self._game.won if self._game else False,
            "reward": self._game.reward if self._game and self._game.done else None,
        }
