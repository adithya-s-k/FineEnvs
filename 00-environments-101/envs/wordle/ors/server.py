"""
Wordle ORS Environment Server.

Exposes 2 tools via the Open Reward Standard (ORS):
  1. guess        — submit a 5-letter word guess
  2. get_history  — view all previous guesses and feedback

Each session maps to one WordleGame. Rewards are embedded in every
ToolOutput — the server returns game reward after each guess.

Usage:
    python server.py                     # localhost:8080
    python server.py --port 9090         # custom port

Deploy:
    docker build -t wordle-ors .
    docker run -p 8080:8080 wordle-ors
"""

import os
import sys
import argparse
from pathlib import Path

from pydantic import BaseModel

from openreward.environments import Environment, Server, tool, ToolOutput, TextBlock, Split

# Add parent to path for game import
_parent = str(Path(__file__).resolve().parents[1])
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from core.game import WordleGame, TASKS


# ---------------------------------------------------------------------------
# Pydantic input models for @tool methods
# ---------------------------------------------------------------------------

class GuessInput(BaseModel):
    word: str

class HistoryInput(BaseModel):
    pass


# ---------------------------------------------------------------------------
# ORS Environment
# ---------------------------------------------------------------------------

class WordleORS(Environment):
    """Wordle environment via ORS protocol.

    Each session gets a WordleGame instance. Tools submit guesses and
    inspect history. Rewards are computed inline from game state.
    """

    def __init__(self, task_spec=None, secrets=None, **kwargs):
        if task_spec is None:
            task_spec = {}
        if secrets is None:
            secrets = {}
        super().__init__(task_spec=task_spec, secrets=secrets)
        self._game = None
        self._rewarded_game = None

    # -- ORS lifecycle ------------------------------------------------------

    def setup(self):
        """Called on first tool invocation — create WordleGame."""
        answer = self.task_spec.get("answer", "")
        self._game = WordleGame(answer=answer) if answer else WordleGame()
        self._rewarded_game = None

    def teardown(self):
        """Called on session delete."""
        self._game = None

    # -- ORS metadata -------------------------------------------------------

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split: str):
        return TASKS

    def get_prompt(self):
        return [TextBlock(
            text=(
                "Play Wordle! Guess the hidden 5-letter word in 6 attempts.\n"
                "After each guess, you'll see feedback:\n"
                "  🟩 = correct letter, correct position\n"
                "  🟨 = correct letter, wrong position\n"
                "  ⬛ = letter not in the word\n"
                "Use the guess tool to submit each attempt."
            )
        )]

    # -- Tools --------------------------------------------------------------

    @tool
    def guess(self, params: GuessInput) -> ToolOutput:
        """Submit a 5-letter word guess to the Wordle game.

        The game will respond with colored feedback:
        🟩 = correct letter in correct position
        🟨 = correct letter in wrong position
        ⬛ = letter not in the word
        """
        if not self._game:
            return ToolOutput(
                blocks=[TextBlock(text="Error: game not initialized.")],
                reward=0.0,
                finished=False,
            )

        result = self._game.guess(params.word)
        reward = self._game.reward
        if self._game.done:
            if self._game is self._rewarded_game:
                reward = None
            else:
                self._rewarded_game = self._game

        return ToolOutput(
            blocks=[TextBlock(text=result)],
            reward=reward,
            finished=self._game.done,
        )

    @tool
    def get_history(self, params: HistoryInput) -> ToolOutput:
        """View all previous guesses and their feedback.

        Shows each guess with its colored feedback and how many guesses remain.
        """
        if not self._game:
            return ToolOutput(
                blocks=[TextBlock(text="No game in progress.")],
                reward=None,
                finished=False,
            )

        result = self._game.get_history()
        return ToolOutput(
            blocks=[TextBlock(text=result)],
            reward=None,
            finished=False,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wordle ORS Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Wordle ORS server on {args.host}:{args.port}")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Tools: guess, get_history")
    Server([WordleORS]).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
