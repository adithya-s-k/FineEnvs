"""
Wordle Verifiers Toolkit.

Exposes 2 tools for TRL's InProcessEnvironment adapter:
  - guess(word: str) → str   — submit a 5-letter guess
  - get_history() → str      — view all previous guesses and feedback

The adapter discovers these automatically via inspect.getmembers().
No changes to any adapter code needed.
"""

import sys
from pathlib import Path

# Add parent to path so we can import game.py
_parent = str(Path(__file__).resolve().parents[1])
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from core.game import WordleGame, TASKS


class WordleToolkit:
    """Stateful Wordle toolkit. One game per episode."""

    def __init__(self, **kwargs):
        self._game = None
        self._answer = ""
        self.last_output = ""
        self.step_count = 0
        self.error_count = 0

    def reset(self, answer: str = "", **kwargs):
        """Reset for a new episode, optionally pinning the hidden word.

        TRL calls reset(**row) with the dataset row, so an `answer` column fixes the word for that
        episode; with no answer the game picks one at random. This replaces a public set_answer():
        TRL exposes every public method except reset as a tool, so the model could have set the
        word itself and then guessed it.
        """
        self._answer = answer
        self._game = None
        self.last_output = ""
        self.step_count = 0
        self.error_count = 0

    def cleanup(self):
        """Nothing to clean up (pure Python, no external resources)."""
        pass

    def _ensure_game(self):
        if self._game is None:
            self._game = WordleGame(answer=self._answer)

    def guess(self, word: str) -> str:
        """Submit a 5-letter word guess to the Wordle game.

        The game will respond with colored feedback:
        🟩 = correct letter in correct position
        🟨 = correct letter in wrong position
        ⬛ = letter not in the word

        Args:
            word: A 5-letter English word guess.
        """
        self._ensure_game()
        self.step_count += 1
        result = self._game.guess(word)
        self.last_output = result
        return result

    def get_history(self) -> str:
        """View all previous guesses and their feedback.

        Shows each guess with its colored feedback, the number of
        guesses used, and whether the game is over.

        Args:
            (no arguments)
        """
        self._ensure_game()
        result = self._game.get_history()
        self.last_output = result
        return result

    @property
    def reward(self):
        if self._game:
            return self._game.reward
        return 0.0
