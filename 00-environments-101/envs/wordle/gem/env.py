"""
Wordle GEM Environment.

A proper GEM (gem.Env) subclass implementing the Gymnasium-style
reset()/step() interface for Wordle.

The model sends text actions which the environment parses for a guess word.
Supports <guess>word</guess> tags or raw 5-letter words.

Usage:
    import gem
    from env import WordleGemEnv

    gem.register("wordle:Wordle-v0", WordleGemEnv)
    env = gem.make("wordle:Wordle-v0", answer="apple")
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step("<guess>crane</guess>")
"""

import re
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, SupportsFloat, Tuple

# Add parent to path for game import
_parent = str(Path(__file__).resolve().parents[1])
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from gem import Env
except ImportError:
    import abc
    class Env(abc.ABC):
        @abc.abstractmethod
        def step(self, action): ...
        @abc.abstractmethod
        def reset(self, seed=None): ...
        def close(self): pass

from core.game import WordleGame, TASKS


class WordleGemEnv(Env):
    """GEM environment for Wordle.

    Follows the standard GEM Env API: reset() -> step() -> (obs, reward, done, truncated, info).
    Actions are text holding a <guess>word</guess> tag, or just the 5-letter word.
    """

    def __init__(
        self,
        answer: str = "",
        max_turns: int = 6,
        task_index: int = -1,
        lenient_parsing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._answer = answer
        self._max_turns = max_turns
        self._task_index = task_index
        self._lenient_parsing = lenient_parsing
        self._game = None
        self.last_output = ""
        self.step_count = 0
        self.error_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset: create new game, return instruction."""
        super().reset(seed)

        self.last_output = ""
        self.step_count = 0
        self.error_count = 0

        # Pick answer
        if self._answer:
            answer = self._answer
        elif self._task_index >= 0:
            answer = TASKS[self._task_index % len(TASKS)]["answer"]
        else:
            answer = random.choice(TASKS)["answer"]

        self._game = WordleGame(answer=answer)

        instruction = (
            f"Play Wordle! Guess the hidden 5-letter word in {self._max_turns} attempts.\n"
            f"After each guess you'll see feedback:\n"
            f"  🟩 = correct letter, correct position\n"
            f"  🟨 = correct letter, wrong position\n"
            f"  ⬛ = letter not in the word\n\n"
            f"Use <guess>word</guess> tags to submit your guess."
        )

        info = {
            "answer": self._game.answer,
            "suffix": "Make your first guess.",
        }

        return instruction, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Parse action for guess word, submit to game, return observation + reward."""
        self.step_count += 1
        already_done = self._game.done

        word = _extract_guess(action, lenient=self._lenient_parsing)
        if not word:
            self.error_count += 1
            result = "Could not parse a 5-letter word. Use <guess>word</guess> tags."
        else:
            result = self._game.guess(word)
            if "Invalid" in result:
                self.error_count += 1

        self.last_output = result
        reward = 0.0 if already_done else self._game.reward
        terminated = self._game.done
        truncated = self.step_count >= self._max_turns and not terminated

        info = {
            "step_count": self.step_count,
            "error_count": self.error_count,
            "last_output": self.last_output,
            "won": self._game.won,
            "suffix": "Game over." if terminated or truncated else "Make your next guess.",
        }

        return result, reward, terminated, truncated, info

    def spawn(self, same_state: bool = False, **kwargs) -> "WordleGemEnv":
        """Spawn a new instance with same config."""
        return WordleGemEnv(
            answer=self._answer if same_state else "",
            max_turns=self._max_turns,
            task_index=self._task_index,
            lenient_parsing=self._lenient_parsing,
            **kwargs,
        )

    def close(self):
        self._game = None


def _extract_guess(text: str, lenient: bool = False) -> str:
    """Extract a 5-letter guess: a <guess>word</guess> tag, or a reply that is only the word.

    Words are not pulled out of surrounding prose, so "Could you share your last guess?" is not played as the
    guess "guess". lenient=True restores the earlier fallback (a "guess is X" phrase, then the last 5-letter
    word anywhere in the text), e.g. to reproduce results produced before this change.
    """
    match = re.search(r"<guess>(.*?)</guess>", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    bare = text.strip().strip(".!?\"'`*").strip()
    if re.fullmatch(r"[A-Za-z]{5}", bare):
        return bare.lower()

    if not lenient:
        return ""

    match = re.search(r"guess(?:\s+is)?[:\s]+([a-zA-Z]{5})\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    words = re.findall(r"\b([a-zA-Z]{5})\b", text)
    if words:
        return words[-1].lower()

    return ""
