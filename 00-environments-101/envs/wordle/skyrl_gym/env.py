"""
Wordle SkyRL Gym Environment.

Subclasses skyrl_gym.envs.base_text_env.BaseTextEnv with init/step/close.
The model sends free text via step(). The env parses a 5-letter guess word
and calls WordleGame.guess().

This is the native SkyRL pattern — for TRL, the adapter uses
WordleToolkit via InProcessEnvironment instead.

Usage with native SkyRL:
    import skyrl_gym
    env = skyrl_gym.make("wordle:Wordle-v0")
    obs, info = env.init([{"role": "user", "content": "Guess the word"}])
    result = env.step("apple")

Usage with TRL (via adapter):
    trainer = GRPOTrainer(
        environment_factory=SkyRLEnvironment,
        environment_config={"toolkit_cls": "environments.wordle.verifiers.env.WordleToolkit"},
    )
"""

import re
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Add parent to path for game import
_parent = str(Path(__file__).resolve().parents[1])
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:
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

from core.game import WordleGame, TASKS


class WordleSkyRLEnv(BaseTextEnv):
    """SkyRL Gym BaseTextEnv for Wordle.

    The model sends free text via step(). The env extracts a 5-letter word
    and uses it as a Wordle guess. Supports <guess>word</guess> tags or
    raw 5-letter words; words are not taken from surrounding prose.
    """

    def __init__(self, answer: str = "", max_turns: int = 6, lenient_parsing: bool = False):
        super().__init__()
        self._answer = answer
        self.max_turns = max_turns
        self._lenient_parsing = lenient_parsing
        self._game = None
        self.last_output = ""
        self.error_count = 0
        self.turns = 0

    def init(self, prompt) -> Tuple[Any, Dict]:
        """Initialize episode. Creates fresh WordleGame."""
        if self._answer:
            self._game = WordleGame(answer=self._answer)
        else:
            self._game = WordleGame()
        self.last_output = ""
        self.turns = 0
        self.error_count = 0
        return prompt, {"max_turns": self.max_turns, "answer": self._game.answer}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process model text — extract guess word and submit to game."""
        self.turns += 1
        already_done = self._game.done

        word = _extract_guess(action, lenient=self._lenient_parsing)
        if not word:
            self.error_count += 1
            result = "Could not parse a 5-letter word from your response. Use <guess>word</guess> or just type a 5-letter word."
        else:
            result = self._game.guess(word)
            if "Invalid" in result:
                self.error_count += 1

        self.last_output = result
        reward = 0.0 if already_done else self._game.reward
        done = self._game.done or self.turns >= self.max_turns

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": result}],
            reward=reward,
            done=done,
            metadata={
                "turns": self.turns,
                "errors": self.error_count,
                "won": self._game.won,
            },
        )

    def close(self):
        self._game = None


# Register with SkyRL Gym registry
try:
    import skyrl_gym
    skyrl_gym.register("wordle:Wordle-v0", WordleSkyRLEnv)
except (ImportError, Exception):
    pass


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
