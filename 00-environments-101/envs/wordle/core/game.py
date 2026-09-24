"""
Core Wordle game logic. Shared across all frameworks.

No framework dependencies — pure Python.
"""

import random
from typing import Optional

# Common 5-letter words for Wordle
WORD_LIST = [
    "apple", "beach", "chair", "dance", "eagle", "flame", "grape", "house",
    "joker", "knife", "lemon", "mango", "night", "ocean", "piano", "queen",
    "river", "stone", "tiger", "ultra", "virus", "whale", "youth", "zebra",
    "bread", "cloud", "dream", "earth", "fairy", "ghost", "heart", "ivory",
    "jewel", "knock", "light", "money", "noble", "olive", "power", "quiet",
    "round", "smile", "train", "unity", "voice", "world", "angel", "blaze",
    "crane", "drift", "ember", "frost", "grain", "haste", "index", "jolly",
    "karma", "lunar", "maple", "nerve", "orbit", "pearl", "quilt", "reign",
    "solar", "thorn", "usher", "vapor", "witch", "xenon", "yield", "zones",
    "charm", "flute", "gleam", "hover", "inlet", "jumbo", "kneel", "latch",
    "mirth", "notch", "omega", "plumb", "quest", "ridge", "sworn", "truce",
    "unwed", "vivid", "weary", "xerox", "yacht", "zesty", "bloom", "crisp",
    "dwarf", "elbow", "fungi", "gloom", "hound", "igloo", "jazzy", "kayak",
]

# Tasks: each is a word to guess
TASKS = [
    {"task": f"Play Wordle! Guess the hidden 5-letter word. You have 6 attempts.", "answer": word}
    for word in WORD_LIST[:50]  # 50 tasks
]


class WordleGame:
    """Stateful Wordle game instance (one per episode)."""

    def __init__(self, answer: str = "", max_guesses: int = 6):
        self.answer = answer.lower() if answer else random.choice(WORD_LIST)
        self.max_guesses = max_guesses
        self.guesses: list[str] = []
        self.feedbacks: list[str] = []
        self.won = False
        self.done = False

    def guess(self, word: str) -> str:
        """Submit a guess. Returns colored feedback string."""
        if self.done:
            # A finished game must not change: a later guess could turn a loss into a win.
            return f"The game is over. The word was '{self.answer}'."

        word = word.lower().strip()

        if len(word) != 5:
            return f"Invalid guess '{word}' — must be exactly 5 letters."

        if not word.isalpha():
            return f"Invalid guess '{word}' — must contain only letters."

        self.guesses.append(word)
        feedback = self._compute_feedback(word)
        self.feedbacks.append(feedback)

        if word == self.answer:
            self.won = True
            self.done = True
            return f"{feedback} — Correct! The word was '{self.answer}'. Solved in {len(self.guesses)} guesses."

        if len(self.guesses) >= self.max_guesses:
            self.done = True
            return f"{feedback} — Game over! The word was '{self.answer}'."

        remaining = self.max_guesses - len(self.guesses)
        return f"{feedback} — {remaining} guesses remaining."

    def get_history(self) -> str:
        """Return all guesses and their feedback."""
        if not self.guesses:
            return "No guesses yet."
        lines = []
        for i, (g, f) in enumerate(zip(self.guesses, self.feedbacks), 1):
            lines.append(f"Guess {i}: {g} → {f}")
        lines.append(f"\nGuesses used: {len(self.guesses)}/{self.max_guesses}")
        if self.done:
            lines.append(f"Result: {'Won!' if self.won else f'Lost. Answer was {self.answer}'}")
        return "\n".join(lines)

    def _compute_feedback(self, word: str) -> str:
        """Compute Wordle feedback: 🟩 (correct), 🟨 (wrong position), ⬛ (not in word)."""
        result = ["⬛"] * 5
        answer_chars = list(self.answer)

        # First pass: exact matches (green)
        for i, (w, a) in enumerate(zip(word, self.answer)):
            if w == a:
                result[i] = "🟩"
                answer_chars[i] = None

        # Second pass: wrong position (yellow)
        for i, w in enumerate(word):
            if result[i] == "🟩":
                continue
            if w in answer_chars:
                result[i] = "🟨"
                answer_chars[answer_chars.index(w)] = None

        return "".join(result)

    @property
    def reward(self) -> float:
        """Compute reward for the current state."""
        if self.won:
            # Faster solve = higher reward
            r = 1.0 + max(0.0, 0.5 * (1.0 - len(self.guesses) / self.max_guesses))
            return r
        if self.done:
            # Partial credit based on closest guess
            best = max(
                (sum(1 for f in fb if f == "🟩") for fb in self.feedbacks),
                default=0,
            )
            return 0.1 * best  # 0.0 to 0.5 partial credit
        return 0.0
