"""One Wordle episode a person can read.

Uses the frequency opener CRANE, then the first remaining answer in list
order. Not a policy — a trajectory, so you can see what the environment
actually returns before anyone trains against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.wordle.core.game import WordleGame
from envs.wordle.core.information import remaining_words
from envs.wordle.core.rewards import process_reward, sparse_reward
from envs.wordle.core.words import ANSWERS


def play(answer: str, opener: str = "crane") -> None:
    game = WordleGame(answer=answer)
    print(f"answer held out of the observation: {answer}")
    print(game.observe())
    guess = opener if opener in ANSWERS else ANSWERS[0]
    while not game.done:
        print(f"\n-- guess {guess} --")
        print(game.guess(guess))
        if game.done:
            break
        left = remaining_words(game.guesses, game.patterns)
        # Next guess: a remaining word. The training policy does not get this
        # list; the rollout does, so a human can see the set shrinking.
        guess = left[0] if left else "zzzzz"
        print(f"  remaining {len(left)}")
    sparse = sparse_reward(won=game.won, n_guesses=len(game.guesses), invalid_count=game.invalid_count)
    process = process_reward(
        won=game.won,
        n_guesses=len(game.guesses),
        guesses=game.guesses,
        patterns=game.patterns,
        invalid_count=game.invalid_count,
    )
    print(f"\nsparse={sparse:.3f}  process={process:.3f}  won={game.won}")
    leaked = "the word was" in game.observe().lower()
    print("answer leaked in the observation:" , leaked)


if __name__ == "__main__":
    play(sys.argv[1] if len(sys.argv) > 1 else "zonal")
