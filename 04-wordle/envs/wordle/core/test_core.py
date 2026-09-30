"""Domain tests — no torch, no GPU."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.wordle.core.game import WordleGame, feedback_pattern
from envs.wordle.core.information import information_gain, remaining_words
from envs.wordle.core.rewards import process_reward, sparse_reward
from envs.wordle.core.tasks import eval_tasks, train_tasks
from envs.wordle.core.words import ANSWERS


def test_answer_list_is_the_training_list_not_the_demo():
    assert len(ANSWERS) > 2000
    assert all(len(w) == 5 and w.isalpha() for w in ANSWERS)
    assert len(set(ANSWERS)) == len(ANSWERS)


def test_train_and_eval_do_not_overlap():
    train, ev = set(train_tasks()), set(eval_tasks())
    assert len(ev) == 200
    assert train.isdisjoint(ev)
    assert train | ev == set(ANSWERS)


def test_feedback_matches_known_colouring():
    # CRANE vs SNARE: C grey, R yellow, A green, N yellow, E green.
    assert feedback_pattern("crane", "snare") == "⬛🟨🟩🟨🟩"


def test_observe_does_not_name_the_answer():
    game = WordleGame(answer="mango", max_guesses=1)
    text = game.guess("apple")
    assert "mango" not in text
    assert "the word was" not in text.lower()
    assert game.done
    assert not game.won


def test_invalid_guess_does_not_consume_a_turn():
    game = WordleGame(answer="crane")
    game.guess("no")
    game.guess("12345")
    assert game.guesses == []
    assert game.invalid_count == 2
    assert not game.done


def test_solving_collapses_the_remaining_set():
    before = ANSWERS
    gain = information_gain("zonal", "🟩🟩🟩🟩🟩", before)
    assert gain == pytest.approx(math.log2(len(ANSWERS)))
    left = remaining_words(["zonal"], ["🟩🟩🟩🟩🟩"])
    assert left == ("zonal",)


def test_a_grey_opener_removes_those_letters():
    pattern = feedback_pattern("xylyl", "crane")
    left = remaining_words(["xylyl"], [pattern])
    assert len(left) < len(ANSWERS)
    assert all("x" not in w and "y" not in w for w in left)


def test_vectorized_patterns_match_the_scalar_colouring():
    from envs.wordle.core.information import _pattern_codes, _pack_pattern, _ANSWER_CODES

    guesses = ["crane", "slate", "xylyl", "zonal", "apple"]
    answers = ANSWERS[:80]
    codes = _ANSWER_CODES[:80]
    for guess in guesses:
        packed = _pattern_codes(guess, codes)
        for i, answer in enumerate(answers):
            assert packed[i] == _pack_pattern(feedback_pattern(guess, answer))


def test_process_reward_separates_two_failed_episodes():
    # Both fail. One guess is the answer's anagram-ish split, the other is
    # disjoint junk. They must not share a reward — that is the cliff.
    answer = "crane"
    useful = WordleGame(answer=answer)
    useful.guess("crate")
    noise = WordleGame(answer=answer)
    noise.guess("xylyl")
    r_useful = process_reward(
        won=False,
        n_guesses=1,
        guesses=useful.guesses,
        patterns=useful.patterns,
    )
    r_noise = process_reward(
        won=False,
        n_guesses=1,
        guesses=noise.guesses,
        patterns=noise.patterns,
    )
    assert r_useful > r_noise
    assert sparse_reward(won=False, n_guesses=1) == 0.0
    assert sparse_reward(won=False, n_guesses=1) == sparse_reward(won=False, n_guesses=6)
