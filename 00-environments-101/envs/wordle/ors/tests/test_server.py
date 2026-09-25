"""The ORS Wordle server pins the task's word, ends on the right guess, and cannot be replayed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("openreward")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import GuessInput, HistoryInput, WordleORS  # noqa: E402
from core.game import WordleGame  # noqa: E402


def _env(answer="crane"):
    env = WordleORS(task_spec={"answer": answer})
    env.setup()
    return env


def _expected(answer, words):
    game = WordleGame(answer=answer)
    for word in words:
        game.guess(word)
    return game.reward


def test_the_task_answer_is_the_hidden_word():
    out = _env("crane").guess(GuessInput(word="crane"))
    assert out.finished is True
    assert "Correct" in out.blocks[0].text


def test_a_guess_that_does_not_end_the_game_is_not_finished():
    out = _env().guess(GuessInput(word="slate"))
    assert out.finished is False
    assert out.reward == 0.0


def test_a_win_reports_the_game_reward():
    env = _env()
    env.guess(GuessInput(word="slate"))
    out = env.guess(GuessInput(word="crane"))
    assert out.finished is True
    assert out.reward == pytest.approx(_expected("crane", ["slate", "crane"]))


def test_six_misses_finish_with_partial_credit():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    env = _env()
    for word in words:
        out = env.guess(GuessInput(word=word))
    assert out.finished is True
    assert out.reward == pytest.approx(_expected("crane", words))


def test_a_guess_after_a_loss_cannot_turn_it_into_a_win():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    env = _env()
    for word in words:
        lost = env.guess(GuessInput(word=word))
    out = env.guess(GuessInput(word="crane"))
    assert out.finished is True
    assert out.reward == pytest.approx(lost.reward)
    assert "Correct" not in out.blocks[0].text


def test_tools_before_setup_report_no_game():
    env = WordleORS(task_spec={"answer": "crane"})
    assert env.guess(GuessInput(word="crane")).finished is False
    assert "No game" in env.get_history(HistoryInput()).blocks[0].text
