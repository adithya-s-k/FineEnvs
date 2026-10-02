"""WordleToolkit exposes only game moves as tools, and a pinned word cannot be changed by the model."""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env import WordleToolkit  # noqa: E402
from core.game import WORD_LIST, WordleGame  # noqa: E402


def _tools_trl_would_expose(toolkit):
    """What TRL's GRPOTrainer(environment_factory=...) turns into tools: public methods but reset/get_reward."""
    return {
        name
        for name, _ in inspect.getmembers(toolkit, predicate=inspect.ismethod)
        if name not in ("reset", "get_reward") and not name.startswith("_")
    }


def _expected(answer, words):
    game = WordleGame(answer=answer)
    for word in words:
        game.guess(word)
    return game.reward


def test_no_exposed_tool_can_choose_the_word():
    """A public set_answer() used to be exposed: the model could set the word and then guess it."""
    exposed = _tools_trl_would_expose(WordleToolkit())
    assert exposed == {"guess", "get_history", "cleanup"}  # cleanup is a no-op here


def test_reset_pins_the_answer():
    tk = WordleToolkit()
    tk.reset(answer="crane")
    assert "Correct" in tk.guess("crane")
    assert tk.reward == pytest.approx(_expected("crane", ["crane"]))


def test_reset_accepts_a_whole_dataset_row():
    """TRL calls reset(**row), so columns other than answer must not break it."""
    tk = WordleToolkit()
    tk.reset(prompt=[{"role": "user", "content": "Play Wordle!"}], answer="crane", task="Play Wordle!")
    assert "Correct" in tk.guess("crane")


def test_reset_without_an_answer_picks_a_word():
    tk = WordleToolkit()
    tk.reset(answer="crane")
    tk.reset()
    tk.guess("slate")
    assert tk._game.answer in WORD_LIST


def test_a_loss_scores_partial_credit():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    tk = WordleToolkit()
    tk.reset(answer="crane")
    for word in words:
        tk.guess(word)
    assert tk.reward == pytest.approx(_expected("crane", words))


def test_a_guess_after_a_loss_cannot_turn_it_into_a_win():
    tk = WordleToolkit()
    tk.reset(answer="crane")
    for word in ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]:
        tk.guess(word)
    lost = tk.reward
    tk.guess("crane")
    assert tk.reward == pytest.approx(lost)
