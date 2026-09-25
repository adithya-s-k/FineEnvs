"""Guess parsing and scoring for the GEM Wordle environment."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("wordle_gem_env", Path(__file__).resolve().parents[1] / "env.py")
env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<guess>crane</guess>", "crane"),
        ("Thinking about vowels first. <GUESS> Slate </GUESS>", "slate"),
        ("crane", "crane"),
        ("  Crane.\n", "crane"),
        ("**crane**", "crane"),
    ],
)
def test_accepts_a_tag_or_a_bare_word(text, expected):
    assert env._extract_guess(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Could you share your last guess?",
        "I need to know which word I guessed before choosing the next one.",
        "My guess is crane",
        "",
    ],
)
def test_does_not_take_words_from_prose(text):
    assert env._extract_guess(text) == ""


def test_lenient_parsing_keeps_the_previous_fallback():
    assert env._extract_guess("My guess is crane", lenient=True) == "crane"
    assert env._extract_guess("Could you share your last guess?", lenient=True) == "guess"


def test_prose_reply_is_not_played_as_a_guess():
    wordle = env.WordleGemEnv(answer="crane")
    wordle.reset()
    obs, reward, terminated, truncated, info = wordle.step("Could you share your last guess?")
    assert obs.startswith("Could not parse")
    assert wordle._game.guesses == []
    assert info["error_count"] == 1


def test_lenient_env_plays_the_word_from_prose():
    wordle = env.WordleGemEnv(answer="crane", lenient_parsing=True)
    wordle.reset()
    wordle.step("Could you share your last guess?")
    assert wordle._game.guesses == ["guess"]
    assert wordle.spawn(same_state=True)._lenient_parsing is True


def test_a_finished_game_accepts_no_more_guesses():
    lost = env.WordleGame(answer="crane")
    for word in ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]:
        lost.guess(word)
    reward = lost.reward
    assert lost.done and not lost.won
    assert lost.guess("crane").startswith("The game is over")
    assert not lost.won and lost.reward == pytest.approx(reward) and len(lost.guesses) == 6

    won = env.WordleGame(answer="crane")
    won.guess("crane")
    reward = won.reward
    won.guess("slate")
    assert won.reward == pytest.approx(reward) and won.guesses == ["crane"]


def test_tagged_win_gets_the_game_reward():
    wordle = env.WordleGemEnv(answer="crane")
    wordle.reset()
    obs, reward, terminated, truncated, info = wordle.step("<guess>crane</guess>")
    assert terminated and info["won"]
    assert reward == pytest.approx(1.0 + 0.5 * (1 - 1 / 6))
