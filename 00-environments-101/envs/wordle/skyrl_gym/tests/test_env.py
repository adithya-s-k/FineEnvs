"""Guess parsing and scoring for the SkyRL Gym Wordle environment."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("wordle_skyrl_env", Path(__file__).resolve().parents[1] / "env.py")
env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<guess>crane</guess>", "crane"),
        ("crane", "crane"),
        ("Crane!", "crane"),
    ],
)
def test_accepts_a_tag_or_a_bare_word(text, expected):
    assert env._extract_guess(text) == expected


@pytest.mark.parametrize("text", ["Could you share your last guess?", "guess: crane then maybe slate", ""])
def test_does_not_take_words_from_prose(text):
    assert env._extract_guess(text) == ""


def test_lenient_parsing_keeps_the_previous_fallback():
    assert env._extract_guess("Could you share your last guess?", lenient=True) == "guess"


def _get(output, key):
    return output[key] if isinstance(output, dict) else getattr(output, key)


def test_prose_reply_is_not_played_as_a_guess():
    wordle = env.WordleSkyRLEnv(answer="crane")
    wordle.init("prompt")
    out = wordle.step("Could you share your last guess?")
    assert _get(out, "observations")[0]["content"].startswith("Could not parse")
    assert wordle._game.guesses == []


def test_tagged_win_gets_the_game_reward():
    wordle = env.WordleSkyRLEnv(answer="crane")
    wordle.init("prompt")
    out = wordle.step("<guess>crane</guess>")
    assert _get(out, "done") is True
    assert _get(out, "reward") == pytest.approx(1.0 + 0.5 * (1 - 1 / 6))
