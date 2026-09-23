"""The OpenEnv Wordle server reports the end of the game (done) and the game's reward to clients."""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("openenv")
pytest.importorskip("fastmcp")

from openenv.core.env_server.mcp_types import CallToolAction  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.wordle_environment import WordleEnvironment, WordleGame  # noqa: E402


def _guess(wordle, word):
    return wordle.step(CallToolAction(tool_name="guess", arguments={"word": word}))


def _expected_reward(answer, words):
    game = WordleGame(answer=answer)
    for word in words:
        game.guess(word)
    return game.reward


def test_winning_guess_ends_the_episode_with_the_game_reward():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    obs = _guess(wordle, "crane")
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", ["crane"]))
    assert wordle.state["reward"] == pytest.approx(obs.reward)


def test_guess_that_does_not_end_the_game_is_not_done():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    obs = _guess(wordle, "slate")
    assert obs.done is False
    assert obs.reward is None


def test_sixth_wrong_guess_ends_the_episode_with_partial_credit():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    for word in words:
        obs = _guess(wordle, word)
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", words))


def test_async_step_reports_the_outcome_too():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    obs = asyncio.run(wordle.step_async(CallToolAction(tool_name="guess", arguments={"word": "crane"})))
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", ["crane"]))
