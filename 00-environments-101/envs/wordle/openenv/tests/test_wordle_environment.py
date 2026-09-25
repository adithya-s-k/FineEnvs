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


def _text(obs):
    result = obs.result
    return result.data if hasattr(result, "data") else result


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


def test_reset_game_does_not_abandon_a_game_in_progress():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    _guess(wordle, "slate")
    obs = wordle.step(CallToolAction(tool_name="reset_game", arguments={}))
    assert "in progress" in _text(obs)
    assert obs.done is False
    obs = _guess(wordle, "crane")
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", ["slate", "crane"]))


def test_guesses_after_the_game_ends_change_nothing():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    reward = _guess(wordle, "crane").reward
    obs = _guess(wordle, "slate")
    assert obs.done is True
    assert wordle.state["reward"] == pytest.approx(reward)


def test_the_final_reward_is_reported_once():
    """Steps after the end earn nothing, so summing per-step rewards gives the game's reward exactly."""
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    rewards = [_guess(wordle, "slate").reward, _guess(wordle, "crane").reward]
    after = [
        _guess(wordle, "slate"),
        wordle.step(CallToolAction(tool_name="get_history", arguments={})),
    ]
    assert all(obs.done is True for obs in after)
    assert all(obs.reward is None for obs in after)
    rewards += [obs.reward for obs in after]
    assert sum(r for r in rewards if r is not None) == pytest.approx(
        _expected_reward("crane", ["slate", "crane"])
    )


def test_a_new_game_reports_its_own_reward():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    _guess(wordle, "crane")
    wordle.step(CallToolAction(tool_name="reset_game", arguments={}))
    answer = wordle._game.answer
    obs = _guess(wordle, answer)
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward(answer, [answer]))


def test_reset_starts_a_fresh_episode_that_reports_its_reward():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    _guess(wordle, "crane")
    wordle.reset(answer="crane")
    obs = _guess(wordle, "crane")
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", ["crane"]))


def test_async_step_reports_the_outcome_too():
    wordle = WordleEnvironment()
    wordle.reset(answer="crane")
    obs = asyncio.run(wordle.step_async(CallToolAction(tool_name="guess", arguments={"word": "crane"})))
    assert obs.done is True
    assert obs.reward == pytest.approx(_expected_reward("crane", ["crane"]))
