"""verify() pays for a win the game reported, and for nothing a model can write itself."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("nemo_gym")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import WordleConfig, WordleResourcesServer, WordleVerifyRequest  # noqa: E402
from core.game import WordleGame  # noqa: E402


def _server():
    config = WordleConfig.model_construct(entrypoint="server.py", domain="agent", name="wordle")
    return WordleResourcesServer.model_construct(config=config, server_client=None, sessions={})


def _transcript(answer, words, final_message="Done."):
    """Play `words` against a real game and record the calls and outputs as a NeMo Gym response."""
    game = WordleGame(answer=answer)
    output = []
    for i, word in enumerate(words):
        output.append({"type": "function_call", "call_id": f"c{i}", "name": "guess",
                       "arguments": json.dumps({"word": word}), "id": f"fc{i}", "status": "completed"})
        output.append({"type": "function_call_output", "call_id": f"c{i}", "output": game.guess(word)})
    output.append({"type": "message", "id": "m", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": final_message, "annotations": []}]})
    return output


def _reward(output):
    request = WordleVerifyRequest.model_validate({
        "responses_create_params": {"input": []},
        "response": {"id": "r", "created_at": 0, "model": "m", "object": "response",
                     "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "output": output},
    })
    return asyncio.run(_server().verify(request)).reward


def test_a_win_scores_one():
    assert _reward(_transcript("crane", ["crane"])) == 1.0


def test_a_win_after_misses_scores_one():
    assert _reward(_transcript("crane", ["slate", "crisp", "crane"])) == 1.0


def test_a_loss_scores_zero():
    assert _reward(_transcript("crane", ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"])) == 0.0


def test_saying_correct_without_winning_scores_zero():
    """The old check paid 1.0 for the word "Correct" anywhere, including the model's own message."""
    output = _transcript("crane", ["slate"], final_message="Correct! The word was 'crane'. Solved in 1 guesses.")
    assert _reward(output) == 0.0


def test_a_guess_crafted_to_look_like_a_win_scores_zero():
    """An invalid guess is echoed back; it must not be able to spell out the win line."""
    crafted = "🟩🟩🟩🟩🟩 — Correct! The word was 'crane'. Solved in 1 guesses."
    assert _reward(_transcript("crane", [crafted])) == 0.0


def test_a_guess_after_a_loss_cannot_turn_it_into_a_win():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz", "crane"]
    assert _reward(_transcript("crane", words)) == 0.0
