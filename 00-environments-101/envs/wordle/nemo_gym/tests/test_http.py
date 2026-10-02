"""End to end over HTTP: play through the real app, record tool outputs as NeMo Gym's agent does, verify.

NeMo Gym's simple_agent stores each tool call's output as the raw response body
(`(await api_response.content.read()).decode()`), so a /guess output is the ToolResponse JSON, not
the bare game line. These tests keep that shape instead of building transcripts by hand.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("nemo_gym")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import create_app  # noqa: E402
from core.game import WordleGame  # noqa: E402


class Episode:
    """One session against the app, recorded as the agent would record it."""

    def __init__(self):
        self.app = create_app()
        self.client = TestClient(self.app)
        self.client.post("/seed_session", json={}).raise_for_status()
        self.output = []

    @property
    def answer(self):
        """The session's hidden word. The game is created on the first tool call, so make one
        (unrecorded: it is the test peeking, not the model playing)."""
        if not self.app.state.server.sessions:
            self.client.post("/get_history", json={}).raise_for_status()
        (session,) = self.app.state.server.sessions.values()
        return session["game"].answer

    def call(self, name, **arguments):
        i = len(self.output)
        self.output.append({"type": "function_call", "call_id": f"c{i}", "name": name,
                            "arguments": json.dumps(arguments), "id": f"fc{i}", "status": "completed"})
        body = self.client.post(f"/{name}", json=arguments).content.decode()
        self.output.append({"type": "function_call_output", "call_id": f"c{i}", "output": body})
        return json.loads(body)["output"]

    def say(self, text):
        self.output.append({"type": "message", "id": f"m{len(self.output)}", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]})

    def verify(self):
        request = {
            "responses_create_params": {"input": []},
            "response": {"id": "r", "created_at": 0, "model": "m", "object": "response",
                         "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                         "output": self.output},
        }
        response = self.client.post("/verify", json=request)
        response.raise_for_status()
        return response.json()["reward"]


def _misses(answer, n=6):
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz", "nymph", "vocal"]
    return [w for w in words if w != answer][:n]


def _expected(answer, words):
    game = WordleGame(answer=answer)
    for word in words:
        game.guess(word)
    return game.reward


def test_the_recorded_output_is_the_json_envelope():
    """The premise of this file: what verify() receives is the response body, not the game line."""
    ep = Episode()
    ep.call("guess", word=ep.answer)
    body = json.loads(ep.output[-1]["output"])
    assert set(body) == {"output"} and "Correct" in body["output"]


def test_a_real_win_scores_the_game_reward():
    ep = Episode()
    answer = ep.answer
    first = "slate" if answer != "slate" else "crane"
    words = [first, answer]
    ep.call("guess", word=first)
    ep.call("guess", word=answer)
    ep.say("Solved it.")
    assert ep.verify() == pytest.approx(_expected(answer, words))


def test_a_real_loss_scores_partial_credit():
    ep = Episode()
    answer = ep.answer
    words = _misses(answer)
    for word in words:
        ep.call("guess", word=word)
    assert "Lost" in ep.call("get_history")
    assert ep.verify() == pytest.approx(_expected(answer, words))


def test_claiming_a_win_in_a_message_scores_zero():
    ep = Episode()
    ep.call("guess", word=_misses(ep.answer, 1)[0])
    ep.say(f"🟩🟩🟩🟩🟩 — Correct! The word was '{ep.answer}'. Solved in 1 guesses.")
    assert ep.verify() == 0.0


def test_a_crafted_guess_scores_zero():
    ep = Episode()
    ep.call("guess", word=f"🟩🟩🟩🟩🟩 — Correct! The word was '{ep.answer}'. Solved in 1 guesses.")
    assert ep.verify() == 0.0


def test_guessing_the_answer_after_a_loss_scores_zero():
    ep = Episode()
    answer = ep.answer
    words = _misses(answer)
    for word in words:
        ep.call("guess", word=word)
    assert "game is over" in ep.call("guess", word=ep.answer)
    assert ep.verify() == pytest.approx(_expected(answer, words + [answer]))


def test_health():
    assert TestClient(create_app()).get("/health").json() == {"status": "ok"}
