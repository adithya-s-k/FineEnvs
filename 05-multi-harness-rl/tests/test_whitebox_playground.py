"""Browser sessions must remain independent and use the training reward unchanged."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def playground(monkeypatch):
    from smoldataenv_whitebox import playground

    monkeypatch.setattr(
        playground, "task_by_name", lambda split, name: {"folder": "/tasks/" + name}
    )
    return playground


def fake_episode(state, reward=1.05):
    episode = Mock(_calls=0, _correctness=1.0)
    episode.get_reward.return_value = reward
    episode.bash.return_value = "two rows"
    state.environment = episode
    return episode


def test_browsers_do_not_share_sandboxes_or_history(playground):
    initial = playground.PlaygroundSession()
    first, second = deepcopy(initial), deepcopy(initial)
    a, b = fake_episode(first), fake_episode(second)
    first.start("test", "one")
    second.start("test", "two")
    first.run("ls")
    assert len(first.history) == 1 and second.history == []
    first.grade("answer")
    assert first.reward == 1.05 and first.correctness == 1.0
    assert not first.active and second.active
    assert b._close.call_count == 1  # Only its own initial reset.
    a.submit_solution.assert_called_once_with("answer")
    b.submit_solution.assert_not_called()


def test_restart_and_state_expiry_close_sandbox(playground):
    state = playground.PlaygroundSession()
    episode = fake_episode(state)
    state.start("test", "one")
    state.run("ls")
    state.start("test", "two")
    assert episode._close.call_count == 2
    assert not state.history and state.reward is None
    playground.close_session(state)
    assert episode._close.call_count == 3 and not state.active


def test_ungraded_attempt_does_not_show_wrong_answer(playground):
    state = playground.PlaygroundSession()
    fake_episode(state, float("nan"))
    state.start("test", "one")
    state.grade("answer")
    assert state.reward is None and state.correctness is None
    assert "unscored" in state.status and not state.active


def test_failed_submission_still_releases_sandbox(playground):
    state = playground.PlaygroundSession()
    episode = fake_episode(state)
    state.start("test", "one")
    episode.submit_solution.side_effect = RuntimeError("expired")
    with pytest.raises(RuntimeError):
        state.grade("answer")
    assert not state.active and episode._close.call_count == 2


def test_browser_trace_escapes_tool_output(playground):
    from smoldataenv_whitebox.ui import trace

    state = playground.PlaygroundSession()
    state.history = [
        {
            "command": "<script>bad()</script>",
            "output": "<img src=x onerror=bad()>",
            "seconds": 0.1,
        }
    ]
    rendered = trace(state)
    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;" in rendered
