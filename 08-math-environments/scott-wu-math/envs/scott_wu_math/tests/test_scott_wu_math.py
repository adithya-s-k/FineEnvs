import json
import random
import time
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scott_wu_math.game import expr as expr_module
from scott_wu_math.game.expr import ExpressionError, check, evaluate, leaves, parse, to_text
from scott_wu_math.game.reward import extract_answer, grade
from scott_wu_math.game.solver import solve
from scott_wu_math.game.tasks import LEVELS, make_task
from scott_wu_math.models import CardAction
from scott_wu_math.server import environment as environment_module
from scott_wu_math.server.environment import InvalidReset, ScottWuMathEnvironment

TASKS = Path(__file__).parents[1] / "data/tasks"


def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (TASKS / f"{split}.jsonl").read_text().splitlines()]


VIDEO = load("video")


@pytest.mark.parametrize("task", VIDEO, ids=lambda t: t["id"])
def test_the_video_hands_check_out(task):
    values = [c["rank"] for c in task["cards"]]
    assert check(task["solution"], values, task["target"]).exact
    assert solve(values, task["target"]) is not None


def test_symbols_letters_and_equals_are_accepted():
    values = [13, 12, 11, 10, 7, 9, 5, 5]
    assert check("((K − Q) + (J − 10) + 7) × 9 × 5 × 5 = 2025", values, 2025).exact


def test_unicode_operators_and_spaces_are_accepted():
    assert check("4\u00a0∗\u00a0（7 － 8 ／ 8）", [4, 7, 8, 8], 24).exact
    assert check("6 ＋ 6\u202f＋ 6 ＋ 6", [6, 6, 6, 6], 24).exact
    assert check("8 ∕ (3 − 8 ∕ 3)", [3, 3, 8, 8], 24).exact


@pytest.mark.parametrize("raised", [ValueError("source code string cannot contain null bytes"), RecursionError()])
def test_parser_failures_are_invalid_answers(monkeypatch, raised):
    """Python 3.11 raises ValueError for a null byte; very deep nesting can raise RecursionError."""

    def fail(*args, **kwargs):
        raise raised

    monkeypatch.setattr(expr_module.ast, "parse", fail)
    with pytest.raises(ExpressionError):
        parse("1 + 2")
    assert grade("1 + 2", [1, 2], 3)["reward"] == 0


def test_null_bytes_and_deep_nesting_end_the_episode_with_zero():
    env = ScottWuMathEnvironment()
    for expression in ["((13 - 12) + (11 - 10) + 7) * 9 * 5 * 5\x00", "(" * 198 + "2025" + ")" * 198]:
        env.reset(split="video", index=0)
        out = env.step(CardAction(expression=expression))
        assert out.done and out.reward == 0.0 and not out.result["valid"] and out.result["error"]


@pytest.mark.parametrize(
    "expr, error",
    [
        ("-13 + 12 + 11 + 10 + 7 + 9 + 5 + 5", "unary"),
        ("13 ** 2", "powers"),
        ("1.5 * 2", "whole card values"),
        ("abs(3)", "only"),
        ("", "empty"),
    ],
)
def test_illegal_expressions_are_rejected(expr, error):
    with pytest.raises(Exception, match=error):
        parse(expr)


def test_every_card_once():
    values = [4, 7, 8, 8]
    assert check("4 * (7 - 8 / 8)", values, 24).exact
    missing = check("4 * 7 - 8 + 8", values[:3] + [1], 24)
    assert not missing.uses_all_cards and missing.extra == [8] and missing.missing == [1]
    assert check("48 / 2", values, 24).extra  # no joining cards into longer numbers


def test_division_is_exact_and_division_by_zero_scores_zero():
    assert check("8 / (3 - 8 / 3)", [3, 3, 8, 8], 24).value == Fraction(24)
    assert grade("8 / (3 - 3) + 8 + 8", [3, 3, 8, 8], 24)["reward"] == 0


def test_shaped_reward():
    values = [13, 12, 11, 10, 7, 9, 5, 5]
    assert grade("(13 - 12 + 11 - 10 + 7) * 9 * 5 * 5", values, 2025)["reward"] == 1.0
    near = grade("(13 - 12 + 11 - 10 + 7) * 9 * 5 * 5", values, 2030)
    assert 0 < near["reward"] <= 0.3 and near["valid"]
    assert grade("(13 - 12 + 11 - 10 + 7) * 9 * 5 * 5", values, 2030, shaped=False)["reward"] == 0
    assert grade("13 + 12", values, 25)["reward"] == 0  # unused cards


def test_extract_answer_ignores_thinking():
    text = "<think><answer>1+1</answer></think> so <answer>4 * (7 - 8 / 8)</answer>"
    assert extract_answer(text) == "4 * (7 - 8 / 8)"
    assert extract_answer("no tags") is None


def test_to_text_keeps_value_and_cards():
    for expr in ["(1 - (2 - 3)) / (4 / 5)", "1 - 2 - 3", "1 / (2 * 3)", "(1 + 2) * 3", "8 / (3 - 8 / 3)"]:
        tree = parse(expr)
        again = parse(to_text(tree))
        assert evaluate(again) == evaluate(tree) and sorted(leaves(again)) == sorted(leaves(tree))


@pytest.mark.parametrize("level", list(LEVELS))
def test_generated_tasks_are_solvable(level):
    rng = random.Random(1)
    for _ in range(5):
        task = make_task(rng, level=level)
        lvl = LEVELS[level]
        assert len(task["cards"]) == lvl.n_cards and lvl.low <= task["target"] <= lvl.high
        assert check(task["solution"], [c["rank"] for c in task["cards"]], task["target"]).exact


def test_episode_is_one_graded_step():
    env = ScottWuMathEnvironment()
    obs = env.reset(split="video", index=0)
    assert obs.target == 2025 and len(obs.cards) == 8 and obs.solution is None and not obs.done
    assert "Target: 2025" in obs.prompt
    out = env.step(CardAction(expression="((13 - 12) + (11 - 10) + 7) * 9 * 5 * 5"))
    assert out.done and out.reward == 1.0 and out.solution
    again = env.step(CardAction(expression="((13 - 12) + (11 - 10) + 7) * 9 * 5 * 5"))
    assert again.reward == 0.0


def test_custom_target_and_task_api_hide_solutions():
    env = ScottWuMathEnvironment()
    obs = env.reset(target=6843, seed=3)
    assert obs.level == "custom" and len(obs.cards) == 9
    assert all("solution" not in t for t in env.list_tasks("test"))
    assert env.num_tasks("test") == 200 and env.num_tasks("train") == 3639


def test_train_shares_no_deal_with_test():
    """Grading sees the target and the card values, so that is what must not repeat (suits do not matter)."""

    def key(task):
        return task["target"], tuple(sorted(c["rank"] for c in task["cards"]))

    test, train = {key(t) for t in load("test")}, [key(t) for t in load("train")]
    assert len(set(train)) == len(train) and not test & set(train)


@pytest.mark.parametrize("level", list(LEVELS))
def test_the_server_deals_every_level_within_its_limits(level):
    env = ScottWuMathEnvironment()
    for seed in range(3):
        obs = env.reset(level=level, seed=seed)
        assert len(obs.cards) == LEVELS[level].n_cards and obs.level == level


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "test", "index": 200},
        {"split": "test", "index": -1},
        {"split": "video", "index": 2},
        {"split": "valid"},
        {"level": "five-digit"},
        {"target": 999_983},
        {"target": 999_983, "n_cards": 9},
        {"target": 6843, "n_cards": 12},
        {"target": 24, "cards": list(range(1, 13))},
        {"target": 24, "cards": [14, 1, 2, 3]},
        {"reward_mode": "dense"},
    ],
)
def test_resets_the_server_cannot_deal_are_refused_at_once(kwargs):
    started = time.monotonic()
    with pytest.raises(InvalidReset):
        ScottWuMathEnvironment().reset(**kwargs)
    assert time.monotonic() - started < 0.5


def test_a_custom_hand_with_no_solution_returns_within_the_solver_limit():
    started = time.monotonic()
    hand = [1, 1, 1, 1, 13, 13, 13, 13, 12]  # cannot make 9973; a full search takes about 5 s
    obs = ScottWuMathEnvironment().reset(target=9973, cards=hand)
    assert time.monotonic() - started < environment_module.DEAL_SECONDS + 1.0
    assert len(obs.cards) == 9 and obs.target == 9973


def test_fresh_custom_deals_stop_at_the_budget(monkeypatch):
    monkeypatch.setattr(environment_module, "DEAL_SECONDS", 0.005)
    monkeypatch.setattr(environment_module, "MAX_DEALS", 100_000)
    monkeypatch.setattr(environment_module, "DEAL_BUDGET", 0.3)
    started = time.monotonic()
    try:
        ScottWuMathEnvironment().reset(target=9973, n_cards=9)
    except InvalidReset as exc:
        assert "0.3 s" in str(exc)
    assert time.monotonic() - started < 1.0


def test_the_solver_stops_at_its_time_limit_for_a_big_hand():
    started = time.monotonic()
    solve(list(range(1, 13)), 999_983, time_limit=0.3)
    assert time.monotonic() - started < 1.0


def test_http_app_serves_the_page_and_task_routes():
    from scott_wu_math.server.app import app

    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert "Scott Wu Math" in client.get("/readme.md").text
    splits = client.get("/scott_wu_math/splits").json()
    assert {s["name"] for s in splits} == {"train", "test", "video"}
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"split": "video", "index": 1}})
        assert ws.receive_json()["data"]["observation"]["target"] == 6843
        ws.send_json({"type": "step", "data": {"expression": "(((12 - 10) * (6 + 4) * 10 + 8) * 11 - 7) * 3"}})
        assert ws.receive_json()["data"]["reward"] == 1.0


def test_http_and_ws_answer_a_bad_reset_with_a_client_error():
    from scott_wu_math.server.app import app

    client = TestClient(app)
    for body in [{"split": "test", "index": 200}, {"split": "test", "index": -1}, {"target": 999_983, "n_cards": 12}]:
        res = client.post("/reset", json=body)
        assert res.status_code == 400 and res.json()["detail"]
    assert client.post("/reset", json={"split": "test", "index": 199}).status_code == 200
    assert client.post("/scott_wu_math/task", json={"split": "test", "index": -1}).status_code == 400
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"split": "test", "index": -1}})
        msg = ws.receive_json()
        assert msg["type"] == "error" and "out of range" in msg["data"]["message"]
