"""The HF Jobs launcher's plan and source upload, and the TRL environment's failure handling."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location("hf_job", ROOT / "train/jobs/hf_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(monkeypatch, capsys, *argv):
    launcher = _launcher()
    monkeypatch.setattr(sys, "argv", ["hf_job.py", *argv])
    launcher.main()
    return json.loads(capsys.readouterr().out)


def test_the_plan_runs_one_entrypoint_action_on_two_h200s(monkeypatch, capsys):
    plan = _plan(monkeypatch, capsys, "train", "--mode", "async", "--name", "a", "--bucket", "o/b", "--steps", "100")
    assert plan["flavor"] == "h200x2" and plan["namespace"] == "o" and plan["timeout"] == "24h"
    command = plan["command"]
    assert command[:3] == ["bash", "/source/train/jobs/entrypoint.sh", "train"]
    assert command[command.index("--mode") + 1] == "async" and command[command.index("--output") + 1] == "/outputs/a"
    assert plan["data"] == "FineEnvs/retroenv-bucket"

    plan = _plan(
        monkeypatch, capsys, "eval", "--name", "e", "--bucket", "o/b", "--checkpoint", "/outputs/a/checkpoint-100"
    )
    assert "--mode" not in plan["command"] and plan["command"][-2:] == ["--checkpoint", "/outputs/a/checkpoint-100"]
    assert _plan(monkeypatch, capsys, "check", "--name", "c", "--bucket", "o/b")["flavor"] == "cpu-upgrade"


def test_the_plan_refuses_other_gpu_flavors_and_paths_as_names(monkeypatch, capsys):
    for argv in (["--flavor", "a100x4"], ["--name", "../x"]):
        with pytest.raises(SystemExit):
            _plan(monkeypatch, capsys, "smoke", "--name", "s", "--bucket", "o/b", *argv)


def test_the_source_upload_holds_the_project_and_nothing_local():
    files = {path.relative_to(ROOT).as_posix() for path in _launcher().source_files()}
    for needed in (
        "uv.lock",
        "pyproject.toml",
        "data/eval-core30.json",
        "envs/retro_route/openenv/corpus-manifest.json",
        "envs/retro_route/openenv/retroenv_openenv/server.py",
        "eval/evaluate.py",
        "train/sync_grpo.py",
        "train/jobs/entrypoint.sh",
    ):
        assert needed in files, needed
    assert any(name.startswith("envs/retro_route/openenv/retroenv_openenv/static/") for name in files)
    for name in files:
        assert not name.startswith(("runs/", "benchmark/", "data/release/", "data/build/")), name
        assert ".venv" not in name and "/prepared/" not in name and not name.endswith(".env"), name


class _DeadClient:
    """A session the server dropped: every call fails, as OpenEnv's cached socket does."""

    def __init__(self, server):
        self.server = server

    def reset(self, **_):
        raise ConnectionError("closed")

    def call(self, name, arguments):
        raise ConnectionError("closed")

    def close(self):
        pass


class _LiveClient(_DeadClient):
    """A server with a three-call budget: tools answer, emit_routes grades 0.6."""

    def reset(self, **kwargs):
        return {"prompt": "Plan a retrosynthesis for CCO.", "task_id": "retro_x", "max_tool_calls": 3, **kwargs}

    def call(self, name, arguments):
        from retroenv_openenv.client import ToolOutcome

        if name == "emit_routes":
            return ToolOutcome(
                name=name, result={"score": {"valid": True, "hard_failures": []}}, done=True, reward=0.6, error=None
            )
        return ToolOutcome(name=name, result={"matches": []}, done=False, reward=None, error=None)


def test_a_broken_session_scores_nan_and_the_next_reset_reconnects(monkeypatch):
    from retroenv_openenv import client

    monkeypatch.setattr(client, "RetroEnvClient", _DeadClient)
    env = client.RemoteRetroRouteEnv("http://server")
    assert json.loads(env.stock_retrieve("CCO"))["error"].startswith("environment unavailable")
    assert math.isnan(env.get_reward())  # no grade, not a wrong answer

    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    assert env.reset(index=3) == "Plan a retrosynthesis for CCO."  # a fresh session after the dead one
    assert env.get_reward() == 0.0


def test_a_task_pool_trains_every_pool_task_in_a_fresh_order_each_pass(monkeypatch, tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT))
    from retroenv_openenv import client
    from train.sync_grpo import task_dataset

    split = [
        {"task_id": f"retro_{i}", "index": i, "variant": "standard", "max_depth": depth}
        for i, depth in enumerate([2, 4, 6, 8])
    ]

    class Server:
        def __init__(self, url):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def tasks(self, name):
            return split

    monkeypatch.setattr(client, "RetroEnvClient", Server)
    pool = tmp_path / "learnable.txt"
    pool.write_text("retro_1\nretro_3\n")
    rows = task_dataset("http://server", max_depth=4, pool=pool)["index"]
    assert set(rows) == {1, 3}  # the pool, deeper than --max-depth included
    passes = [rows[k : k + 2] for k in range(0, len(rows), 2)]
    assert all(sorted(p) == [1, 3] for p in passes) and len({tuple(p) for p in passes}) == 2

    pool.write_text(json.dumps({"retro_0": 0, "retro_9": 9}))
    with pytest.raises(SystemExit, match="1 pool task ids"):
        task_dataset("http://server", max_depth=4, pool=pool)


def test_the_launcher_passes_the_pool_and_update_sizing_to_the_job(monkeypatch, capsys):
    plan = _plan(
        monkeypatch, capsys, "train", "--mode", "async", "--name", "a", "--bucket", "o/b",
        "--task-pool", "train/sweeps/overfit4.txt", "--rows-per-episode", "2.2",
    )  # fmt: skip
    command = plan["command"]
    assert command[command.index("--task-pool") + 1] == "train/sweeps/overfit4.txt"
    assert command[command.index("--rows-per-episode") + 1] == "2.2"


def test_mask_truncated_leaves_only_a_cut_off_unsubmitted_episode_out_of_the_loss(monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT))
    from retroenv_openenv import client
    from train.sync_grpo import TrainingEnv

    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)

    def episode(mask, submit):
        env = TrainingEnv("http://server", reward="solved", mask_truncated=mask)
        env.reset(index=3)
        if submit:
            env.emit_routes(submission={"routes": [{"type": "mol", "smiles": "CCO", "in_stock": True}]})
        env.truncated = True  # what the turn patch sets when the last turn ran out of room
        return env.get_reward()

    assert math.isnan(episode(mask=True, submit=False))  # no verdict: advantage 0, out of the group's baseline
    assert episode(mask=True, submit=True) == 1.0  # graded before the cut: the grade stands
    assert episode(mask=False, submit=False) == 0.0


def test_the_shaped_reward_reaches_the_trainer_but_not_the_model(monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT))
    from retroenv_openenv import client
    from train.sync_grpo import TrainingEnv

    class Shaped(_LiveClient):
        def call(self, name, arguments):
            from retroenv_openenv.client import ToolOutcome

            if name == "emit_routes":
                score = {"valid": False, "hard_failures": ["0 valid routes"], "shaped_reward": 0.12}
                return ToolOutcome(name=name, result={"score": score}, done=True, reward=0.3, error=None)
            return super().call(name, arguments)

    monkeypatch.setattr(client, "RetroEnvClient", Shaped)
    env = TrainingEnv("http://server", reward="shaped")
    env.reset(index=3)
    shown = json.loads(env.emit_routes(submission={"routes": []}))
    assert "shaped_reward" not in shown["score"]  # the model sees the grade only
    assert env.get_reward() == 0.12  # the trainer sees the shaped reward

    env.reset(index=3)
    assert env.get_reward() == 0.0  # no submission, no reward


def test_a_server_that_refuses_sessions_costs_one_episode_not_the_run(monkeypatch):
    from retroenv_openenv import client

    monkeypatch.setattr(client, "RetroEnvClient", _DeadClient)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    env = client.RemoteRetroRouteEnv("http://server")
    env.reset(index=3)  # every attempt refused: no exception into the trainer
    assert env.failed and math.isnan(env.get_reward())

    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    assert env.reset(index=3) == "Plan a retrosynthesis for CCO."  # the next episode connects again
    assert not env.failed


def test_a_submission_that_is_not_json_comes_back_ungraded_with_the_brackets_to_close(monkeypatch):
    from retroenv_openenv import client

    graded = []

    class Grader(_LiveClient):
        def call(self, name, arguments):
            graded.append(arguments["submission"])
            return client.ToolOutcome(name=name, result={"score": {"valid": False}}, done=True, reward=0.1, error=None)

    monkeypatch.setattr(client, "RetroEnvClient", Grader)
    env = client.RemoteRetroRouteEnv("http://server")
    env.reset(index=3)
    broken = '{"routes": [{"type": "mol", "smiles": "C/C=C\\\\C", "children": [{"type": "reaction", "children": []}]}'
    for _ in range(client.RemoteRetroRouteEnv.JSON_RETRIES):
        reply = json.loads(env.emit_routes(submission=broken))
        assert (
            reply["graded"] is False and "closed by ']}'" in reply["error"]
        )  # the escaped quote stays inside its string
    assert graded == [] and not env.submitted
    env.emit_routes(submission=broken)  # retries spent: graded as sent
    assert graded == [broken] and env.submitted

    env.reset(index=3)
    env.emit_routes(submission='{"routes": []}')  # valid JSON in a string reaches the grader parsed
    assert graded[-1] == {"routes": []}


def test_a_failed_trace_write_never_reaches_the_trainer(monkeypatch, tmp_path):
    from retroenv_openenv import client

    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    monkeypatch.setenv("RETROENV_TRACE_PATH", str(tmp_path))  # a directory: opening it to append raises OSError
    env = client.RemoteRetroRouteEnv("http://server")
    env.reset(index=3)
    assert env.get_reward() == 0.0  # the episode is still scored


def test_tool_results_count_down_the_budget_and_the_trace_records_each_episode(monkeypatch, tmp_path):
    from retroenv_openenv import client

    trace = tmp_path / "episodes.jsonl"
    monkeypatch.setenv("RETROENV_TRACE_PATH", str(trace))
    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    env = client.RemoteRetroRouteEnv("http://server")
    env.reset(index=0)
    first = json.loads(env.stock_retrieve("CCO"))
    assert first["tool_calls_remaining"] == 2 and "reminder" in first  # 2 left of 3, under REMIND_AT
    assert env.get_reward() == 0.0  # no submission: 0, as in evaluation
    env.reset(index=1)
    env.inspect_molecule("CCO")
    json.loads(env.emit_routes({"routes": []}))
    assert env.get_reward() == 0.6
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [(r["index"], r["submitted"], r["passed"], r["tool_calls"], r["reward"]) for r in records] == [
        (0, False, False, 1, 0.0),
        (1, True, True, 1, 0.6),
    ]
    assert env.submitted and env.passed and not env.failed
    assert all(r["failure"] is None for r in records)  # unsubmitted and solved episodes carry no grader failure


def test_larger_flavors_split_the_gpus_and_watch_runs_on_cpu(monkeypatch, capsys):
    plan = _plan(
        monkeypatch, capsys, "train", "--mode", "async", "--name", "a", "--bucket", "o/b", "--flavor", "h200x4",
        "--vllm-gpus", "2",
    )  # fmt: skip
    assert plan["flavor"] == "h200x4" and plan["command"][-2:] == ["--vllm-gpus", "2"]
    assert _plan(monkeypatch, capsys, "eval", "--name", "e", "--bucket", "o/b")["flavor"] == "h200"
    watch = _plan(monkeypatch, capsys, "watch", "--run", "a", "--job", "j1", "--bucket", "o/b")
    assert watch["flavor"] == "cpu-basic" and watch["name"] == "a-evals" and watch["timeout"] == "30h"
    assert "eval_watch.py --run a --job j1 --bucket o/b" in watch["command"][-1]
    with pytest.raises(SystemExit):
        _plan(monkeypatch, capsys, "watch", "--run", "a", "--bucket", "o/b")  # no training job to follow


def test_a_difficulty_sweep_samples_several_attempts_of_project_task_ids(monkeypatch, capsys, tmp_path):
    ids = "train/sweeps/train-sweep250-0.txt"
    plan = _plan(
        monkeypatch, capsys, "eval", "--name", "s0", "--bucket", "o/b", "--set", "train", "--task-ids", ids,
        "--attempts", "8", "--temperature", "0.8",
    )  # fmt: skip
    command = plan["command"]
    assert command[command.index("--task-ids") + 1] == ids and command[command.index("--attempts") + 1] == "8"
    assert command[command.index("--temperature") + 1] == "0.8" and command[command.index("--set") + 1] == "train"
    assert ids in {path.relative_to(ROOT).as_posix() for path in _launcher().source_files()}
    plain = _plan(monkeypatch, capsys, "eval", "--name", "e", "--bucket", "o/b")["command"]
    assert "--attempts" not in plain and "--temperature" not in plain  # the board's protocol stays the default
    assert "--thinking" not in plain
    thinking = _plan(monkeypatch, capsys, "eval", "--name", "t", "--bucket", "o/b", "--thinking")["command"]
    assert thinking[-1] == "--thinking"
    with pytest.raises(SystemExit):  # the job only sees project files
        _plan(monkeypatch, capsys, "eval", "--name", "e", "--bucket", "o/b", "--task-ids", str(tmp_path / "x.txt"))


def test_the_fsdp2_config_reaches_the_job():
    files = {path.relative_to(ROOT).as_posix() for path in _launcher().source_files()}
    assert "train/jobs/fsdp2.yaml" in files and "train/jobs/eval_watch.py" in files


class _FakeLLM:
    """An OpenAI-compatible chat endpoint that plays back scripted turns and records each request."""

    def __init__(self, turns):
        from types import SimpleNamespace as NS

        self.requests, self._turns, self._ns = [], list(turns), NS
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        NS = self._ns
        self.requests.append(kwargs)
        calls = [
            NS(
                id=f"c{i}",
                function=NS(name=name, arguments=json.dumps(args)),
                model_dump=lambda exclude_none=True, n=name, a=args, i=i: {
                    "id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}
                },
            )
            for i, (name, args) in enumerate(self._turns.pop(0))
        ]  # fmt: skip
        message = NS(content="", tool_calls=calls or None)
        return NS(
            choices=[NS(message=message)], usage=NS(prompt_tokens=10, completion_tokens=5, model_extra={}), model="m"
        )


def test_the_training_harness_eval_runs_the_loop_training_runs(monkeypatch):
    from retroenv_openenv import agent_training
    from retroenv_openenv.agent import AgentConfig

    class Server(_LiveClient):  # grades a submission 0.9 and valid
        def call(self, name, arguments):
            from retroenv_openenv.client import ToolOutcome

            if name == "emit_routes":
                return ToolOutcome(
                    name=name, result={"score": {"valid": True, "hard_failures": []}}, done=True, reward=0.9, error=None
                )
            return ToolOutcome(name=name, result={"matches": []}, done=False, reward=None, error=None)

    opening = {"prompt": "Plan a retrosynthesis for CCO.", "task_id": "retro_x", "max_tool_calls": 32}
    config = AgentConfig(model="m", max_turns=agent_training.MAX_TURNS, tool_choice="auto")
    broken = '{"routes": [{"type": "mol"'
    llm = _FakeLLM([
        [("stock_retrieve", {"query": "CCO"})],
        [("emit_routes", {"submission": broken})],  # handed back ungraded, as in training
        [("emit_routes", {"submission": {"routes": []}})],
        [("stock_retrieve", {"query": "never sent: the episode is graded"})],
    ])  # fmt: skip
    result = agent_training.run_episode(llm, Server.__new__(Server), opening, config)
    assert llm.requests[0]["tools"] is agent_training.TOOLS and llm.requests[0]["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in agent_training.TOOLS} == {
        "emit_routes", "inspect_molecule", "pubchem_lookup", "reaction_class_lookup", "reaction_conditions_search",
        "reaction_precedent_search", "search_literature", "stock_retrieve", "validate_disconnection",
    }  # fmt: skip
    tool_results = [json.loads(m["content"]) for m in result["transcript"] if m["role"] == "tool"]
    assert tool_results[0]["tool_calls_remaining"] == 31  # the training environment's countdown
    assert tool_results[1]["graded"] is False  # the malformed submission came back to fix
    assert result["valid"] and result["reward"] == 0.9 and result["turns"] == 3 and result["json_retries"] == 1

    quiet = _FakeLLM([[("stock_retrieve", {"query": "CCO"})], []])  # a turn without a tool call ends it
    result = agent_training.run_episode(quiet, Server.__new__(Server), opening, config)
    assert result["turns"] == 2 and not result["valid"] and result["reward"] == 0.0
    assert result["hard_failures"] == ["no submission"] and not result["auto_emitted"]


def test_a_thinking_eval_sends_thinking_back_and_fits_the_context_like_training():
    from types import SimpleNamespace as NS

    from retroenv_openenv import agent_training
    from retroenv_openenv.agent import AgentConfig

    class Server(_LiveClient):
        def call(self, name, arguments):
            from retroenv_openenv.client import ToolOutcome

            return ToolOutcome(name=name, result={"matches": []}, done=False, reward=None, error=None)

    class Thinker:  # thinks every turn; refuses a turn whose max_tokens passes a 1,000-token context
        def __init__(self):
            self.requests, self.chat = [], NS(completions=NS(create=self._create))

        def _create(self, **kwargs):
            self.requests.append(kwargs)
            if kwargs["max_tokens"] > 1000 - 900:
                raise RuntimeError(
                    "Error code: 400 - This model's maximum context length is 1000 tokens. However, you requested "
                    f"{kwargs['max_tokens']} output tokens and your prompt contains at least 900 input tokens"
                )
            turn = len([r for r in self.requests if r["max_tokens"] <= 100])
            calls = [] if turn > 1 else [
                NS(id="c0", function=NS(name="stock_retrieve", arguments='{"query": "CCO"}'),
                   model_dump=lambda exclude_none=True: {"id": "c0", "type": "function",
                                                         "function": {"name": "stock_retrieve", "arguments": "{}"}})
            ]  # fmt: skip
            message = NS(content="", tool_calls=calls or None, reasoning=f"thought {turn}")
            return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=900, completion_tokens=5, model_extra={}))

    opening = {"prompt": "Plan a retrosynthesis for CCO.", "task_id": "retro_x", "max_tool_calls": 32}
    llm = Thinker()
    config = AgentConfig(model="m", max_turns=agent_training.MAX_TURNS, tool_choice="auto", max_tokens=32768)
    result = agent_training.run_episode(llm, Server.__new__(Server), opening, replace(config, send_reasoning=True))
    assert result["turns"] == 2 and not result["errors"]  # each turn got the 100 tokens left, not an error
    sent = llm.requests[-1]["messages"][2]
    assert sent["role"] == "assistant" and sent["reasoning"] == sent["reasoning_content"] == "thought 1"
    assert "reasoning_content" not in result["transcript"][0] and result["transcript"][0]["reasoning"] == "thought 1"

    llm = Thinker()  # without send_reasoning (the board, and non-thinking runs) thinking is never sent
    agent_training.run_episode(llm, Server.__new__(Server), opening, config)
    assert "reasoning" not in llm.requests[-1]["messages"][2]


def test_a_thinking_run_is_evaluated_with_thinking_at_its_effort(monkeypatch, capsys):
    plan = _plan(
        monkeypatch, capsys, "eval", "--name", "t", "--bucket", "o/b", "--harness", "training", "--thinking",
        "--reasoning-effort", "low",
    )  # fmt: skip
    assert plan["command"][-3:] == ["--thinking", "--reasoning-effort", "low"]
    watch = _plan(
        monkeypatch, capsys, "watch", "--run", "a", "--job", "j1", "--bucket", "o/b", "--harness", "training",
        "--thinking", "--reasoning-effort", "low", "--eval-concurrency", "32", "--base", "b0,b1",
    )["command"][-1]  # fmt: skip
    assert "--thinking --reasoning-effort low --concurrency 32" in watch and "--base b0,b1" in watch
    plain = _plan(monkeypatch, capsys, "watch", "--run", "a", "--job", "j1", "--bucket", "o/b")["command"][-1]
    assert "--thinking" not in plain and "--concurrency" not in plain
