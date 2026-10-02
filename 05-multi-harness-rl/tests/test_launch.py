"""Exercise service wiring without allocating GPUs or starting real subprocesses."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("action", ["train", "eval"])
@pytest.mark.parametrize("mode", ["whitebox", "opencode", "multi_harness"])
def test_each_mode_gets_the_right_services_and_gpu(tmp_path, monkeypatch, action, mode):
    pytest.importorskip("httpx")
    from jobs import run

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared/ready.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["run.py", action, "--mode", mode, "--output", "outputs"]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(run.subprocess, "run", Mock())
    monkeypatch.setattr(run.signal, "signal", Mock())
    monkeypatch.setattr(run.os, "killpg", Mock())
    monkeypatch.setattr(
        run.httpx,
        "get",
        Mock(
            return_value=SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        ),
    )
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs["env"]))
        is_worker = command[1] == "-m" and (
            command[2].startswith("train.") or command[2] == "eval.evaluate"
        )
        if command[1] == "jobs/inference_proxy.py":
            Path(command[command.index("--output") + 1]).write_text(
                "https://sandbox-inference.test"
            )
        return SimpleNamespace(
            poll=lambda: 0 if is_worker else None,
            returncode=0,
            pid=1234,
            wait=lambda **kw: 0,
        )

    monkeypatch.setattr(run.subprocess, "Popen", spawn)
    run.main()
    engine, worker = calls[0], calls[-1]
    assert engine[1]["CUDA_VISIBLE_DEVICES"] == ("2" if action == "train" else "2,3")
    assert worker[1]["CUDA_VISIBLE_DEVICES"] == "3"
    engine_port = engine[0][engine[0].index("--port") + 1]
    assert (
        worker[0][worker[0].index("--vllm-url") + 1]
        == f"http://127.0.0.1:{engine_port}"
    )
    has_harbor = any("smoldataenv_harbor.server:app" in command for command, _ in calls)
    assert has_harbor == (
        mode != "whitebox" and (mode == "multi_harness" or action == "eval")
    )
    assert any("jobs/inference_proxy.py" in command for command, _ in calls) == (
        action == "train" and mode == "opencode"
    )
    assert worker[0][2] == (f"train.{mode}" if action == "train" else "eval.evaluate")
    if action == "eval":
        assert worker[0][worker[0].index("--mode") + 1] == (
            "whitebox" if mode == "whitebox" else "blackbox"
        )
