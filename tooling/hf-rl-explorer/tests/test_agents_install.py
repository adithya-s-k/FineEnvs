"""How agents get into a task's sandbox: OpenCode from its release binary, Harbor's npm install only as the fallback."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app import runner


class FakeEnv:
    def __init__(self, ok: bool):
        self.ok, self.calls = ok, []

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None, **_):
        self.calls.append((command, user))
        install = "releases/download" in command
        return SimpleNamespace(return_code=0 if (self.ok or not install) else 1, stdout="", stderr="no route")


def _agent(tmp_path):
    from harbor.agents.installed.opencode import OpenCode

    return OpenCode(logs_dir=tmp_path, model_name="p/m")


def test_opencode_installs_from_the_release_binary(tmp_path):
    runner._patch_opencode_install()
    from harbor.agents.installed.opencode import OpenCode

    env = FakeEnv(ok=True)
    asyncio.run(OpenCode.install(_agent(tmp_path), env))
    assert "releases/download" in env.calls[0][0] and env.calls[0][1] == "root"
    assert not any("npm i -g" in c for c, _ in env.calls)   # no apt, nvm or npm


def test_opencode_falls_back_to_harbors_install(monkeypatch, tmp_path):
    runner._patch_opencode_install()
    from harbor.agents.installed.opencode import OpenCode

    fell_back = []
    monkeypatch.setattr(runner, "log", SimpleNamespace(warning=lambda *a, **k: fell_back.append(a)))
    env = FakeEnv(ok=False)
    try:
        asyncio.run(OpenCode.install(_agent(tmp_path), env))
    except Exception:  # noqa: BLE001 - Harbor's own install against a fake sandbox: only that it was tried matters
        pass
    assert fell_back
    assert len(env.calls) > 1   # Harbor's install ran after the binary failed


def test_a_failed_agent_reads_as_what_went_wrong_not_its_command():
    out = ('RuntimeError: Command failed (exit 1): opencode run -- \'Prepare the 401(k) note for Sarah ...\' | tee x\n'
           'stdout: {"type":"error","error":{"name":"APIError","data":{"message":"Gateway Time-out","statusCode":504,'
           '"metadata":{"url":"https://abc.gradio.live/v1/chat/completions"}}}}\n\nstderr: None')
    msg = runner._friendly_text(out)
    assert msg.startswith("The agent's model calls failed with Gateway Time-out (504) at the gradio.live tunnel")
    assert "401(k)" not in msg and "Sign in again" not in msg   # the prompt's "401" is not an auth failure
    other = runner._friendly_text("RuntimeError: Command failed (exit 2): python run.py --prompt 'the whole task'\nstdout: boom")
    assert "the whole task" not in other and "exit 2" in other and "boom" in other
