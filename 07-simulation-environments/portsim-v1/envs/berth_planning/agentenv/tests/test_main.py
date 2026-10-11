"""`python -m berth_agentenv.server`, as the image runs it, serves portsim whether or not agent-env names it."""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.mark.parametrize("environment_name", [None, "portsim"])
def test_main_serves_portsim(environment_name):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environ = {k: v for k, v in os.environ.items() if k != "ENVIRONMENT_NAME"}
    environ |= {"MCP_HOST": "127.0.0.1", "MCP_PORT": str(port)} | (
        {"ENVIRONMENT_NAME": environment_name} if environment_name else {})
    proc = subprocess.Popen([sys.executable, "-m", "berth_agentenv.server"], env=environ,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                card = httpx.get(f"http://127.0.0.1:{port}/.well-known/agent-env.json").json()
                break
            except httpx.HTTPError:
                assert proc.poll() is None and time.monotonic() < deadline, "the env server did not start"
                time.sleep(0.1)
    finally:
        proc.terminate()
        proc.wait(10)
    assert (card["name"], {t["name"] for t in card["capabilities"]["tools"]}) == (
        "portsim", {"get_situation", "check_plan", "submit_plan"})
