"""The env server as the image runs it: `python -m port_week_agentenv.server` on a free port, or, with
PORTSIM_LIVE_IMAGE set to an image built from agentenv/Dockerfile, a container of that image."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

IMAGE = os.environ.get("PORTSIM_LIVE_IMAGE")
CONTAINER_TASKS = "/fixture-tasks"
HOST_GATEWAY = ["--add-host=host.docker.internal:host-gateway"] if sys.platform == "linux" else []
"""How a container reaches the host's fake gateway: Docker Desktop and Colima name the host themselves."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_card(base: str, alive) -> None:
    deadline = time.monotonic() + 60
    while True:
        try:
            httpx.get(f"{base}/.well-known/agent-env.json").raise_for_status()
            return
        except httpx.HTTPError:
            assert alive() and time.monotonic() < deadline, "the env server did not start"
            time.sleep(0.1)


@contextmanager
def served(environment_name: str | None, tasks_dir: Path | None = None):
    """`python -m port_week_agentenv.server` as the image runs it, on a free port, or a container of
    PORTSIM_LIVE_IMAGE; ``tasks_dir`` is its PORT_WEEK_TASKS_DIR (its packs are mounted into the container)."""
    named = {"ENVIRONMENT_NAME": environment_name} if environment_name else {}
    if IMAGE:
        mounts = [arg for p in sorted(tasks_dir.iterdir()) for arg in
                  ("-v", f"{p.resolve()}:{CONTAINER_TASKS}/{p.name}:ro")] if tasks_dir else []
        tasks_env = ["-e", f"PORT_WEEK_TASKS_DIR={CONTAINER_TASKS}"] if tasks_dir else []
        name_env = [arg for k, v in named.items() for arg in ("-e", f"{k}={v}")]
        container = subprocess.run(
            ["docker", "run", "-d", "-p", "127.0.0.1::18765", *HOST_GATEWAY, *name_env, *tasks_env, *mounts, IMAGE],
            check=True, capture_output=True, text=True).stdout.strip()
        try:
            port = subprocess.run(["docker", "port", container, "18765/tcp"], check=True, capture_output=True,
                                  text=True).stdout.split()[0].rsplit(":", 1)[1]
            base = f"http://127.0.0.1:{port}"

            def alive() -> bool:
                state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                                       capture_output=True, text=True)
                return state.stdout.strip() == "true"

            try:
                wait_for_card(base, alive)
            except AssertionError:
                logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
                raise AssertionError(f"the env server did not start:\n{logs.stdout}{logs.stderr}") from None
            yield base
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        return
    port = free_port()
    environ = {k: v for k, v in os.environ.items() if k != "ENVIRONMENT_NAME"}
    environ |= {"MCP_HOST": "127.0.0.1", "MCP_PORT": str(port)} | named | (
        {"PORT_WEEK_TASKS_DIR": str(tasks_dir)} if tasks_dir else {})
    proc = subprocess.Popen([sys.executable, "-m", "port_week_agentenv.server"], env=environ,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_card(base, lambda: proc.poll() is None)
        yield base
    finally:
        proc.terminate()
        proc.wait(10)
