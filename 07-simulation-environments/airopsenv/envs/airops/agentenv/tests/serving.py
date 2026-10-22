"""An env server as its image runs it: `python -m airops_agentenv.server` (or `.occ_server`, `.occ3_server`) on a free
port, or, with AIROPS_IMAGE (AIROPS_OCC_IMAGE, AIROPS_OCC3_IMAGE) set to an image built from agentenv/Dockerfile
(Dockerfile.occ, Dockerfile.occ3), a container of it, which serves the fetched pack it holds."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import httpx

IMAGES = {"airops_agentenv.server": os.environ.get("AIROPS_IMAGE"),
          "airops_agentenv.occ_server": os.environ.get("AIROPS_OCC_IMAGE"),
          "airops_agentenv.occ3_server": os.environ.get("AIROPS_OCC3_IMAGE")}
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
def served(module: str, environ: dict[str, str]):
    """The server of ``module`` with ``environ`` (its pack variables) in process, or a container of its image, which
    keeps the pack it was built with."""
    if image := IMAGES[module]:
        container = subprocess.run(["docker", "run", "-d", "-p", "127.0.0.1::18765", *HOST_GATEWAY, image],
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
    proc = subprocess.Popen([sys.executable, "-m", module],
                            env={**os.environ, **environ, "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_card(base, lambda: proc.poll() is None)
        yield base
    finally:
        proc.terminate()
        proc.wait(10)
