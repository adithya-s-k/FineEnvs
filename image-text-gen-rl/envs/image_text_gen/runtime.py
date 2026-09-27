"""Start the same server locally for notebooks, calibration and smoke runs."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests


@contextmanager
def local_server(source_dir=None, sessions=64, web=False, fixture_verifier=False):
    """Serve on a free port and yield its URL; the server is stopped on exit.

    `source_dir` holds leffff-format CSVs (default: the pinned Hub revision).
    `fixture_verifier` swaps the provider-backed verifier for the offline fixture one.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "IMAGE_TEXT_GEN_MAX_SESSIONS": str(sessions),
        "ENABLE_WEB_INTERFACE": str(web).lower(),
    }
    env.pop("IMAGE_TEXT_GEN_SOURCE_DIR", None)
    env.pop("IMAGE_TEXT_GEN_VERIFIER", None)
    if source_dir is not None:
        env["IMAGE_TEXT_GEN_SOURCE_DIR"] = str(Path(source_dir).resolve())
    if fixture_verifier:
        env["IMAGE_TEXT_GEN_VERIFIER"] = "fixture"
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "image_text_gen.server.app:create_server",
            "--factory", "--host", "127.0.0.1", "--port", str(port),
            "--ws", "websockets", "--log-level", "warning",
        ],
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            try:
                if requests.get(f"{url}/healthz", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError("Server did not become healthy in 90 seconds")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
