"""Start the same server locally for notebooks, smoke runs, and jobs."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests


@contextmanager
def local_server(corpus, sessions=16):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # A manifest file selects the indexed corpus; a directory is a prepared snapshot.
    resolved = Path(corpus).resolve()
    key = "FLEURS_CORPUS_MANIFEST" if resolved.is_file() else "ASR_SNAPSHOT"
    env = {**os.environ, key: str(resolved), "ASR_MAX_SESSIONS": str(sessions)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "multilingual_asr.server.app:create_server",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ws",
            "websockets",
            "--log-level",
            "warning",
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
