"""Manage a local server for notebooks, training, and smoke tests."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import requests


@contextmanager
def local_server(*, mode="materialize", max_rows=50, sessions=16, dataset=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = dict(os.environ)
    config.update(
        LATEX_OCR_MODE=mode,
        LATEX_OCR_MAX_ROWS=str(max_rows),
        LATEX_OCR_MAX_SESSIONS=str(sessions),
        ENABLE_WEB_INTERFACE="false",
    )
    if dataset:
        config["LATEX_OCR_DATASET"] = str(dataset)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "latex_ocr_env.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ws",
            "websockets",
            "--log-level",
            "warning",
        ],
        env=config,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Environment exited with code {process.returncode}")
            try:
                response = requests.get(f"{url}/healthz", timeout=2)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Environment did not become healthy within 90 seconds")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
