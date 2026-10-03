"""Start the same server locally for notebooks and jobs."""

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests


@contextmanager
def local_server(
    snapshot,
    sessions=16,
    web=False,
    *,
    source_root=None,
    cache_dir=None,
    local_source=False,
):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "NAYANA_MAX_SESSIONS": str(sessions),
        "ENABLE_WEB_INTERFACE": str(web).lower(),
    }
    env.pop("NAYANA_CORPUS_MANIFEST", None)
    env.pop("NAYANA_SNAPSHOT", None)
    for key in ("NAYANA_SOURCE_ROOT", "NAYANA_LOCAL_SOURCE"):
        env.pop(key, None)
    if str(snapshot).startswith("hf://"):
        env["NAYANA_CORPUS_MANIFEST"] = str(snapshot)
    else:
        path = Path(snapshot).resolve()
        manifest = path / "manifest.json" if path.is_dir() else path
        storage = json.loads(manifest.read_text()).get("storage")
        env[
            "NAYANA_CORPUS_MANIFEST"
            if storage == "bucket-parquet"
            else "NAYANA_SNAPSHOT"
        ] = str(path)
    if source_root is not None:
        env["NAYANA_SOURCE_ROOT"] = str(Path(source_root).resolve())
    if cache_dir is not None:
        env["NAYANA_CACHE_DIR"] = str(Path(cache_dir).resolve())
    if local_source:
        env["NAYANA_LOCAL_SOURCE"] = "true"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "nayana_ocr.server.app:create_server",
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
