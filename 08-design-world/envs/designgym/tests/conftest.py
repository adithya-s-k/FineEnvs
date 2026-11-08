"""Shared fixtures.

Unit tests that only need documents run anywhere. Tests that need real
tasks use the pack at DESIGN_CANVAS_PACK (or ~/.cache/designgym/pack) and
are skipped when there is none (`designgym pull`).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

PACK = Path(os.environ.get("DESIGN_CANVAS_PACK", Path.home() / ".cache/designgym/pack"))


def have_pack() -> bool:
    return (PACK / "manifest.json").exists()


needs_pack = pytest.mark.skipif(not have_pack(), reason="task pack not built")


@pytest.fixture(scope="session")
def pack():
    if not have_pack():
        pytest.skip("task pack not built")
    from designgym.core.pack import load_pack

    return load_pack(PACK)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def server(pack):
    """A real uvicorn server (HTTP + WebSocket + editor) on a free port."""
    import uvicorn

    port = _free_port()
    os.environ["DESIGN_CANVAS_PACK"] = str(PACK)
    os.environ["PORT"] = str(port)
    os.environ["DESIGN_CANVAS_DEBUG"] = "1"  # the scripted GUI agent locates controls
    from designgym.server.app import create_server

    app = create_server()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    import urllib.request

    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    """Headless Chromium by default; E2E_BROWSER=firefox or webkit runs the same checks there."""
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        b = getattr(p, os.environ.get("E2E_BROWSER", "chromium")).launch()
        yield b
        b.close()
