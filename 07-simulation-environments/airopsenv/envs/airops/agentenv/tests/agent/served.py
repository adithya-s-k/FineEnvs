"""A real airops-day env serving a fixture day on a free port, for the harness to play."""

import socket
import threading
import time
from contextlib import contextmanager

import uvicorn
from paths import FIXTURES
from starlette.responses import JSONResponse

from airops_agentenv.live import AirOpsLiveEnv

SMALL = "b6-jfk-20230626-1400-1700"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join()


@contextmanager
def served_day(monkeypatch, task_id: str = SMALL):
    """A fresh env serving the fixture day, its clock re-armed on a stand-in gateway route of the same server:
    (base url, env)."""
    monkeypatch.setenv("AIROPS_TASKS_DIR", str(FIXTURES))
    env = AirOpsLiveEnv()
    mcp = env.create_app()

    @mcp.custom_route("/clock/set-time", methods=["PUT"])
    async def set_time(request):
        return JSONResponse({"armed": True})

    env.live_load(task_id)
    with serve(mcp.http_app()) as url:
        env.set_time_url = f"{url}/clock/set-time"
        yield url, env
