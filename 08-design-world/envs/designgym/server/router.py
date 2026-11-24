"""One URL in front of several server processes: `designgym serve --workers N`.

uvicorn's own --workers is not safe for this server. OpenEnv's client opens an episode in two requests: an HTTP
`openenv/session/create` on /mcp, then a WebSocket that attaches to that session by id. The built-in editor's
browser page also calls back for its episode. Both have to reach the process that holds the state, and a shared
socket hands each connection to any worker. So each worker here is a complete single-process server on its own
loopback port, and this router keeps every request on the worker that owns it:

    POST /mcp openenv/session/create     -> the least busy worker; the session id it returns is remembered
    /mcp and /ws with a session_id       -> that session's worker
    /ws without one                      -> the least busy worker (the WebSocket owns its session)
    /api/episodes/{id}/...               -> the episode's worker: ids carry it (w2-...), or were seen in its replies
    /web, /gradio_api, ...               -> worker 0 (the playground keeps its sessions in memory)
    everything else                      -> any worker (stateless)

Each worker's GUI driver loads the editor from its own port (DESIGN_CANVAS_SELF_URL), and workers share real-app
desktops through lock files (realapps/desktop/client.py), so no two workers drive the same desktop.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

import httpx
import websockets
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

# hop-by-hop headers stay here. Host is forwarded: the playground (Gradio) builds its URLs from it, and with the
# worker's own address the browser is sent to 127.0.0.1 (an unstyled page on the Space)
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length"}
EPISODE_ID = re.compile(rb'"episode_id"\s*:\s*"([A-Za-z0-9_.-]{1,64})"')
STICKY_PREFIXES = ("/web", "/gradio_api", "/gradio", "/theme.css", "/assets", "/file=", "/upload", "/queue",
                   "/config", "/info", "/heartbeat")


class Router:
    def __init__(self, backends: list[str]):
        self.backends = backends
        self.sessions: dict[str, int] = {}  # OpenEnv session id -> worker
        self.episodes: dict[str, int] = {}  # client-chosen episode id -> worker (seen in replies)
        self.load = [0] * len(backends)     # open WebSockets and in-flight session creations per worker
        self.rr = itertools.cycle(range(len(backends)))
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10), limits=httpx.Limits(
            max_connections=2048, max_keepalive_connections=256))

    # ------------------------------------------------------------------ choosing a worker

    def least_busy(self) -> int:
        start = next(self.rr)  # ties go round-robin
        return min(range(len(self.backends)), key=lambda i: (self.load[i], (i - start) % len(self.backends)))

    def for_episode(self, episode_id: str) -> int:
        m = re.match(r"w(\d+)-", episode_id)
        if m and int(m[1]) < len(self.backends):
            return int(m[1])
        return self.episodes.get(episode_id, 0)

    def learn(self, worker: int, body: bytes) -> None:
        for eid in EPISODE_ID.findall(body[:200_000]):
            eid = eid.decode()
            if not re.match(r"w\d+-", eid):
                self.episodes[eid] = worker
                if len(self.episodes) > 100_000:  # old entries first
                    for k in list(self.episodes)[:20_000]:
                        del self.episodes[k]

    def for_path(self, path: str) -> int | None:
        m = re.match(r"/api/(?:episodes|debug/gui)/([^/]+)", path)
        if m:
            return self.for_episode(m[1])
        if path.startswith(STICKY_PREFIXES):
            return 0
        return None

    # ------------------------------------------------------------------ HTTP

    async def http_proxy(self, request: Request) -> Response:
        body = await request.body()
        worker = self.for_path(request.url.path)
        closing = None
        if worker is None and request.url.path.rstrip("/") == "/mcp" and request.method == "POST":
            try:
                rpc = json.loads(body or b"{}")
            except ValueError:
                rpc = {}
            sid = (rpc.get("params") or {}).get("session_id") if isinstance(rpc, dict) else None
            if sid and sid in self.sessions:
                worker = self.sessions[sid]
                if rpc.get("method") == "openenv/session/close":
                    closing = sid
            elif isinstance(rpc, dict) and rpc.get("method") == "openenv/session/create":
                return await self.create_session(request, body)
        if worker is None:
            worker = next(self.rr)
        url = self.backends[worker] + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        try:
            req = self.http.build_request(request.method, url, headers=headers, content=body)
            resp = await self.http.send(req, stream=True)
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"worker {worker} unreachable: {e}"}, status_code=502)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
        if closing:
            self.sessions.pop(closing, None)
        if request.url.path.startswith("/api/episodes"):
            data = await resp.aread()
            await resp.aclose()
            self.learn(worker, data)
            return Response(data, status_code=resp.status_code, headers=out_headers)

        async def stream():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(stream(), status_code=resp.status_code, headers=out_headers)

    async def create_session(self, request: Request, body: bytes) -> Response:
        """A new OpenEnv session: the least busy worker first, the next one if it is full (each worker caps its
        sessions at MAX_CONCURRENT_ENVS), and the session's worker remembered for /mcp and /ws."""
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        order = sorted(range(len(self.backends)), key=lambda i: (self.load[i], i))
        first = self.least_busy()
        order.remove(first)
        data, status, out_headers = b"", 502, {}
        for worker in [first, *order]:
            self.load[worker] += 1
            try:
                resp = await self.http.post(self.backends[worker] + "/mcp", headers=headers, content=body)
            except httpx.HTTPError as e:
                data, status = json.dumps({"error": f"worker {worker} unreachable: {e}"}).encode(), 502
                continue
            finally:
                self.load[worker] -= 1
            data, status = resp.content, resp.status_code
            out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
            try:
                reply = json.loads(data)
            except ValueError:
                reply = {}
            sid = (reply.get("result") or {}).get("session_id") if isinstance(reply, dict) else None
            if sid:
                self.sessions[sid] = worker
                if len(self.sessions) > 200_000:  # sessions closed without telling us, oldest first
                    for k in list(self.sessions)[:50_000]:
                        del self.sessions[k]
                break
            if "capacity" not in json.dumps(reply.get("error", "")).lower():
                break  # a real error, not a full worker: pass it on
        return Response(data, status_code=status, headers=out_headers)

    # ------------------------------------------------------------------ WebSocket

    async def ws_proxy(self, ws: WebSocket) -> None:
        path = ws.url.path
        sid = ws.query_params.get("session_id")
        worker = self.for_path(path)
        if worker is None:
            worker = self.sessions.get(sid) if sid else None
        if worker is None:
            worker = self.least_busy()
        target = self.backends[worker].replace("http", "ws", 1) + path + (f"?{ws.url.query}" if ws.url.query else "")
        sub = ws.headers.get("sec-websocket-protocol")
        self.load[worker] += 1
        try:
            async with websockets.connect(target, max_size=None, open_timeout=30, ping_interval=None,
                                          subprotocols=[s.strip() for s in sub.split(",")] if sub else None,
                                          compression=None) as up:
                await ws.accept(subprotocol=up.subprotocol)

                async def down_to_up():
                    try:
                        while True:
                            msg = await ws.receive()
                            if msg["type"] == "websocket.disconnect":
                                break
                            await up.send(msg["text"] if msg.get("text") is not None else msg["bytes"])
                    except WebSocketDisconnect:
                        pass
                    finally:
                        await up.close()  # also ends up_to_down when the client goes

                async def up_to_down():
                    try:
                        async for msg in up:
                            if isinstance(msg, str):
                                if '"episode_id"' in msg:
                                    self.learn(worker, msg.encode())
                                await ws.send_text(msg)
                            else:
                                await ws.send_bytes(msg)
                    except websockets.ConnectionClosed:
                        pass
                    finally:
                        try:
                            await ws.close(code=up.close_code or 1000)
                        except (RuntimeError, WebSocketDisconnect):  # the client left first
                            pass

                await asyncio.gather(down_to_up(), up_to_down())
        except (OSError, websockets.InvalidHandshake) as e:
            try:
                await ws.close(code=1011, reason=f"worker {worker}: {e}"[:120])
            except RuntimeError:
                pass
        finally:
            self.load[worker] -= 1

    # ------------------------------------------------------------------ health

    async def health(self, request: Request) -> Response:
        states = []
        for i, b in enumerate(self.backends):
            try:
                r = await self.http.get(b + "/health", timeout=5)
                states.append({"worker": i, "ok": r.status_code == 200, "load": self.load[i]})
            except httpx.HTTPError:
                states.append({"worker": i, "ok": False, "load": self.load[i]})
        ok = all(s["ok"] for s in states)
        return JSONResponse({"status": "healthy" if ok else "degraded", "workers": states,
                             "sessions": len(self.sessions)}, status_code=200 if ok else 503)


def build_app(backends: list[str]) -> Starlette:
    r = Router(backends)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await r.http.aclose()

    return Starlette(routes=[
        Route("/health", r.health),
        WebSocketRoute("/{path:path}", r.ws_proxy),
        Route("/{path:path}", r.http_proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]),
    ], lifespan=lifespan)


# ---------------------------------------------------------------------- launcher


def _wait_healthy(url: str, procs: list, timeout: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(p.poll() is not None for p in procs):
            raise SystemExit("a worker exited during startup")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2):
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit(f"worker {url} did not start")


def serve(host: str, port: int, workers: int) -> None:
    """Start `workers` server processes on loopback ports and the router on host:port."""
    import uvicorn

    if os.environ.get("DESIGN_CANVAS_APPS", "").lower() == "daytona":
        _prepare_daytona()  # once, here: workers then find the sandboxes instead of each creating them
    base = int(os.environ.get("DESIGN_CANVAS_WORKER_BASE_PORT", str(port + 100)))
    procs, urls = [], []
    for k in range(workers):
        p = base + k
        env = {**os.environ, "PORT": str(p), "DESIGN_CANVAS_WORKER_ID": str(k), "DESIGN_CANVAS_WORKERS": "1",
               "DESIGN_CANVAS_SELF_URL": f"http://127.0.0.1:{p}"}
        procs.append(subprocess.Popen([sys.executable, "-m", "designgym.server.app", "--host", "127.0.0.1",
                                       "--port", str(p)], env=env))
        urls.append(f"http://127.0.0.1:{p}")

    def stop(*_):
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pr.kill()

    signal.signal(signal.SIGTERM, lambda *a: (stop(), sys.exit(0)))
    try:
        for u in urls:
            _wait_healthy(u, procs)
        print(f"DesignGym on http://{host}:{port}: {workers} workers on ports {base}-{base + workers - 1}",
              flush=True)
        uvicorn.run(build_app(urls), host=host, port=port, ws_ping_interval=60, ws_ping_timeout=300,
                    log_level="warning")
    finally:
        stop()


def _prepare_daytona() -> None:
    from pathlib import Path

    from ..core.pack import load_pack
    from ..realapps import registry
    from ..realapps.daytona import DaytonaApps

    n = int(os.environ.get("DESIGN_CANVAS_DAYTONA_DESKTOPS", "2"))
    apps = DaytonaApps()
    apps.desktops(n, Path(load_pack().root) / "fonts")
    for app in registry.apps().values():  # a plugin's shared sandboxes (a web app's), started once before any worker
        if app.prepare_daytona is not None:
            app.prepare_daytona(apps)
