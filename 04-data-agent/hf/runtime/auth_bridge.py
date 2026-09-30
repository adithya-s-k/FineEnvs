"""Local HTTP/SSE/WebSocket bridge to an authenticated, fixed HF Space origin.

It keeps HF access credentials outside the native tool payload and sandbox. Bind on loopback only.
"""
import asyncio
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

HOP_HEADERS = {"host", "connection", "transfer-encoding", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "upgrade", "content-length"}


def make_app(origin, token, *, ping_interval=20):
    origin = origin.rstrip("/")
    if not origin.startswith("https://") and not origin.startswith("http://127.0.0.1:"):
        raise ValueError("Bridge requires HTTPS or a loopback test origin")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=60),
                                     limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                                     follow_redirects=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
    async def http(request: Request, path: str):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        headers["authorization"] = "Bearer " + token
        url = origin + "/" + path
        if request.url.query:
            url += "?" + request.url.query
        upstream = await app.state.client.send(app.state.client.build_request(
            request.method, url, headers=headers, content=request.stream()), stream=True)
        headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=headers,
                                 background=BackgroundTask(upstream.aclose))

    @app.websocket("/{path:path}")
    async def websocket(socket: WebSocket, path: str):
        url = origin.replace("https://", "wss://").replace("http://", "ws://") + "/" + path
        if socket.url.query:
            url += "?" + socket.url.query
        try:
            async with connect(url, additional_headers={"Authorization": "Bearer " + token},
                               max_size=None, ping_interval=ping_interval, ping_timeout=60, open_timeout=60) as upstream:
                await socket.accept()

                async def outbound():
                    while True:
                        event = await socket.receive()
                        if event["type"] == "websocket.disconnect":
                            return
                        await upstream.send(event.get("text") if event.get("text") is not None else event["bytes"])

                async def inbound():
                    async for value in upstream:
                        if isinstance(value, str):
                            await socket.send_text(value)
                        else:
                            await socket.send_bytes(value)

                tasks = [asyncio.create_task(outbound()), asyncio.create_task(inbound())]
                try:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except ConnectionClosed:
            # Surface a broken upstream promptly to the native retry policy.
            try:
                await socket.close(code=1011, reason="Environment connection closed")
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            try:
                await socket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(make_app(os.environ["SPACE_URL"], os.environ["HF_TOKEN"]),
                host="127.0.0.1", port=int(os.environ.get("BRIDGE_PORT", "8100")), log_level="warning")
