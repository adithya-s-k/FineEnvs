"""Exercise real HTTP streaming and WebSocket boundaries used by private Spaces."""
import json
import asyncio
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

import httpx
from fastapi import FastAPI, Request, WebSocket
from starlette.responses import StreamingResponse
import uvicorn
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from auth_bridge import make_app


def launch(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws_ping_interval=None))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    return server, thread, port


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = FastAPI()

        @upstream.post("/mcp")
        async def mcp(req: Request):
            assert req.headers["authorization"] == "Bearer integration-test-token"
            body = await req.body()
            async def chunks():
                yield b"data: "
                yield body
                yield b"\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream", headers={"x-test": "preserved"})

        @upstream.websocket("/ws")
        async def websocket(ws: WebSocket):
            assert ws.headers["authorization"] == "Bearer integration-test-token"
            await ws.accept()
            for _ in range(2):
                value = await ws.receive()
                if value.get("text") is not None:
                    await ws.send_text(value["text"])
                else:
                    await ws.send_bytes(value["bytes"])
            await ws.close()

        cls.remote, cls.remote_thread, port = launch(upstream)
        cls.bridge, cls.bridge_thread, cls.port = launch(make_app(f"http://127.0.0.1:{port}", "integration-test-token"))

    @classmethod
    def tearDownClass(cls):
        cls.bridge.should_exit = True
        cls.remote.should_exit = True
        cls.bridge_thread.join(5)
        cls.remote_thread.join(5)

    def test_streamed_mcp_body_and_headers_survive(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "params": {"text": "tool output α\n"}}).encode()
        with httpx.Client() as client:
            response = client.post(f"http://127.0.0.1:{self.port}/mcp", content=body,
                                   headers={"Authorization": "Bearer caller-value"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"data: " + body + b"\n\n")
        self.assertEqual(response.headers["x-test"], "preserved")
        self.assertNotIn("integration-test-token", response.text)

    def test_websocket_text_and_binary_survive(self):
        with connect(f"ws://127.0.0.1:{self.port}/ws", ping_interval=None) as ws:
            ws.send('{"completion_token_ids":[1,4,8]}')
            self.assertEqual(ws.recv(), '{"completion_token_ids":[1,4,8]}')
            ws.send(b"\x00\xff\x01")
            self.assertEqual(ws.recv(), b"\x00\xff\x01")


class IdleProxyTest(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_result_survives_idle_proxy_with_keepalive(self):
        from websockets.asyncio.client import connect as async_connect
        from websockets.exceptions import ConnectionClosed
        upstream = FastAPI()
        @upstream.websocket("/delayed")
        async def delayed(ws: WebSocket):
            await ws.accept()
            value = await ws.receive_text()
            await asyncio.sleep(1.2)
            try:
                await ws.send_text(value)
            except Exception:
                pass  # The no-keepalive control intentionally loses its connection.
        remote, remote_thread, port = launch(upstream)
        closed_idle = []
        async def proxy(reader, writer):
            other_reader, other_writer = await asyncio.open_connection("127.0.0.1", port)
            last = [time.monotonic()]
            async def relay(src, dest):
                while chunk := await src.read(65536):
                    last[0] = time.monotonic()
                    dest.write(chunk)
                    await dest.drain()
            async def expire():
                while time.monotonic() - last[0] < 0.4:
                    await asyncio.sleep(0.03)
                closed_idle.append(True)
            tasks = [asyncio.create_task(relay(reader, other_writer)),
                     asyncio.create_task(relay(other_reader, writer)), asyncio.create_task(expire())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                writer.close()
                other_writer.close()
        proxy_server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        proxy_port = proxy_server.sockets[0].getsockname()[1]
        try:
            for interval, expected in [(None, False), (0.1, True)]:
                bridge, thread, bridge_port = launch(make_app(f"http://127.0.0.1:{proxy_port}", "test", ping_interval=interval))
                try:
                    async with async_connect(f"ws://127.0.0.1:{bridge_port}/delayed", ping_interval=None) as ws:
                        await ws.send('{"completion_token_ids":[1,2,3]}')
                        if expected:
                            self.assertEqual(await asyncio.wait_for(ws.recv(), 3), '{"completion_token_ids":[1,2,3]}')
                        else:
                            with self.assertRaises(ConnectionClosed):
                                await asyncio.wait_for(ws.recv(), 3)
                finally:
                    bridge.should_exit = True
                    await asyncio.to_thread(thread.join, 5)
            self.assertEqual(len(closed_idle), 1)
        finally:
            proxy_server.close()
            await proxy_server.wait_closed()
            remote.should_exit = True
            await asyncio.to_thread(remote_thread.join, 5)


if __name__ == "__main__":
    unittest.main()
