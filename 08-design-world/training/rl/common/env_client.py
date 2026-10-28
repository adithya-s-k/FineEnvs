"""A minimal async client for one DesignGym (OpenEnv) episode over the server's WebSocket.

It speaks the same protocol as `openenv.core.mcp_client.MCPToolClient` (`reset`, `step` with `list_tools` /
`call_tool`, `close`) but needs only `websockets`, so it installs into any trainer's Python without pulling
openenv's FastAPI / Gradio / huggingface_hub>=1.29 pins into Miles' or SkyRL's environments.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any

import websockets


@dataclass
class ToolOutcome:
    """One tool call's result as the agent sees it (mirrors `rollout.py`'s ToolOutcome)."""

    text: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (media type, base64)
    is_error: bool = False
    done: bool = False
    reward: float | None = None
    grade: dict | None = None


def _ws_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return base + "/ws"


class EnvSession:
    """One episode: `await reset(**task)`, `await tools()`, `await call(name, args)`, `await close()`.

    Each session holds its own socket, so the server gives it its own environment instance.
    """

    def __init__(self, base_url: str, timeout_s: float = 300.0, connect_retries: int = 5):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.connect_retries = connect_retries
        self._ws = None
        self._lock = asyncio.Lock()
        self.metadata: dict[str, Any] = {}

    async def _connect(self) -> None:
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token and ".hf.space" in self.base_url:
            headers["Authorization"] = f"Bearer {token}"
        kwargs: dict[str, Any] = {"max_size": 200 * 2**20, "open_timeout": 60, "ping_interval": 60,
                                  "ping_timeout": 300}
        params = inspect.signature(websockets.connect).parameters
        if headers:
            kwargs["additional_headers" if "additional_headers" in params else "extra_headers"] = headers
        if "proxy" in params and ("127.0.0.1" in self.base_url or "localhost" in self.base_url):
            kwargs["proxy"] = None
        delay = 2.0
        for attempt in range(self.connect_retries):
            try:
                self._ws = await websockets.connect(_ws_url(self.base_url), **kwargs)
                return
            except Exception:  # noqa: BLE001 - server starting up / at capacity: back off
                if attempt == self.connect_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def _rpc(self, message: dict) -> dict:
        async with self._lock:
            if self._ws is None:
                await self._connect()
            await self._ws.send(json.dumps(message))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        response = json.loads(raw)
        if response.get("type") == "error":
            data = response.get("data", {})
            raise RuntimeError(f"env server error: {data.get('message', data)}")
        return response.get("data", {})

    async def reset(self, **kwargs: Any) -> dict[str, Any]:
        data = await self._rpc({"type": "reset", "data": kwargs})
        obs = data.get("observation", {})
        self.metadata = data.get("metadata") or obs.get("metadata") or {}
        return self.metadata

    async def tools(self) -> list[dict[str, Any]]:
        """MCP tool specs as OpenAI function objects: {"name", "description", "parameters"}."""
        data = await self._rpc({"type": "step", "data": {"type": "list_tools"}})
        out = []
        for t in data.get("observation", {}).get("tools", []):
            out.append({"name": t["name"], "description": t.get("description") or "",
                        "parameters": t.get("input_schema") or t.get("inputSchema")
                        or {"type": "object", "properties": {}}})
        return out

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        data = await self._rpc({"type": "step", "data": {"type": "call_tool", "tool_name": name,
                                                         "arguments": arguments}})
        obs = data.get("observation", {})
        meta = data.get("metadata") or obs.get("metadata") or {}
        done, reward = bool(data.get("done")), data.get("reward")
        if obs.get("error"):
            err = obs["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return ToolOutcome(f"Error: {msg}", [], True, done, reward, meta.get("grade"))
        res = obs.get("result") or {}
        texts, images = [], []
        for c in res.get("content") or []:
            if c.get("type") == "text":
                texts.append(c.get("text", ""))
            elif c.get("type") == "image":
                images.append((c.get("mimeType") or "image/png", c["data"]))
        return ToolOutcome("\n".join(texts) or ("(image)" if images else "(no output)"), images,
                           bool(res.get("is_error")), done, reward, meta.get("grade"))

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "close"}))
        except Exception:  # noqa: BLE001
            pass
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass

    async def __aenter__(self) -> EnvSession:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
