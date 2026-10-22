"""A Tinker-backed OpenAI chat endpoint that OpenEnv's capture proxy can train through.

OpenEnv's capture proxy (RFC 006) needs an engine that returns, for every chat completion, the
prompt's token ids (`prompt_token_ids`), the sampled ids (`choices[0].token_ids`) and their
logprobs (`choices[0].logprobs.content`), as vLLM does with `--return-tokens-as-token-ids`. This
server provides exactly that on top of a Tinker `SamplingClient` and a cookbook renderer, so any
black-box harness (e.g. a Harbor task run by OpenCode or Claude Code) can be trained on Tinker.

Swap in fresh weights after each optimiser step with `set_sampling_client()`; in-flight requests
finish on the weights they started with.

Only text prompts are supported: a prompt with images cannot be expressed as `prompt_token_ids`.
Screenshot-driven environments should use the white-box path (`OpenEnvEnvGroupBuilder`).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import tinker
from aiohttp import web
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import ToolCall, get_text_content

from .content import openai_content_to_parts

logger = logging.getLogger(__name__)


class BadRequest(ValueError):
    pass


def _to_cookbook(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system, out = [], []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            content = openai_content_to_parts(m.get("content"))
            system.append(content if isinstance(content, str) else "".join(p.get("text", "") for p in content))
            continue
        content = openai_content_to_parts(m.get("content"))
        if not isinstance(content, str):
            raise BadRequest("image content is not supported: prompts must be expressible as token ids")
        msg: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = [
                ToolCall(
                    id=tc.get("id"),
                    function=ToolCall.FunctionBody(
                        name=tc["function"]["name"],
                        arguments=tc["function"].get("arguments") or "{}",
                    ),
                )
                for tc in m["tool_calls"]
            ]
        if role == "tool":
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                msg["name"] = m["name"]
        out.append(msg)
    return "\n\n".join(s for s in system if s), out


def _tool_specs(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    specs = []
    for t in tools or []:
        fn = t.get("function", t)
        specs.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return specs


class TinkerChatServer:
    """OpenAI-compatible `/v1/chat/completions` over Tinker, with token ids and logprobs.

    Args:
        sampling_client: The Tinker sampler to serve (base model or a training checkpoint).
        renderer: The cookbook renderer for that model (`make_renderer(model_name)`).
        model_name: Name reported in responses and `/v1/models`.
        max_tokens: Default completion budget when a request sets none.
        auth_token: If set, requests must send `Authorization: Bearer <auth_token>`.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: Renderer,
        *,
        model_name: str = "tinker",
        max_tokens: int = 4096,
        auth_token: str | None = None,
    ):
        self._client = sampling_client
        self._renderer = renderer
        self._model = model_name
        self._max_tokens = max_tokens
        self._auth = auth_token
        self._runner: web.AppRunner | None = None
        self.requests = 0

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Serve new weights from the next request on."""
        self._client = sampling_client

    # ------------------------------------------------------------------ app

    def app(self) -> web.Application:
        middlewares = [self._auth_middleware] if self._auth else []
        app = web.Application(middlewares=middlewares, client_max_size=64 * 1024 * 1024)
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/chat/completions", self._chat)
        return app

    async def start(self, host: str = "127.0.0.1", port: int = 8300) -> str:
        """Start serving in the current event loop; returns the base URL (`.../v1`)."""
        self._runner = web.AppRunner(self.app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        return f"http://{host}:{port}/v1"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        if request.path != "/health" and request.headers.get("Authorization") != f"Bearer {self._auth}":
            return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        return await handler(request)

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "model": self._model})

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": self._model, "object": "model", "owned_by": "tinker"}],
            }
        )

    async def _chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            return web.json_response(await self.complete(body))
        except BadRequest as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status=400,
            )
        except Exception as exc:
            logger.exception("chat completion failed")
            return web.json_response({"error": {"message": f"{type(exc).__name__}: {exc}"}}, status=500)

    # ------------------------------------------------------------------ completion

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """One chat completion as an OpenAI response dict (also usable without the HTTP server)."""
        if body.get("stream"):
            raise BadRequest("streaming is not supported; the capture proxy calls engines non-streaming")
        if int(body.get("n") or 1) != 1:
            raise BadRequest("only n=1 is supported")
        system, messages = _to_cookbook(body.get("messages") or [])
        specs = _tool_specs(body.get("tools"))
        if specs:
            messages = (
                self._renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=system) + messages
            )
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        prompt = self._renderer.build_generation_prompt(messages)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or self._max_tokens)
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0),
            top_p=float(body.get("top_p") or 1.0),
            top_k=int(body.get("top_k") if body.get("top_k") is not None else -1),
            stop=self._renderer.get_stop_sequences(),
            seed=body.get("seed"),
        )
        result = await self._client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        seq = result.sequences[0]
        tokens = list(seq.tokens)
        logprobs = list(seq.logprobs or [])
        parsed, _ = self._renderer.parse_response(tokens)
        tool_calls = [
            {
                "id": tc.id or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in parsed.get("tool_calls") or []
        ]
        thinking = ""
        if isinstance(parsed.get("content"), list):
            thinking = "".join(p.get("thinking", "") for p in parsed["content"] if p.get("type") == "thinking")
        finish = "length" if seq.stop_reason == "length" else ("tool_calls" if tool_calls else "stop")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": get_text_content(parsed) or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking:
            message["reasoning_content"] = thinking
        self.requests += 1
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model,
            "prompt_token_ids": prompt.to_ints(),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                    "token_ids": tokens,
                    "logprobs": {
                        "content": [
                            {
                                # vLLM's `--return-tokens-as-token-ids` spelling: tells the
                                # capture proxy these are exact ids, not re-tokenised text.
                                "token": f"token_id:{t}",
                                "logprob": float(lp),
                                "bytes": None,
                                "top_logprobs": [],
                            }
                            for t, lp in zip(tokens, logprobs)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt.length,
                "completion_tokens": len(tokens),
                "total_tokens": prompt.length + len(tokens),
            },
        }
