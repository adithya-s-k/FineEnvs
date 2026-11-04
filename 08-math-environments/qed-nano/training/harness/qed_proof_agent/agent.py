"""The qed-proof harness as a Harbor agent.

It runs on the host, inside the process that runs the Harbor trial (OpenEnv's Harbor server during training), and
talks to the model over an OpenAI-compatible chat endpoint. During training that endpoint is OpenEnv's capture
proxy, so every call is recorded as training data; the session id is the API key. The sandbox is only where the
final proof lands: `/workspace/proof.md`, which the task's collect hook hands to the verifier.

Registered with OpenEnv in training/server.py; with plain Harbor, use
`harbor run -a qed_proof_agent.agent:QedProofAgent -m <model> --ak api_base=<url>/v1 --ak api_key_env=<VAR>`
(the key is read from that environment variable, so it never lands in Harbor's job config). `extra_body` (JSON)
adds provider options, e.g. `--ak 'extra_body={"chat_template_kwargs": {"enable_thinking": false}}'`, and
`--ak stream=true` streams replies, which hosted routers need for long generations (they time out otherwise).
Calls are retried on rate limits, 5xx and dropped connections.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import loop

PROOF_PATH = "/workspace/proof.md"


class QedProofAgent(BaseAgent):
    def __init__(self, logs_dir: Path, model_name: str | None = None, *args, api_base: str | None = None,
                 api_key: str | None = None, api_key_env: str | None = None, rounds: int | str = 1,
                 max_tokens: int | str = 8192, extra_body: dict | str | None = None,
                 stream: bool | str = False, retries: int | str = 4, request_timeout_sec: float = 900, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        if not api_base:
            raise ValueError("qed-proof needs api_base (an OpenAI-compatible /v1 endpoint)")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "unused"
        self.extra_body = json.loads(extra_body) if isinstance(extra_body, str) else dict(extra_body or {})
        self.stream = str(stream).lower() in ("1", "true", "yes")
        self.retries = int(retries)
        self.rounds = int(rounds)
        self.max_tokens = int(max_tokens)
        self.request_timeout_sec = float(request_timeout_sec)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    @staticmethod
    def name() -> str:
        return "qed-proof"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None  # nothing to install: the model is called from the host

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        async with httpx.AsyncClient(timeout=self.request_timeout_sec) as client:
            async def call(messages: list[dict]) -> tuple[str, dict]:
                # Only what the capture proxy accepts while training: no sampling overrides, n=1, plain text.
                payload = {**self.extra_body, "model": self.model_name or "policy", "messages": messages,
                           "max_tokens": self.max_tokens}
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.api_base}/chat/completions"
                if not self.stream:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    return body["choices"][0]["message"].get("content") or "", body.get("usage") or {}
                payload.update(stream=True, stream_options={"include_usage": True})
                parts, usage = [], {}
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:") or line.strip() == "data: [DONE]":
                            continue
                        chunk = json.loads(line[5:])
                        usage = chunk.get("usage") or usage
                        for choice in chunk.get("choices") or []:
                            parts.append((choice.get("delta") or {}).get("content") or "")
                return "".join(parts), usage

            async def complete(messages: list[dict]) -> str:
                for attempt in range(self.retries + 1):
                    try:
                        text, usage = await call(messages)
                        break
                    except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        retryable = status is None or status == 429 or status >= 500
                        if not retryable or attempt == self.retries:
                            raise
                        await asyncio.sleep(10 * 2 ** attempt)
                self.usage["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.usage["completion_tokens"] += usage.get("completion_tokens") or 0
                self.usage["calls"] += 1
                context.n_input_tokens = self.usage["prompt_tokens"]
                context.n_output_tokens = self.usage["completion_tokens"]
                return text.strip()

            proof, messages = await loop.run(complete, instruction, rounds=self.rounds)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "conversation.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False))
        (self.logs_dir / "proof.md").write_text(proof + "\n")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
            handle.write(proof + "\n")
        try:
            await environment.exec("mkdir -p /workspace")
            await environment.upload_file(handle.name, PROOF_PATH)
        finally:
            Path(handle.name).unlink(missing_ok=True)
        context.metadata = {**(context.metadata or {}), "qed_proof": {**self.usage, "rounds": self.rounds,
                                                                      "proof_chars": len(proof)}}
