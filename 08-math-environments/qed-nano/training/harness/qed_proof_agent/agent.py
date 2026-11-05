"""The qed-proof harness as a Harbor agent.

It runs on the host, inside the process that runs the Harbor trial (OpenEnv's Harbor server during training), and
talks to the model over an OpenAI-compatible chat endpoint. During training that endpoint is OpenEnv's capture
proxy, so every call is recorded as training data; the session id is the API key. The sandbox is only where the
final proof lands: `/workspace/proof.md`, the task's declared artifact, which Harbor hands to the verifier.

Registered with OpenEnv in training/server.py; with plain Harbor, use
`harbor run -a qed_proof_agent.agent:QedProofAgent -m <model> --ak api_base=<url>/v1 --ak api_key_env=<VAR>`
(the key is read from that environment variable, so it never lands in Harbor's job config). `extra_body` (JSON)
adds provider options, e.g. `--ak 'extra_body={"chat_template_kwargs": {"enable_thinking": false}}'`, and
`--ak stream=true` streams replies, which hosted routers need for long generations (they time out otherwise).
Calls are retried on rate limits, 5xx and dropped connections; a call that still fails raises one of Harbor's
model-provider error names, so Harbor classifies the trial as a provider failure rather than an unknown error.
`harbor agent schema qed_proof_agent.agent:QedProofAgent` lists the options.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from . import loop

PROOF_PATH = "/workspace/proof.md"


# Harbor's model-provider failure names (harbor/models/trial/error_policy.py): a trial ending in one of these is
# classified as a provider failure and retried, instead of an unclassified error.
class ApiError(RuntimeError):
    pass


class ApiRateLimitError(ApiError):
    pass


class ApiInternalServerError(ApiError):
    pass


class ApiConnectionClosedError(ApiError):
    pass


class QedProofOptions(AgentOptions):
    api_base: str = Field(description="OpenAI-compatible endpoint ending in /v1 (the capture proxy when training).")
    api_key: str | None = Field(default=None, description="Key for api_base. Prefer api_key_env.")
    api_key_env: str | None = Field(default=None, description="Read the key from this environment variable.")
    rounds: int = Field(default=1, ge=0, description="Check-and-rewrite rounds after the first proof.")
    max_tokens: int = Field(default=8192, ge=1, description="Output tokens per model call.")
    extra_body: str | dict | None = Field(default=None, description="JSON merged into every request.")
    stream: bool = Field(default=False, description="Stream replies (hosted routers time out long ones otherwise).")
    retries: int = Field(default=4, ge=0, description="Retries on rate limits, 5xx and dropped connections.")
    request_timeout_sec: float = Field(default=900, gt=0, description="Timeout per model call.")


class QedProofAgent(BaseAgent):
    options_model = QedProofOptions

    def __init__(self, logs_dir: Path, model_name: str | None = None, *args, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        o = self.options or QedProofOptions.model_validate(kwargs)
        self.api_base = o.api_base.rstrip("/")
        self.api_key = o.api_key or (os.environ.get(o.api_key_env) if o.api_key_env else None) or "unused"
        self.extra_body = json.loads(o.extra_body) if isinstance(o.extra_body, str) else dict(o.extra_body or {})
        self.stream, self.retries, self.rounds = o.stream, o.retries, o.rounds
        self.max_tokens, self.request_timeout_sec = o.max_tokens, o.request_timeout_sec
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    @staticmethod
    def name() -> str:
        return "qed-proof"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None  # nothing to install: the model is called from the host

    async def set_web_tools_disabled(self, disabled: bool, environment: BaseEnvironment) -> None:
        return None  # Harbor >= 0.25 calls this under a restricted network; qed-proof has no web tools to disable

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
                            error = (ApiRateLimitError if status == 429 else ApiInternalServerError
                                     if status and status >= 500 else ApiConnectionClosedError if status is None
                                     else ApiError)
                            raise error(f"model call failed: {exc}") from exc
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
