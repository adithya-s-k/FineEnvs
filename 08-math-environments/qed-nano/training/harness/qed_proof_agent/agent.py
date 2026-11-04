"""The qed-proof harness as a Harbor agent.

It runs on the host, inside the process that runs the Harbor trial (OpenEnv's Harbor server during training), and
talks to the model over an OpenAI-compatible chat endpoint. During training that endpoint is OpenEnv's capture
proxy, so every call is recorded as training data; the session id is the API key. The sandbox is only where the
final proof lands: `/workspace/proof.md`, which the task's collect hook hands to the verifier.

Registered with OpenEnv in training/server.py; with plain Harbor, use
`harbor run -a qed_proof_agent.agent:QedProofAgent -m <model> --ak api_base=<url>/v1 --ak api_key=<key>`.
"""

from __future__ import annotations

import json
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
                 api_key: str | None = None, rounds: int | str = 1, max_tokens: int | str = 8192,
                 request_timeout_sec: float = 900, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        if not api_base:
            raise ValueError("qed-proof needs api_base (an OpenAI-compatible /v1 endpoint)")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "unused"
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
            async def complete(messages: list[dict]) -> str:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    # Only what the capture proxy accepts while training: no sampling overrides, n=1, plain text.
                    json={"model": self.model_name or "policy", "messages": messages, "max_tokens": self.max_tokens},
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                self.usage["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.usage["completion_tokens"] += usage.get("completion_tokens") or 0
                self.usage["calls"] += 1
                context.n_input_tokens = self.usage["prompt_tokens"]
                context.n_output_tokens = self.usage["completion_tokens"]
                return (body["choices"][0]["message"].get("content") or "").strip()

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
