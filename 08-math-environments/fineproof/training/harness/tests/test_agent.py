"""FineProofAgent.run against a stub endpoint: what it uploads and what it records about the submitted call."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import httpx
from harbor.models.agent.context import AgentContext

from fineproof_agent import agent as agent_module
from fineproof_agent.agent import FineProofAgent


class StubEnvironment:
    def __init__(self):
        self.uploads = {}

    async def exec(self, command, *args, **kwargs):
        return None

    async def upload_file(self, source, target):
        self.uploads[target] = Path(source).read_text()


def run_agent(replies, rounds, stream=False):
    """Run the agent with each model call answered by the next (content, finish_reason) in `replies`."""
    answers = iter(replies)

    def handler(request):
        content, finish = next(answers)
        if not stream:
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                             "usage": {"prompt_tokens": 3, "completion_tokens": 5}})
        chunks = [{"choices": [{"delta": {"content": content}, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": finish}]},
                  {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    # Only the agent module's view of httpx changes: its client talks to `handler`.
    stub_httpx = types.SimpleNamespace(
        AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
        HTTPStatusError=httpx.HTTPStatusError, TransportError=httpx.TransportError)
    with tempfile.TemporaryDirectory() as logs, mock.patch.dict(os.environ, {}, clear=True):
        options = {"api_base": "http://proxy/v1", "stream": stream, **({} if rounds is None else {"rounds": rounds})}
        agent = FineProofAgent(Path(logs), model_name="policy", **options)
        environment, context = StubEnvironment(), AgentContext()
        with mock.patch.object(agent_module, "httpx", stub_httpx):
            asyncio.run(agent.run("Prove X.", environment, context))
    return environment.uploads[agent_module.PROOF_PATH].strip(), context.metadata["fineproof"]


class SubmittedCall(unittest.TestCase):
    def test_one_call_by_default(self):
        proof, record = run_agent([("A proof.", "stop")], rounds=None)
        self.assertEqual(proof, "A proof.")
        self.assertEqual((record["calls"], record["rounds"], record["submitted_call"], record["finish_reasons"]),
                         (1, 0, 0, ["stop"]))

    def test_rewrite_without_heading_submits_the_first_call(self):
        # The rewrite ran into the cap before its heading: the complete first proof is graded, and the record says
        # so, so a trainer does not zero it for the rewrite's cut-off.
        proof, record = run_agent([("First proof.", "stop"), ("A critique that was cut", "length")], rounds=1)
        self.assertEqual(proof, "First proof.")
        self.assertEqual((record["submitted_call"], record["finish_reasons"]), (0, ["stop", "length"]))

    def test_rewrite_with_heading_submits_the_rewrite(self):
        for stream in (False, True):
            proof, record = run_agent([("First proof, cut", "length"), ("Gaps.\n## Final proof\nBetter.", "stop")],
                                      rounds=1, stream=stream)
            self.assertEqual(proof, "Better.")
            self.assertEqual((record["submitted_call"], record["finish_reasons"]), (1, ["length", "stop"]))


if __name__ == "__main__":
    unittest.main()
