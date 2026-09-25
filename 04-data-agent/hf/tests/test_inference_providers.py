"""Visitor credentials, provider selection, and relay boundaries."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import inference_providers as ip


def oauth(token="visitor-a", **kwargs):
    return NS(token=token, scope=kwargs.get("scope", "openid profile inference-api"),
              expires_at=kwargs.get("expires_at", time.time()+3600))


class ProviderTests(unittest.TestCase):
    def test_only_live_tool_models_and_no_arbitrary_selection(self):
        rows = {"data": [{"id": "org/model", "providers": [
            {"provider": "live", "status": "live", "supports_tools": True},
            {"provider": "offline", "status": "error", "supports_tools": True},
            {"provider": "unknown", "status": "live"}]}]}
        catalog = ip.Catalog()
        with patch("inference_providers.httpx.get", return_value=NS(raise_for_status=lambda: None, json=lambda: rows)) as get:
            self.assertEqual(catalog.select("live", "org/model"), "org/model:live")
            self.assertEqual(len(catalog.rows()), 1)
            self.assertEqual(get.call_count, 1)
            with self.assertRaises(ValueError): catalog.select("offline", "org/model")
            with self.assertRaises(ValueError): catalog.select("live", "https://attacker/model")

    def test_requires_current_visitor_inference_permission(self):
        with patch.dict("os.environ", {"HF_TOKEN": "team-secret"}):
            for value in [None, oauth(expires_at=0), oauth(scope="openid profile"), oauth(token="")]:
                with self.assertRaises(ValueError): ip.visitor_token(value)
        self.assertEqual(ip.visitor_token(oauth()), "visitor-a")

    def test_leases_are_isolated_revoked_and_bounded(self):
        registry = ip.VisitorCredentials()
        with registry.issue(oauth(), "a:model") as a, registry.issue(oauth("visitor-b"), "b:model") as b:
            self.assertNotEqual(a, b)
            self.assertEqual(registry.get(a).token, "visitor-a")
            self.assertEqual(registry.get(b).token, "visitor-b")
            self.assertNotIn("visitor-a", repr(registry.get(a)))
            for _ in range(32): registry.get(a, consume=True)
            with self.assertRaises(HTTPException) as error: registry.get(a, consume=True)
            self.assertEqual(error.exception.status_code, 429)
            registry.get(b).expires = 0
            with self.assertRaises(HTTPException): registry.get(b)
        with self.assertRaises(HTTPException): registry.get(a)
        self.assertEqual(registry._leases, {})


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_model_and_visitor_credential_forwarded(self):
        app = ip.mount_provider_relay(FastAPI())
        outgoing = []
        def upstream(request):
            outgoing.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok", "role": "assistant"}}]})
        original = httpx.AsyncClient
        with ip.credentials.issue(oauth(), "org/model:provider") as key:
            headers = {"authorization": "Bearer " + key}
            async with original(transport=httpx.ASGITransport(app), base_url="http://app") as client:
                self.assertEqual((await client.get("/hf-inference/v1/models")).status_code, 401)
                with patch("inference_providers.httpx.AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(upstream), **kw)):
                    bad = await client.post("/hf-inference/v1/chat/completions", headers=headers, json={"model": "other"})
                    self.assertEqual(bad.status_code, 400)
                    self.assertEqual(outgoing, [])
                    response = await client.post("/hf-inference/v1/chat/completions", headers=headers,
                        json={"model": "org/model:provider", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 99999})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(outgoing[0].headers["authorization"], "Bearer visitor-a")
                self.assertEqual(str(outgoing[0].url), ip.ROUTER + "/chat/completions")
                self.assertEqual(json.loads(outgoing[0].content)["max_tokens"], 4096)
                self.assertNotIn("visitor-a", response.text)
        async with original(transport=httpx.ASGITransport(app), base_url="http://app") as client:
            self.assertEqual((await client.get("/hf-inference/v1/models", headers=headers)).status_code, 401)

    async def test_provider_errors_cannot_echo_secrets(self):
        app = ip.mount_provider_relay(FastAPI())
        original = httpx.AsyncClient
        with ip.credentials.issue(oauth(), "org/model:provider") as key:
            async with original(transport=httpx.ASGITransport(app), base_url="http://app") as client:
                with patch("inference_providers.httpx.AsyncClient", side_effect=lambda **kw: original(
                        transport=httpx.MockTransport(lambda r: httpx.Response(402, text="visitor-a raw secret")), **kw)):
                    response = await client.post("/hf-inference/v1/chat/completions", headers={"authorization": "Bearer " + key},
                        json={"model": "org/model:provider", "messages": []})
                self.assertEqual(response.status_code, 402)
                self.assertNotIn("visitor-a", response.text)
                self.assertIn("credits", response.text)


if __name__ == "__main__":
    unittest.main()
