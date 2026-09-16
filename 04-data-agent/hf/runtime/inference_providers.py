"""HF OAuth model selection and temporary credentials for interactive agent demos.

The experiment's vLLM path does not import this module. Provider availability comes
from HF's live catalog; a visitor credential is never replaced by a Space secret.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import secrets
import threading
import time

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROUTER = "https://router.huggingface.co/v1"


class Catalog:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self._rows = {}
        self._updated = 0
        self._lock = threading.Lock()

    def rows(self):
        with self._lock:
            if time.monotonic() - self._updated < self.ttl and self._rows:
                return dict(self._rows)
            response = httpx.get(ROUTER + "/models", timeout=20)
            response.raise_for_status()
            rows = {}
            for model in response.json()["data"]:
                for provider in model.get("providers", []):
                    if provider.get("status") == "live" and provider.get("supports_tools") is True:
                        rows[(provider["provider"], model["id"])] = provider
            if not rows:
                raise ValueError("No live providers with tool calling were returned")
            self._rows, self._updated = rows, time.monotonic()
            return dict(rows)

    def select(self, provider, model):
        if (provider, model) not in self.rows():
            raise ValueError("Choose a live provider/model pair from the catalog")
        return model + ":" + provider


catalog = Catalog()


def visitor_token(oauth):
    if oauth is None or not oauth.token or oauth.expires_at <= time.time():
        raise ValueError("Sign in with Hugging Face to use Inference Providers")
    if "inference-api" not in oauth.scope.split():
        raise ValueError("Sign in again and allow Inference Providers access")
    return oauth.token


@dataclass
class Lease:
    model: str
    token: str = field(repr=False)
    expires: float
    remaining: int = 32  # Includes native capture capability probes.


class VisitorCredentials:
    """Keep OAuth credentials out of Harbor's long-lived upstream client cache."""
    def __init__(self):
        self._leases = {}
        self._lock = threading.Lock()

    @contextmanager
    def issue(self, oauth, model):
        token = visitor_token(oauth)
        key = secrets.token_urlsafe(32)
        with self._lock:
            self._leases = {k: v for k, v in self._leases.items() if v.expires > time.time()}
            self._leases[key] = Lease(model, token, min(oauth.expires_at, time.time() + 660))
        try:
            yield key
        finally:
            with self._lock:
                self._leases.pop(key, None)

    def get(self, key, consume=False):
        with self._lock:
            lease = self._leases.get(key)
            if lease is None or lease.expires <= time.time():
                self._leases.pop(key, None)
                raise HTTPException(401, "Interactive inference session expired")
            if consume:
                if lease.remaining <= 0:
                    raise HTTPException(429, "Interactive model-call limit reached")
                lease.remaining -= 1
            return lease


credentials = VisitorCredentials()


def mount_provider_relay(app):
    """Only an opaque, short-lived key reaches Harbor; the HF token stays here."""
    def lease_for(request, consume=False):
        value = request.headers.get("authorization", "")
        key = value[7:] if value.lower().startswith("bearer ") else ""
        return credentials.get(key, consume)

    @app.get("/hf-inference/v1/models")
    async def models(request: Request):
        lease = lease_for(request)
        return {"object": "list", "data": [{"id": lease.model, "object": "model"}]}

    @app.post("/hf-inference/v1/chat/completions")
    async def chat(request: Request):
        lease = lease_for(request)
        body = await request.json()
        if body.get("model") != lease.model:
            raise HTTPException(400, "This inference session is bound to the selected model")
        lease_for(request, consume=True)
        body["max_tokens"] = min(int(body.get("max_tokens") or 4096), 4096)
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = min(int(body["max_completion_tokens"]), 4096)
            body.pop("max_tokens")
        client = httpx.AsyncClient(timeout=120)
        try:
            upstream = await client.send(client.build_request("POST", ROUTER + "/chat/completions",
                headers={"Authorization": "Bearer " + lease.token}, json=body), stream=True)
        except httpx.HTTPError:
            await client.aclose()
            raise HTTPException(502, "The selected inference provider could not be reached") from None
        if upstream.status_code >= 400:
            status = upstream.status_code
            await upstream.aclose()
            await client.aclose()
            message = {401: "Sign in again to renew your inference access",
                       402: "Your HF account needs inference credits",
                       403: "Your account cannot access this provider or model",
                       429: "This provider is rate limited; try again later"}.get(status,
                       "The provider rejected the model request; try another model")
            return JSONResponse({"error": {"message": message}}, status_code=status)

        async def chunks():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
        return StreamingResponse(chunks(), media_type=upstream.headers.get("content-type", "application/json"))

    return app
