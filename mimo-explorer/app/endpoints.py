"""Bring your own model: any OpenAI-compatible endpoint (a LiteLLM proxy, vLLM, OpenAI, Together, OpenRouter...).

The sandbox still runs on the user's Hugging Face account, so they sign in with HF either way; only the agent's
model calls go to their endpoint. The API key is held in memory for the rollout and handed to the sandbox
through a root-only file. It is never written to a trace, to the run record or to disk on this server.

This server calls the endpoint too (to test it, list its models, and for Music, which has no sandbox), so the
URL must be https and resolve to public addresses only: otherwise anyone could make the Space probe its own
private network.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urlparse

import httpx

TIMEOUT = 30


class EndpointError(ValueError):
    pass


def check_url(base_url: str) -> str:
    u = urlparse((base_url or "").strip())
    if u.scheme != "https" or not u.hostname:
        raise EndpointError("Use an https URL, like https://api.example.com/v1. The sandbox reaches it over the internet.")
    if u.username or u.password:
        raise EndpointError("Put the key in the API key field, not in the URL.")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise EndpointError(f"Couldn't resolve {u.hostname}.")
    for *_, addr in infos:
        ip = ipaddress.ip_address(addr[0])
        if not ip.is_global:
            raise EndpointError(f"{u.hostname} resolves to a private address. The endpoint has to be reachable from "
                                "the internet, because the agent calls it from an HF Sandbox.")
    return u.geturl().rstrip("/")


def _headers(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _explain(r: httpx.Response) -> str:
    try:
        j = r.json()
        msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error") or j.get("detail") or j.get("message")
    except ValueError:
        msg = r.text[:200]
    hint = {401: "the key was rejected", 403: "the key isn't allowed to use this", 404: "no such path or model",
            429: "rate limited"}.get(r.status_code, "")
    return f"HTTP {r.status_code}{f' ({hint})' if hint else ''}{f': {str(msg)[:240]}' if msg else ''}"


def list_models(base_url: str, key: str | None) -> list[str]:
    base = check_url(base_url)
    r = httpx.get(f"{base}/models", headers=_headers(key), timeout=TIMEOUT)
    if r.status_code != 200:
        raise EndpointError(f"Couldn't list models: {_explain(r)}")
    data = r.json()
    items = data.get("data") if isinstance(data, dict) else data
    return sorted({m.get("id") for m in items or [] if isinstance(m, dict) and m.get("id")})[:500]


PROBE_TOOL = {"type": "function", "function": {"name": "record_answer", "description": "Record the answer.",
              "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}


def test(base_url: str, key: str | None, model: str) -> dict:
    """One small chat completion that asks for a tool call. Agents need tool calling; Music only needs chat."""
    base = check_url(base_url)
    if not (model or "").strip():
        raise EndpointError("Enter the model name your endpoint expects.")
    t = time.time()
    r = httpx.post(f"{base}/chat/completions", headers=_headers(key), timeout=60, json={
        "model": model.strip(), "max_tokens": 400, "tools": [PROBE_TOOL], "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Call record_answer with the answer to 2+2."}]})
    ms = round((time.time() - t) * 1000)
    if r.status_code != 200:
        # some servers reject the tools field outright: tell chat-only apart from broken
        plain = httpx.post(f"{base}/chat/completions", headers=_headers(key), timeout=60, json={
            "model": model.strip(), "max_tokens": 50, "messages": [{"role": "user", "content": "Say OK."}]})
        if plain.status_code == 200:
            return {"ok": True, "tools": False, "ms": ms, "detail": f"Chat works, but tool calls failed: {_explain(r)}"}
        raise EndpointError(f"The endpoint answered {_explain(plain)}")
    try:
        msg = r.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError):
        raise EndpointError("The reply wasn't an OpenAI-style chat completion.")
    tools = bool(msg.get("tool_calls"))
    return {"ok": True, "tools": tools, "ms": ms,
            "detail": "Tool calling works." if tools else "Chat works, but the model answered in text instead of calling the tool."}
