"""Models on HF Inference Providers, what they cost, and which make good judges.

The agent needs tool calling, so only models with a live tool-capable provider are offered. Each
call is pinned to one provider (`model:provider`) so the price shown is the price paid.
"""

from __future__ import annotations

import threading
import time

import httpx

from . import config

_cache: dict = {"at": 0.0, "models": []}
_lock = threading.Lock()

# Frontier agentic models, shown first. Everything else with tool calling is still selectable.
FEATURED = ["moonshotai/Kimi-K3", "zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3.8-2.4T-A95B",
            "thinkingmachines/Inkling", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "MiniMaxAI/MiniMax-M3",
            "Qwen/Qwen3.5-397B-A17B", "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3-Flash"]
DEFAULT_AGENT = "zai-org/GLM-5.3"

# Judges measured on the real General verifier prompt (see README): all returned a parseable verdict
# on every check, and the fast ones agreed with each other. Reasoning-first models were dropped: they
# often spend the judge's 4k-token budget thinking and return no verdict, which masks the reward.
TEXT_JUDGES = [
    {"id": "openai/gpt-oss-120b", "note": "fastest (~1s a check), agrees with the majority"},
    {"id": "zai-org/GLM-5.3-Flash", "note": "fast, agrees with the majority"},
    {"id": "thinkingmachines/Inkling", "note": "fast, agrees with the majority"},
    {"id": "moonshotai/Kimi-K3", "note": "frontier, agrees with the majority"},
    {"id": "deepseek-ai/DeepSeek-V4-Pro", "note": "frontier, more lenient"},
]
VISION_JUDGES = [
    {"id": "moonshotai/Kimi-K3", "note": "frontier, vision"},
    {"id": "thinkingmachines/Inkling", "note": "fast, vision"},
    {"id": "Qwen/Qwen3.8-27B", "note": "fast, vision"},
    {"id": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "note": "vision"},
]


def _fetch() -> list[dict]:
    r = httpx.get(f"{config.ROUTER}/models", timeout=20)
    r.raise_for_status()
    out = []
    for m in r.json()["data"]:
        live = [p for p in m.get("providers", []) if p.get("status") == "live"]
        tools = [p for p in live if p.get("supports_tools")]
        if not live:
            continue
        pick = min(tools or live, key=lambda p: ((p.get("pricing") or {}).get("output") or 1e9))
        price = pick.get("pricing") or {}
        arch = m.get("architecture") or {}
        out.append({
            "id": m["id"],
            "provider": pick["provider"],
            "tools": bool(tools),
            "vision": "image" in (arch.get("input_modalities") or []),
            "input": price.get("input"), "output": price.get("output"),
            "context": pick.get("context_length"), "speed": round(pick.get("throughput") or 0),
            "latency_ms": round(pick.get("first_token_latency_ms") or 0),
            "providers": [{"name": p["provider"], "input": (p.get("pricing") or {}).get("input"),
                           "output": (p.get("pricing") or {}).get("output"), "tools": bool(p.get("supports_tools")),
                           "speed": round(p.get("throughput") or 0)} for p in live],
            "featured": m["id"] in FEATURED,
        })
    order = {m: i for i, m in enumerate(FEATURED)}
    out.sort(key=lambda m: (order.get(m["id"], 999), -(m["output"] or 0)))
    return out


def all_models() -> list[dict]:
    with _lock:
        if time.time() - _cache["at"] > 900 or not _cache["models"]:
            try:
                _cache["models"] = _fetch()
                _cache["at"] = time.time()
            except Exception:
                if not _cache["models"]:
                    raise
        return _cache["models"]


def get(model_id: str) -> dict | None:
    return next((m for m in all_models() if m["id"] == model_id), None)


def price_of(model_id: str, provider: str | None = None) -> tuple[float, float]:
    """$ per 1M (input, output) tokens for this model on this provider."""
    m = get(model_id) or {}
    p = next((x for x in m.get("providers", []) if x["name"] == (provider or m.get("provider"))), None) or m
    return float(p.get("input") or 0), float(p.get("output") or 0)


def cost(model_id: str, provider: str | None, tokens: dict) -> float:
    pin, pout = price_of(model_id, provider)
    inp = tokens.get("input", 0) + tokens.get("cache_read", 0) + tokens.get("cache_write", 0)
    out = tokens.get("output", 0) + tokens.get("reasoning", 0)   # reasoning tokens bill as output
    return (inp * pin + out * pout) / 1e6


def catalog() -> dict:
    ms = all_models()
    ids = {m["id"] for m in ms}
    return {
        "agents": [m for m in ms if m["tools"]],
        "text_judges": [{**j, **(get(j["id"]) or {})} for j in TEXT_JUDGES if j["id"] in ids],
        "vision_judges": [{**j, **(get(j["id"]) or {})} for j in VISION_JUDGES if j["id"] in ids],
        "default_agent": DEFAULT_AGENT,
        "sandbox_price_per_hour": config.FLAVOR_PRICE_PER_HOUR,
    }
