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
# Measured on a real webdev render with Xiaomi's vision rubric (45 vision models on the router, two passes).
# These returned a usable verdict both times and scored near the median (~0.70). Judges disagree a lot
# (0.34-0.87 on the same page) and the rubric runs at temperature 1.0, so compare scores per judge.
VISION_JUDGES = [
    {"id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "note": "fastest (~4s), near the median"},
    {"id": "moonshotai/Kimi-K2.7-Code", "note": "fast (~7s), near the median"},
    {"id": "Qwen/Qwen3.8-27B", "note": "fastest (~4s), slightly strict"},
    {"id": "deepseek-ai/DeepSeek-V4.1-Flash", "note": "fast, slightly strict"},
    {"id": "moonshotai/Kimi-K3", "note": "frontier, slow (~1 min)"},
]


def _p(x):
    """Provider prices arrive as floats with noise (5.999999999999999); keep four significant digits."""
    return None if x is None else float(f"{float(x):.4g}")


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
            "input": _p(price.get("input")), "output": _p(price.get("output")),
            "context": pick.get("context_length"), "speed": round(pick.get("throughput") or 0),
            "latency_ms": round(pick.get("first_token_latency_ms") or 0),
            "providers": [{"name": p["provider"], "input": _p((p.get("pricing") or {}).get("input")),
                           "output": _p((p.get("pricing") or {}).get("output")), "tools": bool(p.get("supports_tools")),
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
