"""A Harbor task's variables, and graders that call a model.

A task's `task.toml` names variables Harbor fills from the host: `[environment.env]` (the agent, the grader and setup
see them), `[verifier.env]` and each step's (the grader only). This explorer never fills them from its own
environment (app/runner.py); this module decides, for one task, what each one gets in a rollout here:

  a key a grader uses to call a model    a per-rollout capability for a relay to HF Inference Providers, with a judge
  (OPENAI_API_KEY, GA_JUDGE_KEY=${HF_TOKEN}, …)   model the visitor picks; never a real token
  that model's base URL                  the relay (OpenAI clients get /v1, Anthropic's don't)
  its model name                         the task's own when it gives one (the relay serves the picked judge either way)
  a key or token the agent would see     empty: the agent gets no model access but its own
  anything else without a default        the task can't run here (named, so the page can say so)

`plan(toml)` reads the task; `bindings(plan, relay, capability, judge)` gives each phase's values.
"""

from __future__ import annotations

import re
import tomllib
from typing import Any

TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
KEY = re.compile(r"(API_?KEY|JUDGE_KEY|_KEY$|^KEY$|TOKEN$|SECRET$|PASSWORD$)", re.I)
URL = re.compile(r"(BASE_URL|API_BASE|API_URL|_URL$|ENDPOINT$|_HOST$)", re.I)
MODEL = re.compile(r"MODEL", re.I)
# names that say "a model provider" (an LLM key or URL), not a database password or a service of the task's own
LLM = re.compile(r"OPENAI|ANTHROPIC|OPENROUTER|JUDGE|LLM|GEMINI|GOOGLE_API|MISTRAL|TOGETHER|GROQ|DEEPSEEK|FIREWORKS|COHERE|XAI|"
                 r"HF_TOKEN|HUGGING|INFERENCE|REWARDKIT|LITELLM", re.I)
VISION = re.compile(r"WEBDEV|VISION|SCREENSHOT|IMAGE", re.I)
# a judge's sampling settings a task asks for without a default: the deterministic choice
PARAMS = {"TEMPERATURE": "0", "TOP_P": "1", "SEED": "0", "MAX_TOKENS": "4096", "MAX_OUTPUT_TOKENS": "4096"}


def _param(key: str) -> str | None:
    up = key.upper()
    return next((v for k, v in PARAMS.items() if up.endswith(k)), None) if LLM.search(key) else None


def _entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every env entry of a task, with the phase it applies to."""
    out = []

    def add(env: Any, section: str, phase: str) -> None:
        for key, value in (env or {}).items() if isinstance(env, dict) else []:
            v = str(value)
            m = TEMPLATE.fullmatch(v)
            out.append({"key": str(key), "section": section, "phase": phase, "value": None if m else v,
                        "var": m.group(1) if m else None, "default": m.group(2) if m else None})

    add((cfg.get("environment") or {}).get("env"), "environment.env", "environment")
    add((cfg.get("verifier") or {}).get("env"), "verifier.env", "verifier")
    add(((cfg.get("verifier") or {}).get("environment") or {}).get("env"), "verifier.environment.env", "verifier")
    add((cfg.get("solution") or {}).get("env"), "solution.env", "solution")
    for st in cfg.get("steps") or []:
        if isinstance(st, dict):
            name = st.get("name") or "step"
            add((st.get("verifier") or {}).get("env"), f"steps.{name}.verifier.env", "verifier")
    return out


def _role(e: dict[str, Any]) -> str:
    k = e["key"]
    if MODEL.search(k):
        return "model"
    if KEY.search(k):
        return "key"
    if URL.search(k):
        return "url"
    return "other"


def plan(toml_text: str) -> dict[str, Any]:
    """What a task's variables need, and whether its grader calls a model: {vars, judge, missing, agent_keys}."""
    try:
        cfg = tomllib.loads(toml_text or "")
    except (tomllib.TOMLDecodeError, TypeError):
        return {"vars": [], "judge": None, "missing": [], "agent_keys": [], "invalid": True}
    entries = [{**e, "role": _role(e)} for e in _entries(cfg)]
    graded = [e for e in entries if e["phase"] == "verifier"]
    llm = [e for e in graded if e["role"] in ("key", "url") and (LLM.search(e["key"]) or LLM.search(e["var"] or ""))
           and (e["var"] or e["role"] == "url")]
    judge = None
    if any(e["role"] == "key" for e in llm) or any(e["role"] == "url" for e in llm):
        models = [e for e in graded if e["role"] == "model"]
        requested = next((e["value"] or e["default"] for e in models if e["value"] or e["default"]), None)
        names = " ".join(e["key"] for e in graded)
        judge = {"requested": requested, "vision": bool(VISION.search(names)),
                 "anthropic": any("ANTHROPIC" in e["key"].upper() for e in llm), "keys": sorted({e["key"] for e in llm})}
    missing = []
    agent_keys = []
    for e in entries:
        if e["var"] is None or e["default"] is not None:
            continue
        if e["phase"] == "verifier" and judge and (e["role"] in ("key", "url", "model") or _param(e["key"])):
            continue
        if e["role"] == "key":   # a key the agent would see, or a grader's key to something that isn't a model: left empty
            agent_keys.append(e["key"])
            continue
        missing.append(e["var"])
    return {"vars": entries, "judge": judge, "missing": sorted(set(missing)), "agent_keys": sorted(set(agent_keys))}


def bindings(p: dict[str, Any], relay: str | None = None, capability: str | None = None, judge: str | None = None) -> dict[str, dict[str, str]]:
    """Each phase's values, by variable (the `${VAR}`'s name) and by key, for app.runner's resolver:
    {"environment": {...}, "verifier": {...}}. `relay` is the capture proxy's base URL, `capability` the judge
    session's id, `judge` the model it serves."""
    out: dict[str, dict[str, str]] = {"environment": {}, "verifier": {}, "solution": {}}
    for e in p["vars"]:
        if e["var"] is None:
            continue
        phase = out.setdefault(e["phase"], {})
        val: str | None = None
        if e["phase"] == "verifier" and p.get("judge") and capability and e["role"] in ("key", "url", "model"):
            if e["role"] == "key":
                val = capability
            elif e["role"] == "url":
                val = relay if "ANTHROPIC" in e["key"].upper() else f"{relay}/v1"
            else:
                val = e["default"] if e["default"] else judge
        elif e["phase"] == "verifier" and p.get("judge") and e["default"] is None and _param(e["key"]):
            val = _param(e["key"])
        elif e["default"] is not None:
            continue   # the task's own default
        elif e["role"] == "key":
            val = ""   # no key of ours for it: empty, never this server's
        if val is not None:
            phase[e["key"]] = val
            phase.setdefault(e["var"], val)
    return out


def table(p: dict[str, Any]) -> list[tuple[str, str]]:
    """The variables as rows for the task page: name (phase) and how a rollout here fills it."""
    rows = []
    for e in p["vars"]:
        where = {"environment": "agent and grader", "verifier": "grader only", "solution": "reference solution"}[e["phase"]]
        if e["var"] is None:
            how = f"`{e['value'][:60]}`" if len(e["value"] or "") <= 60 else "a literal value"
        elif e["phase"] == "verifier" and p.get("judge") and e["role"] in ("key", "url", "model"):
            how = {"key": "a per-rollout key for the judge relay", "url": "the judge relay",
                   "model": f"`{e['default']}` (the task's)" if e["default"] else "the judge you pick"}[e["role"]]
        elif e["phase"] == "verifier" and p.get("judge") and e["default"] is None and _param(e["key"]):
            how = f"`{_param(e['key'])}`, the deterministic choice"
        elif e["default"] is not None:
            how = f"its default, `{e['default']}`" if e["default"] else "its default (empty)"
        elif e["role"] == "key":
            how = "empty: the agent gets no key of ours"
        else:
            how = "**not provided here**: the task can't run"
        rows.append((f"{e['key']} · {where}", ("from `${" + e["var"] + "}`: " if e["var"] else "") + how))   # keys are plain text
    return rows
