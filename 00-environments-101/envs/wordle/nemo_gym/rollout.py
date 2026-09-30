"""End-to-end rollout: LLM playing Wordle on the deployed NeMo Gym server.

NeMo Gym uses plain FastAPI endpoints + cookie-based sessions. No SDK needed.

    POST /seed_session    -> sets a session cookie
    POST /guess           -> {"word": "..."}    -> {"output": "<feedback>"}
    POST /get_history     -> {}                  -> {"output": "..."}

Run:
    cd 00-environments-101/envs/wordle/nemo_gym
    uv sync
    uv run python rollout.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

ENV_URL = os.environ.get("NEMO_GYM_URL", "https://fineenvs-wordle-nemo-gym.hf.space")
MODEL = os.environ.get("ROLLOUT_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct:together")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "guess",
            "description": "Submit a 5-letter word guess. Returns colored feedback (🟩🟨⬛).",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "View all previous guesses with their feedback.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = """You are playing Wordle. Use `guess(word)` to submit a 5-letter word.

Feedback symbols:
    🟩  letter is in the correct position
    🟨  letter is in the word but wrong position
    ⬛  letter is not in the word

You have 6 attempts. Start with a high-information word like 'crane' or 'slate'."""


def main() -> int:
    if ":" in MODEL:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN missing")
        llm = OpenAI(api_key=token, base_url="https://router.huggingface.co/v1")
        provider = "hf-router"
    else:
        token = os.environ.get("OPENAI_API_KEY")
        if not token:
            sys.exit("OPENAI_API_KEY missing")
        llm = OpenAI(api_key=token)
        provider = "openai"

    print("=" * 80)
    print(f"NeMo Gym server: {ENV_URL}")
    print(f"Provider: {provider}    Model: {MODEL}")
    print("=" * 80)

    s = requests.Session()
    s.post(f"{ENV_URL}/seed_session", json={}, timeout=60).raise_for_status()
    print(f"[seed_session] cookies={dict(s.cookies)}\n")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Start playing Wordle. Make your first guess."},
    ]

    for turn in range(1, MAX_TURNS + 1):
        print(f"──── turn {turn} ────────────────────────────────────────")
        r = llm.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, max_completion_tokens=256
        )
        msg = r.choices[0].message
        if msg.content:
            print(f"[assistant] {msg.content}")
        if not msg.tool_calls:
            print("[done] no tool calls.")
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"[tool-call] {name}({args})")
            resp = s.post(f"{ENV_URL}/{name}", json=args, timeout=60)
            text = resp.json().get("output", resp.text) if resp.ok else f"[HTTP {resp.status_code}]"
            print(f"[tool-result] {text}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": text}
            )
            if any(s in text for s in ("Game over", "Correct!", "🟩🟩🟩🟩🟩")):
                print("\n[done] game ended, stopping.")
                return 0
    else:
        print(f"[hit MAX_TURNS={MAX_TURNS}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
