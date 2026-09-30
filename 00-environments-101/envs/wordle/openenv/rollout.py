"""End-to-end rollout: LLM playing Wordle on the deployed OpenEnv server.

Pattern (same as the jupyter_agent/openenv rollout):
    1. Connect to the OpenEnv server with `openenv`'s generic MCPToolClient.
    2. Auto-discover the tools (`guess`, `get_history`, `reset_game`).
    3. Convert MCP tool schemas to OpenAI tool schemas.
    4. Drive a multi-turn Wordle loop with Qwen-Coder via HF Router (or any
       OpenAI model if ROLLOUT_MODEL has no ":" suffix).
    5. Stop after 6 guesses, when the model wins, or when MAX_TURNS is hit.

Run:
    cd 00-environments-101/envs/wordle/openenv
    uv sync
    uv run python rollout.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openenv.core.mcp_client import MCPToolClient

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

ENV_URL = os.environ.get("OPENENV_URL", "https://fineenvs-wordle-openenv.hf.space")
MODEL = os.environ.get("ROLLOUT_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct:together")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "8"))   # 6 guesses + a couple of slack turns

SYSTEM = """You are playing Wordle. The hidden word is exactly 5 lowercase letters.

Each call to `guess(word)` returns colored feedback:
    🟩  letter is in the correct position
    🟨  letter is in the word but wrong position
    ⬛  letter is not in the word

You have 6 attempts. Use the feedback from each guess to narrow down the answer.
Start with a high-information word like 'crane' or 'slate'. Only output a tool
call — do not narrate. Stop calling tools once you've won or used 6 guesses."""


def mcp_tools_to_openai(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]


def main() -> int:
    if ":" in MODEL:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN missing in repo-root .env")
        llm = OpenAI(api_key=token, base_url="https://router.huggingface.co/v1")
        provider = "hf-router"
    else:
        token = os.environ.get("OPENAI_API_KEY")
        if not token:
            sys.exit("OPENAI_API_KEY missing in repo-root .env")
        llm = OpenAI(api_key=token)
        provider = "openai"

    print("=" * 80)
    print(f"OpenEnv server: {ENV_URL}")
    print(f"Provider:       {provider}")
    print(f"Model:          {MODEL}")
    print("=" * 80)

    with MCPToolClient(base_url=ENV_URL).sync() as env:
        env.reset()
        tools = mcp_tools_to_openai(env.list_tools())
        print(f"\nDiscovered {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Start playing. Make your first guess."},
        ]
        won = False

        for turn in range(1, MAX_TURNS + 1):
            print(f"──── turn {turn} ────────────────────────────────────────")
            r = llm.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, max_completion_tokens=256
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
                result = env.call_tool(name, **args)
                text = result if isinstance(result, str) else str(result)
                print(f"[tool-result] {text}")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": text}
                )
                if "🟩🟩🟩🟩🟩" in text or "won" in text.lower():
                    won = True
                if "Game over" in text or "game over" in text.lower():
                    print(f"\n[done] env reported game over (won={won}).")
                    return 0

            if won:
                print("\n[done] looks like a win 🎉")
                break
        else:
            print(f"[hit MAX_TURNS={MAX_TURNS}]")

    print("\n" + "=" * 80)
    print(f"FINAL  won={won}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
