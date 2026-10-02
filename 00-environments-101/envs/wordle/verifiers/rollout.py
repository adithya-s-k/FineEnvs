"""End-to-end rollout: LLM playing Wordle via the in-process Verifiers env.

Verifiers is in-process. The 2 tool methods (`guess`, `get_history`) live on
the WordleToolkit class. We auto-build OpenAI tool schemas from each method's
signature + docstring via `inspect`, then drive the loop manually.

Run:
    cd 00-environments-101/envs/wordle/verifiers
    uv sync
    uv run python rollout.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

from env import WordleToolkit  # noqa: E402

MODEL = os.environ.get("ROLLOUT_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct:together")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))

SYSTEM = """You are playing Wordle. Use `guess(word)` to submit a 5-letter word.

Feedback symbols:
    🟩  letter is in the correct position
    🟨  letter is in the word but wrong position
    ⬛  letter is not in the word

You have 6 attempts. Start with a high-information word like 'crane'."""


def method_to_openai_tool(name, fn) -> dict:
    sig = inspect.signature(fn)
    doc = (fn.__doc__ or "").strip()
    description = doc.split("\n\n")[0].strip()
    properties: dict = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        ann = p.annotation
        json_type = "string"
        if ann is int:
            json_type = "integer"
        elif ann is float:
            json_type = "number"
        elif ann is bool:
            json_type = "boolean"
        properties[pname] = {"type": json_type, "description": pname}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def discover_tools(toolkit) -> tuple[list[dict], dict]:
    """Find public methods on the toolkit instance, build schemas + dispatch table."""
    tools: list[dict] = []
    table: dict = {}
    for name, fn in inspect.getmembers(toolkit, predicate=inspect.ismethod):
        if name.startswith("_") or name in {"reset", "cleanup"}:
            continue
        tools.append(method_to_openai_tool(name, fn))
        table[name] = fn
    return tools, table


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

    toolkit = WordleToolkit()
    toolkit.reset()
    tools, table = discover_tools(toolkit)

    print("=" * 80)
    print(f"Verifiers env: in-process WordleToolkit (pure Python, no E2B)")
    print(f"Provider:      {provider}    Model: {MODEL}")
    print(f"Tools:         {list(table.keys())}")
    print("=" * 80)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Start playing. Make your first guess."},
    ]

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n──── turn {turn} ────────────────────────────────────────")
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
            try:
                result = table[name](**args)
            except Exception as e:
                result = f"Tool error: {type(e).__name__}: {e}"
            text = result if isinstance(result, str) else json.dumps(result)
            print(f"[tool-result] {text}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": text}
            )
            if any(s in text for s in ("Game over", "won", "🟩🟩🟩🟩🟩")):
                print(f"\n[done] game ended (reward={toolkit.reward}).")
                return 0
    else:
        print(f"[hit MAX_TURNS={MAX_TURNS}]")

    print("\n" + "=" * 80)
    print(f"FINAL  reward={toolkit.reward}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
