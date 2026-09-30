"""End-to-end rollout: LLM playing Wordle on the deployed ORS server.

Run:
    cd 00-environments-101/envs/wordle/ors
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
from openreward import EnvironmentsAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

ENV_URL = os.environ.get("ORS_URL", "https://fineenvs-wordle-ors.hf.space")
ENV_NAME = os.environ.get("ORS_ENV_NAME", "wordleors")
SPLIT = os.environ.get("ORS_SPLIT", "train")
TASK_INDEX = int(os.environ.get("TASK_INDEX", "0"))
MODEL = os.environ.get("ROLLOUT_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct:together")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))

SYSTEM = """You are playing Wordle. Use the `guess` tool with a 5-letter word.

Feedback symbols:
    🟩  letter is in the correct position
    🟨  letter is in the word but wrong position
    ⬛  letter is not in the word

Use the feedback to refine your next guess. You have 6 attempts. Start with a
high-information word like 'crane' or 'slate'."""


def ors_tools_to_openai(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def blocks_to_text(blocks) -> str:
    return "\n".join(getattr(b, "text", str(b)) for b in (blocks or []))


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
    print(f"ORS server: {ENV_URL}")
    print(f"Env name:   {ENV_NAME}    split={SPLIT}  task_index={TASK_INDEX}")
    print(f"Provider:   {provider}    Model: {MODEL}")
    print("=" * 80)

    api = EnvironmentsAPI(base_url=ENV_URL, api_key="")
    env = api.get(ENV_NAME)
    tasks = env.list_tasks(SPLIT)
    print(f"\n{len(tasks)} tasks in '{SPLIT}'. Using task #{TASK_INDEX}: answer={tasks[TASK_INDEX].task_spec.get('answer','?')}")
    tools = ors_tools_to_openai(env.list_tools())
    print(f"Discovered {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

    with env.session(task=tasks[TASK_INDEX]) as session:
        user_msg = blocks_to_text(session.get_prompt())
        print(f"[task] {user_msg}\n")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        cumulative_reward = 0.0
        finished = False

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
                out = session.call_tool(name, args)
                text = blocks_to_text(out.blocks)
                cumulative_reward += float(out.reward or 0.0)
                print(f"[tool-result reward={out.reward} finished={out.finished}] {text}")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": text}
                )
                if out.finished:
                    finished = True
            if finished:
                print("\n[done] env reported finished=True, stopping.")
                break
        else:
            print(f"[hit MAX_TURNS={MAX_TURNS}]")

    print("\n" + "=" * 80)
    print(f"FINAL  cumulative_reward={cumulative_reward}  finished={finished}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
