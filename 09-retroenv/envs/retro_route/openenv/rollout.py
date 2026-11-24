#!/usr/bin/env python3
"""Run one RetroEnv episode against a server and print the trajectory.

    python rollout.py --server http://127.0.0.1:8000 --split eval --index 0 \\
        --provider anthropic --model claude-opus-5-5

Providers: anthropic (Messages API), openai (Responses API), or any
OpenAI-compatible chat endpoint via --endpoint (vLLM, HF router, OpenRouter).
For many tasks, use eval/run_eval.py in the project folder.
"""

from __future__ import annotations

import argparse
import json
import os

from retroenv_openenv import agent, agent_anthropic, agent_responses
from retroenv_openenv.client import RetroEnvClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=os.getenv("RETROENV_SERVER", "http://127.0.0.1:8000"))
    parser.add_argument("--split", default="eval")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--provider", choices=("anthropic", "openai", "chat"), default="anthropic")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL for --provider chat")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("MAX_TURNS", "16")))
    args = parser.parse_args()

    with RetroEnvClient(args.server) as env:
        opening = env.reset(args.split, index=args.index)
        print(f"task {opening['task_id']} ({args.split}[{args.index}])  target {opening['target_smiles']}")
        print(f"tools: {', '.join(opening['tools'])}\n")
        if args.provider == "anthropic":
            import anthropic

            llm = anthropic.Anthropic(api_key=os.getenv(args.api_key_env or "ANTHROPIC_API_KEY"))
            result = agent_anthropic.run_episode(
                llm, env, opening, agent_anthropic.ClaudeConfig(model=args.model, max_turns=args.max_turns)
            )
        else:
            from openai import OpenAI

            if args.provider == "openai":
                llm = OpenAI(api_key=os.getenv(args.api_key_env or "OPENAI_API_KEY"))
                result = agent_responses.run_episode(
                    llm, env, opening, agent_responses.ResponsesConfig(model=args.model, max_turns=args.max_turns)
                )
            else:
                if not args.endpoint:
                    parser.error("--endpoint is required for --provider chat")
                llm = OpenAI(base_url=args.endpoint, api_key=os.getenv(args.api_key_env or "OPENAI_API_KEY", "unused"))
                result = agent.run_episode(
                    llm, env, opening, agent.AgentConfig(model=args.model, max_turns=args.max_turns)
                )

    for message in result["transcript"]:
        if message["role"] == "tool":
            print(f"    -> {message['content'][:200]}")
            continue
        content = message.get("content")
        calls = message.get("tool_calls") or [
            block
            for block in (content if isinstance(content, list) else [])
            if block.get("type") in ("tool_use", "function_call")
        ]
        for call in calls:
            name = call.get("name") or call.get("function", {}).get("name")
            arguments = call.get("input") or call.get("arguments") or call.get("function", {}).get("arguments")
            print(f"  {name}({json.dumps(arguments)[:160] if not isinstance(arguments, str) else arguments[:160]})")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    print(f"    -> {str(block.get('content'))[:200]}")
    print(
        f"\nreward {result['reward']:.3f}  valid {result['valid']}  exact {result['exact_match']}  "
        f"turns {result['turns']}  tool calls {result['tool_calls']}"
    )
    if result["hard_failures"]:
        print("failures: " + "; ".join(result["hard_failures"][:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
