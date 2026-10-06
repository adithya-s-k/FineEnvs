"""Run one model on a few tasks against a berth-planning server and print what happened.

    uv run python rollout.py                                   # starts a local server, Claude Sonnet 5.5, 1 task
    uv run python rollout.py --base-url http://127.0.0.1:8011 --model hf:Qwen/Qwen3.8-27B:novita --tasks 3

Keys from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY or HF_TOKEN). Full rollouts for the viewer come from
../../../eval/run_eval.py; this script is the smoke test.
"""

from __future__ import annotations

import argparse
import json

from berth_core import load_pack
from berth_openenv.agent import run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="anthropic:claude-sonnet-5-5")
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()
    base = args.base_url
    if base is None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
        from run_eval import start_server

        base = start_server()
    pack = load_pack()
    for t in [t for t in pack.tasks if t.split == args.split][: args.tasks]:
        rec = run_episode(args.model, base, t.task_id, max_turns=args.max_turns)
        print(f"\n=== {t.task_id} ({t.difficulty}, {len(t.ships)} ships) ===")
        for m in rec["messages"][2:]:
            if m["role"] == "assistant":
                calls = ", ".join(f"{c['name']}" for c in m.get("tool_calls") or [])
                print(f"assistant: {m['content'][:200]!r} -> {calls or 'no tool call'}")
            elif m["role"] == "tool":
                print(f"  {m['name']}: {m['content'][:240]}")
        g = (rec["final"] or {}).get("grade") or {}
        print(f"reward {rec['reward']:.3f}  end {rec['end_reason']}  cost {g.get('cost')}  "
              f"naive {t.reference['naive_cost']}  optimum {t.reference['optimal_cost']}  turns {rec['turns']}")
        if rec.get("errors"):
            print("errors:", json.dumps(rec["errors"]))


if __name__ == "__main__":
    main()
