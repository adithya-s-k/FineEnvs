"""Oracle: play each task's proven-optimal plan (CP-SAT, shipped in the task pack) through a running server.

Every task should score 1.0. Used by `openenv.yaml` (validation.capabilities.oracle) and as the deploy smoke test:

    python oracle/solve.py --url https://fineenvs-portsimenv.hf.space --split eval --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys

from berth_core import load_pack
from berth_openenv.agent import EnvSession


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--task", action="append", help="task id (repeatable); default: the first --limit tasks of --split")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.001)
    args = ap.parse_args(argv)
    pack = load_pack()
    tasks = [pack.get(t) for t in args.task] if args.task else [t for t in pack.tasks if t.split == args.split][: args.limit]
    worst = 1.0
    for task in tasks:
        env = EnvSession(args.url)
        try:
            env.reset(task.task_id)
            _, done, reward, meta = env.call("submit_plan", {"plan": task.reference["optimal_plan"]})
        finally:
            env.client.close()
        worst = min(worst, reward if reward is not None else 0.0)
        print(json.dumps({"task_id": task.task_id, "done": done, "reward": reward}))
    ok = worst >= 1.0 - args.tolerance
    print(json.dumps({"tasks": len(tasks), "min_reward": worst, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
