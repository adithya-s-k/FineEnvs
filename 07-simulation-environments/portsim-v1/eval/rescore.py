"""Re-grade a run's saved final plans with the current grader (after a reward change) and rewrite its index.

    python eval/rescore.py dock-eval-probe --tasks envs/berth_planning/tasks/dock-v1-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envs" / "berth_planning" / "core"))

from berth_core import TaskPack, grade, parse_plan  # noqa: E402
from berth_core.model import PlanError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tasks", nargs="+", required=True)
    args = ap.parse_args()
    pack = TaskPack([ROOT / p for p in args.tasks])
    run = ROOT / "results" / "rollouts" / args.run
    idx = json.loads((run / "index.json").read_text())
    for e in idx["episodes"]:
        slug = e["model"].replace(":", "__").replace("@", "__").replace("/", "__")
        rec_path = run / slug / f"{e['task_id']}.json"
        rec = json.loads(rec_path.read_text())
        task = pack.get(e["task_id"])
        if not rec["final"]["submitted"]:
            continue
        try:
            plan, problems = parse_plan(task, rec["final"]["plan"])
        except PlanError as err:
            plan, problems = {}, [str(err)]
        g = grade(task, plan, problems)
        rec["final"]["grade"], rec["reward"] = g.as_dict(), g.reward
        rec_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        e.update(reward=g.reward, feasible=g.feasible, cost=g.cost)
    (run / "index.json").write_text(json.dumps(idx, indent=1))
    print(f"re-graded {len(idx['episodes'])} episodes in {run}")


if __name__ == "__main__":
    main()
