"""Export the PortSimEnv v1 data the article's figures read: a few eval tasks with every model's rollout (each step
re-checked, the final plan evaluated ship by ship and graded) plus the reference plans, and the eval board.

    cd 07-simulation-environments/portsim-v1/envs/berth_planning/core
    .venv/bin/python ../../../../content/articles/simulation-rl-environments/scripts/export_portsim_data.py

Writes app/src/content/assets/data/portsim-rollouts.json and portsim-results.json next to this script's article.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from berth_core import TaskPack, evaluate, grade, parse_plan, plan_from_list
from berth_core.baselines import published
from berth_core.check import unavoidable_cost
from berth_core.model import PlanError

ARTICLE = Path(__file__).resolve().parents[1]
REPO = ARTICLE.parents[2]
ENV = REPO / "07-simulation-environments" / "portsim-v1"
RUN = ENV / "results" / "rollouts" / "dock-eval50"
OUT = ARTICLE / "app" / "src" / "content" / "assets" / "data"

TASKS = ["dock-24B-w37x1-standard-0", "dock-24B-w16x1-busy-0", "dock-36A-w06x1-busy-0", "dock-24B-w35x2-storm-1",
         "dock-24B-w16x2-extreme-0"]
MODELS = [
    ("openai:gpt-6.1-sol", "GPT-6.1 Sol"),
    ("anthropic:claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("hf:zai-org/GLM-5.3-Flash:baseten", "GLM-5.3-Flash"),
    ("hf:Qwen/Qwen3.8-2.4T-A95B:together", "Qwen3.8-2.4T"),
    ("hf:zai-org/GLM-5.3:together", "GLM-5.3"),
    ("hf:Qwen/Qwen3.8-27B:cerebras|ovhcloud", "Qwen3.8-27B"),
]


def slug(model: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model)


def plan_rows(task, plan_list):
    """Parse + evaluate + grade one plan; per-ship rows for the chart."""
    try:
        plan, problems = parse_plan(task, plan_list)
    except PlanError as e:
        plan, problems = {}, [str(e)]
    res = evaluate(task, plan)
    g = grade(task, plan, problems)
    ships = []
    for s in res.ships:
        ships.append({"id": s.ship, "berth": s.berth_hour, "section": s.section, "cranes": s.cranes, "dep": s.departure,
                      "delay": s.delay_h, "cost": s.cost, "moved": s.moved, "problems": s.problems})
    return {"reward": round(g.reward, 4), "feasible": g.feasible, "cost": g.cost, "quality": round(g.quality, 4),
            "clean": round(g.clean_fraction, 3), "violations": len(g.violations) + len(g.parse_problems),
            "parse_problems": g.parse_problems[:3], "ships": ships}


def task_json(t):
    d = t.to_dict()
    ships = [{k: s[k] for k in ("id", "name", "length_m", "sections", "arrival", "due", "workload", "min_cranes",
                                "max_cranes", "std_cranes", "weight", "berth_deadline", "deadline_penalty",
                                "planned_hour", "planned_section", "from_port", "to_port")} for s in d["ships"]]
    return {
        "id": t.task_id, "quay": t.quay, "terminal": d["terminal"], "tier": t.difficulty, "week": d["week"],
        "week_start": d["week_start_utc"], "first": d["first_section"], "last": d["last_section"],
        "section_m": d["section_m"], "crane_pool": d["rules"].get("crane_pool"), "crane_rate": d["rules"].get("crane_rate"),
        "max_moves": d["rules"].get("max_moves_per_hour"), "outages": d["rules"].get("crane_outages", []),
        "winds": d["rules"].get("no_moves", []), "blocks": d["blocks"], "notices": d["notices"], "ships": ships,
        "optimal_cost": d["reference"]["optimal_cost"], "naive_cost": d["reference"]["naive_cost"],
        "floor": unavoidable_cost(t),
    }


def main():
    pack = TaskPack([ENV / "envs" / "berth_planning" / "tasks" / "dock-v1-eval"])
    tasks = []
    for tid in TASKS:
        t = pack.get(tid)
        tj = task_json(t)
        plans = []
        ref = t.reference
        for key, label, pl in [("optimal", "Optimal (CP-SAT)", ref["optimal_plan"]), ("naive", "Naive re-plan", ref["naive_plan"])]:
            row = plan_rows(t, pl)
            plans.append({"key": key, "label": label, "kind": "reference", **row})
        pub = published(t)
        row = plan_rows(t, [{"ship": sid, "berth_hour": e[0], "section": e[1], **({"cranes": e[2]} if len(e) > 2 else {})}
                            for sid, e in pub.items()])
        plans.append({"key": "published", "label": "Port's original plan", "kind": "reference", **row})
        for model, label in MODELS:
            rec = json.loads((RUN / slug(model) / f"{tid}.json").read_text())
            final = rec["final"]
            if final["submitted"]:
                row = plan_rows(t, final["plan"])
            else:
                row = {"reward": 0.0, "feasible": False, "cost": None, "quality": 0.0, "clean": 0.0, "violations": 0,
                       "parse_problems": [], "ships": []}
            steps = []
            for s in rec["steps"]:
                if s["tool"] in ("check_plan", "submit_plan") and s.get("plan") is not None:
                    r = plan_rows(t, s["plan"])
                    steps.append({"tool": s["tool"], "feasible": r["feasible"], "cost": r["cost"], "violations": r["violations"],
                                  "reward": r["reward"] if s["tool"] == "submit_plan" else None})
                else:
                    steps.append({"tool": s["tool"]})
            plans.append({"key": model, "label": label, "kind": "model", "submitted": final["submitted"],
                          "end_reason": rec["end_reason"], "turns": rec["turns"], "seconds": round(rec["seconds"]),
                          "tokens_in": rec["usage"]["input_tokens"], "tokens_out": rec["usage"]["output_tokens"],
                          "steps": steps, **row})
        tasks.append({**tj, "plans": plans})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "portsim-rollouts.json").write_text(json.dumps({"tasks": tasks}, separators=(",", ":")))

    # the eval board: mean reward per model and tier, submit / feasible / optimal rates, tokens and time
    idx = json.loads((RUN / "index.json").read_text())
    by = defaultdict(list)
    for e in idx["episodes"]:
        by[e["model"]].append(e)
    board = []
    for model, label in MODELS:
        eps = by[model]
        tiers = defaultdict(list)
        for e in eps:
            tiers[e["difficulty"]].append(e["reward"])
        board.append({
            "model": label, "key": model, "n": len(eps), "mean": round(sum(e["reward"] for e in eps) / len(eps), 4),
            "tiers": {k: round(sum(v) / len(v), 4) for k, v in tiers.items()},
            "submitted": sum(e["submitted"] for e in eps), "feasible": sum(e["feasible"] for e in eps),
            "optimal": sum(e["reward"] >= 0.999 for e in eps),
            "tokens_out": sum(e["output_tokens"] for e in eps), "tokens_in": sum(e["input_tokens"] for e in eps),
            "median_s": round(statistics.median(e["seconds"] for e in eps)),
        })
    (OUT / "portsim-results.json").write_text(json.dumps({"tasks": 50, "board": board}, indent=1))
    print("wrote", OUT / "portsim-rollouts.json", (OUT / "portsim-rollouts.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
