"""Build the Hugging Face dataset FineEnvs/PortSimEnv (Parquet + card) from the task packs, the source calls and
the eval rollouts.

    tasks/{train,eval}.parquet   one row per task: the prompt the agent sees, the full task, the reference plans
    calls/train.parquet          the 1,784 Port of Barcelona container calls of 2024 the tasks are built from
    rollouts/eval.parquet        the dock-eval50 rollouts: 6 models x 50 eval tasks, transcript, final plan, grade

    python tools/build_dataset.py OUT_DIR      (needs pyarrow + pandas + berth_core)
    hf upload FineEnvs/PortSimEnv OUT_DIR . --repo-type dataset
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

from berth_core import PROMPT_VERSION, TaskPack, rules, situation

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PACKS = {"eval": HERE / "tasks" / "dock-v1-eval", "train": HERE / "tasks" / "dock-v1-train"}
CALLS = ROOT / "data" / "barcelona" / "container_calls_2024.csv"
RUN = ROOT / "results" / "rollouts" / "dock-eval50"
MAX_CHECKS = 10
MODEL_NAMES = {
    "openai:gpt-6.1-sol": "GPT-6.1 Sol",
    "anthropic:claude-sonnet-5-5": "Claude Sonnet 5.5",
    "hf:zai-org/GLM-5.3-Flash:baseten": "GLM-5.3-Flash",
    "hf:Qwen/Qwen3.8-2.4T-A95B:together": "Qwen3.8-2.4T",
    "hf:zai-org/GLM-5.3:together": "GLM-5.3",
    "hf:Qwen/Qwen3.8-27B:cerebras|ovhcloud": "Qwen3.8-27B",
}


def task_rows(split: str) -> list[dict]:
    pack = TaskPack([PACKS[split]])
    path = PACKS[split] / "tasks.jsonl"
    lines = path.open() if path.is_file() else gzip.open(path.with_suffix(".jsonl.gz"), "rt")
    raw = {d["task_id"]: d for d in map(json.loads, lines)}
    rows = []
    for t in pack.tasks:
        d = raw[t.task_id]
        ref = d["reference"]
        rows.append({
            "task_id": t.task_id,
            "split": split,
            "quay": d["quay"],
            "terminal": d["terminal"],
            "difficulty": d["difficulty"],
            "week": d["week"],
            "week_start_utc": d["week_start_utc"],
            "num_ships": len(d["ships"]),
            "num_disruptions": len(d.get("disruptions") or []),
            "system_prompt": rules(t, MAX_CHECKS),
            "situation": situation(t),
            "optimal_cost": ref["optimal_cost"],
            "naive_cost": ref["naive_cost"],
            "greedy_cost": ref.get("greedy_cost"),
            "proven_optimal": bool(ref.get("proven_optimal")),
            "optimal_plan": json.dumps(ref["optimal_plan"]),
            "naive_plan": json.dumps(ref["naive_plan"]),
            "task": json.dumps(d, separators=(",", ":")),
            "prompt_version": PROMPT_VERSION,
        })
    return rows


def rollout_rows() -> list[dict]:
    index = json.loads((RUN / "index.json").read_text())
    rows = []
    for ep in index["episodes"]:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", ep["model"])  # as the viewer's api.model_slug
        path = RUN / slug / f"{ep['task_id']}.json"
        body = json.loads(path.read_text()) if path.is_file() else {}
        final = body.get("final") or {}
        rows.append({
            "run": index.get("run", "dock-eval50"),
            "model": ep["model"],
            "model_name": MODEL_NAMES.get(ep["model"], ep["model"]),
            "task_id": ep["task_id"],
            "difficulty": ep["difficulty"],
            "quay": ep["quay"],
            "num_ships": ep["ships"],
            "reward": ep["reward"],
            "submitted": ep["submitted"],
            "feasible": ep["feasible"],
            "cost": ep.get("cost"),
            "optimal_cost": ep["optimal_cost"],
            "naive_cost": ep["naive_cost"],
            "turns": ep["turns"],
            "checks": ep["checks"],
            "seconds": ep["seconds"],
            "input_tokens": ep["input_tokens"],
            "output_tokens": ep["output_tokens"],
            "end_reason": ep["end_reason"],
            "final_plan": json.dumps(final.get("plan")) if final.get("plan") is not None else None,
            "grade": json.dumps(final.get("grade")) if final.get("grade") is not None else None,
            "steps": json.dumps(body.get("steps") or []),
            "messages": json.dumps(body.get("messages") or []),
        })
    return rows


CARD = """---
license: cc-by-sa-4.0
pretty_name: PortSimEnv v1
language: [en]
task_categories: [reinforcement-learning, text-generation]
tags: [openenv, rl-environment, simulation, logistics, scheduling, operations-research, berth-allocation, real-world-data]
size_categories: [1K<n<10K]
configs:
- config_name: tasks
  default: true
  data_files:
  - split: train
    path: tasks/train.parquet
  - split: eval
    path: tasks/eval.parquet
- config_name: rollouts
  data_files:
  - split: eval
    path: rollouts/eval.parquet
- config_name: calls
  data_files:
  - split: train
    path: calls/train.parquet
---

# PortSimEnv v1

Re-plan container-ship dockings at the Port of Barcelona. Each task is a real week (or two or three) at one container
quay, built from the port's own 2024 records, with disruptions added: late and bunched ships, closed quay sections,
crane breakdowns, gales under the port's wind rules, traffic diverted from the other terminal, emergencies and priority
cargo. The agent decides when and where each ship docks and with how many cranes, and is graded once, deterministically,
against the plan a CP-SAT solver proved optimal.

| | |
|---|---|
| Environment (OpenEnv Space, 3D viewer, eval explorer) | [FineEnvs/PortSimEnv](https://huggingface.co/spaces/FineEnvs/PortSimEnv) |
| Article | [Simulation RL Environments](https://huggingface.co/spaces/FineEnvs/simulation-rl-environments) |
| Bucket (3D twin data, raw eval rollouts) | [FineEnvs/PortSimEnv](https://huggingface.co/buckets/FineEnvs/PortSimEnv) |
| Code | [adithya-s-k/FineEnvs: 07-simulation-environments/portsim-v1](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments/portsim-v1) |
| Ideas for v2, v3, post-training, data | [GitHub Discussions](https://github.com/adithya-s-k/FineEnvs/discussions/36) |

## Configs

| config | split | rows | what |
|---|---|---|---|
| `tasks` | `train` | {n_train:,} | training tasks |
| `tasks` | `eval` | {n_eval} | held-out tasks: whole week groups that never appear in train |
| `rollouts` | `eval` | {n_ro} | {n_models} models x {n_eval} eval tasks: transcript, tool calls, final plan, grade |
| `calls` | `train` | {n_calls:,} | the 2024 container calls at quays 36A (BEST) and 24B (APM Terminals) the tasks are built from |

**tasks**: `system_prompt` and `situation` are exactly what the environment sends (the rules, then the opening
message, which is also what `get_situation()` returns). `task` is the full task as JSON (ships, closures, disruptions,
rules). `optimal_plan` / `optimal_cost` are the CP-SAT reference (`proven_optimal` says whether optimality was
proven); `naive_plan` / `naive_cost` re-plan by pushing ships later. A plan is a JSON list of
`{{"ship": id, "berth_hour": h, "section": s, "cranes": c}}`.

**rollouts**: one row per (model, task) from the eval run `dock-eval50`: 12 turns and 32k output tokens per turn,
the same three tools. `messages` is the full transcript, `steps` the tool calls with each `check_plan` result,
`grade` the final grade.

## Results on the 50 eval tasks

{board}

## Reward

Deterministic, no LLM judge. A plan that breaks any rule scores at most 0.2 (0.2 x the share of ships placed cleanly).
A valid plan's cost (hours each ship leaves after its due time x its size x its priority, plus penalties for late
emergency dockings and moved ships) is compared with the optimum:
`gap = (cost - optimum) / (optimum - unavoidable + 100)`, `reward = 0.2 + 0.8 * exp(-gap / 0.5)`. No submission
scores 0.

## Use

```python
import json
from datasets import load_dataset

tasks = load_dataset("FineEnvs/PortSimEnv", "tasks", split="eval")
task = tasks[0]
print(task["system_prompt"], task["situation"], sep="\\n\\n")
```

Every task can be played on the environment Space (OpenEnv; MCP tools `get_situation`, `check_plan`,
`submit_plan`). Submitting the reference optimum scores 1.0:

```python
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient("https://fineenvs-portsimenv.hf.space").sync()
env.reset(task_id=task["task_id"])
step = env.step(CallToolAction(tool_name="submit_plan", arguments={{"plan": json.loads(task["optimal_plan"])}}))
print(step.reward)  # 1.0
```

## Source and licence

Contains data from the Port de Barcelona open data portal (https://opendata.portdebarcelona.cat/), via the 2024
snapshot in [alberto-santini/berth-allocation-problems](https://github.com/alberto-santini/berth-allocation-problems),
licensed CC BY-SA 4.0. This dataset is shared under the same licence. Crane fleets, wind rules and handling rates come
from the terminals' and the port's published information; disruptions are generated. Work in progress (v1).
"""


def board_md(ro: pd.DataFrame) -> str:
    g = ro.groupby("model_name")
    rows = []
    for name, d in sorted(g, key=lambda kv: -kv[1]["reward"].mean()):
        tiers = " | ".join(f"{d[d.difficulty == t].reward.mean():.2f}" for t in ("standard", "busy", "storm", "extreme"))
        optimal = int((d.reward >= 0.999).sum())
        rows.append(f"| {name} | **{d.reward.mean():.3f}** | {tiers} | {int(d.submitted.sum())} | "
                    f"{int(d.feasible.sum())} | {optimal} |")
    head = ("| model | mean reward | standard | busy | storm | extreme | submitted | valid | optimal |\n"
            "|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    out = args.out
    for sub in ("tasks", "calls", "rollouts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "eval"):
        df = pd.DataFrame(task_rows(split))
        df.to_parquet(out / "tasks" / f"{split}.parquet", index=False)
        counts[f"tasks/{split}"] = len(df)
    calls = pd.read_csv(CALLS, dtype={"imo": "string", "mmsi": "string", "call": "string"})
    calls.to_parquet(out / "calls" / "train.parquet", index=False)
    counts["calls/train"] = len(calls)
    ro = pd.DataFrame(rollout_rows())
    ro.to_parquet(out / "rollouts" / "eval.parquet", index=False)
    counts["rollouts/eval"] = len(ro)
    missing = int((ro["messages"] == "[]").sum())
    (out / "README.md").write_text(CARD.format(n_train=counts["tasks/train"], n_eval=counts["tasks/eval"],
                                               n_ro=len(ro), n_models=ro.model.nunique(), n_calls=len(calls),
                                               board=board_md(ro)))
    print(json.dumps({"counts": counts, "rollouts_without_transcript": missing}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
