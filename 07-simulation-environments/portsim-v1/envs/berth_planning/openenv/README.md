---
title: PortSimEnv v1
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
base_path: /web
pinned: false
license: cc-by-sa-4.0
tags: [openenv, logistics, scheduling, operations-research]
short_description: Dock planning on real Port of Barcelona calls
---

# PortSimEnv v1 · OpenEnv

Plan who docks where and when, and with how many cranes, at a Port of Barcelona container quay on real 2024 calls:
one to three weeks, the terminal's real crane fleet, the port's movement and wind rules, closures, late and bunched
ships, crane outages, gales, diverted traffic, emergencies and priority cargo. One graded submit per episode against
the proven CP-SAT optimum. Serves the dock-v1 packs (1,050 train, 50 eval) by default. See the project README for the
task, the reward and the results.

## Run

```bash
uv venv -p 3.12 && uv pip install -e '.[agents,dev]'
.venv/bin/uvicorn berth_openenv.server:app --port 8011
```

| path | what |
|---|---|
| `/web` | OpenEnv's web UI: **Try Environment** (`reset(split=)`, `reset(index=)`, then play the episode in the editor and 3D view) and OpenEnv's MCP playground |
| `/viewer/` | the editor and 3D view embedded in `/web`, and live episodes. Model rollouts are in the eval Space, [FineEnvs/PortSimEnv-Eval](https://huggingface.co/spaces/FineEnvs/PortSimEnv-Eval) |
| `/reset`, `/step`, `/ws`, `/mcp` | OpenEnv |
| `/berth_planning/...` | OpenEnv Task API (`splits`, `num_tasks`, `task`, `task_range`) |
| `/api/...` | read-only JSON for the viewer (`berth_openenv/api.py`) |

## Tools

| tool | returns |
|---|---|
| `get_situation()` | quay, notices, ships already alongside, closed sections, ships to berth |
| `check_plan(plan)` | violations per ship, departure/delay/moved/cost per ship, the plan's cost; 10 per episode |
| `submit_plan(plan)` | ends the episode; reward from `BerthPlanRubric` |

`plan` is `[{"ship": id, "berth_hour": h, "section": s, "cranes": c}, ...]`, one entry per ship (`cranes` is optional
and defaults to the ship's planned cranes; a JSON string of the list is accepted too).
`reset(task_id=...)`, `reset(split=..., index=...)` or `reset(seed=...)`.

## Reward (OpenEnv rubrics)

The episode is graded once, on `submit_plan`, by `BerthPlanRubric` (every other action scores 0.0). A plan that breaks
any rule scores 0.2 × the share of ships placed cleanly. A valid plan scores 0.2 + 0.8 × exp(−gap / 0.5), with
gap = (cost − optimum) / (optimum − unavoidable + 100), so the proven CP-SAT optimum scores 1.0. No submission scores 0.
The children `ships_clean` and `cost_quality` are reported in the final observation's `metadata.rubric`.

## Smoke test

```bash
.venv/bin/python -m pytest tests -q
ANTHROPIC_API_KEY=... .venv/bin/python rollout.py --tasks 1
```

## Docker

```bash
cd ..   # envs/berth_planning
docker build -f openenv/Dockerfile -t berth-planning . && docker run -p 8000:8000 berth-planning
```

Data: Port de Barcelona open data, CC BY-SA 4.0.
