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
| `/web` | OpenEnv's web UI: the **Dock planner** tab (play an episode by hand) and OpenEnv's playground |
| `/viewer/` | the viewer: tasks, the 3D quay with a berth chart, model rollouts, live episodes |
| `/reset`, `/step`, `/ws`, `/mcp` | OpenEnv |
| `/berth_planning/...` | OpenEnv Task API (`splits`, `num_tasks`, `task`, `task_range`) |
| `/api/...` | read-only JSON for the viewer (`berth_openenv/api.py`) |

## Tools

| tool | returns |
|---|---|
| `get_situation()` | quay, notices, ships already alongside, closed sections, ships to berth |
| `check_plan(plan)` | violations per ship, departure/delay/moved/cost per ship, the plan's cost; 10 per episode |
| `submit_plan(plan)` | ends the episode; reward from `BerthPlanRubric` |

`plan` is `[{"ship": id, "berth_hour": h, "section": s}, ...]` (a JSON string of it is accepted too).
`reset(task_id=...)`, `reset(split=..., index=...)` or `reset(seed=...)`.

## Reward (OpenEnv rubrics)

`BerthPlanRubric` returns 0.0 for every action except `submit_plan`; its children `ships_clean` (fraction of ships
with no violation) and `cost_quality` (0 at the naive plan, 1 at the optimum) are reported in the final observation's
`metadata.rubric`. Bands: infeasible 0-0.2, feasible but worse than naive 0.2-0.6, naive to optimum 0.6-1.0.

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
