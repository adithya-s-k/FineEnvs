<div align="center">

<h1>PortSimEnv v1</h1>

<h3>Re-plan a week of container-ship dockings at the Port of Barcelona, on the port's real 2024 records</h3>

</div>

PortSimEnv is an OpenEnv environment for berth planning. The agent gets one quay at the Port of Barcelona, the ships
that really called there in a 2024 week, and a week that has just gone wrong: late ships, closed quay sections, crane
breakdowns, gales, emergencies. It decides when, where and with how many cranes every ship docks, and is graded once,
deterministically, against a plan CP-SAT proved optimal. There are 1,050 training and 50 eval tasks.

## Try it

| | |
|---|---|
| Play an episode in the browser (OpenEnv Space) | [FineEnvs/PortSimEnv](https://huggingface.co/spaces/FineEnvs/PortSimEnv) |
| Watch the six eval models in 3D | [FineEnvs/PortSimEnv-Eval](https://huggingface.co/spaces/FineEnvs/PortSimEnv-Eval) |
| Tasks, source calls and eval rollouts | [datasets/FineEnvs/PortSimEnv](https://huggingface.co/datasets/FineEnvs/PortSimEnv) |
| The write-up | [Simulation RL Environments, part 1](https://huggingface.co/spaces/FineEnvs/simulation-rl-environments) |
| Ideas and questions | [GitHub discussion #36](https://github.com/adithya-s-k/FineEnvs/discussions/36) |

Connect an agent to the hosted environment:

```python
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient("https://fineenvs-portsimenv.hf.space").sync()
obs = env.reset(task_id="dock-24B-w07x1-busy-0")       # or reset(split="train", index=0)
rules = obs.observation.metadata["instructions"]         # the system prompt
situation = env.step(CallToolAction(tool_name="get_situation", arguments={}))
plan = [{"ship": 0, "berth_hour": 0, "section": 9, "cranes": 3}]   # one entry per ship in the situation
print(env.step(CallToolAction(tool_name="check_plan", arguments={"plan": plan})).observation)
```

## Why the plans are real

A container terminal's quay is a row of numbered sections. Every ship needs a run of consecutive sections for its
time alongside, and the berth planner decides where and when each one docks and how many quay cranes work it. The
Port of Barcelona publishes every 2024 call with the sections the port actually assigned, so the plans here are real:
1,784 container calls at quay 36A (Terminal Catalunya, BEST, sections 2-30) and quay 24B (APM Terminals, sections
2-22).

The recorded plan is almost conflict-free (ETA is effectively docking time), so an undisturbed week has nothing to
optimise. Each task takes real weeks, adds what berth planners deal with, and asks for a new plan.

## Packs

| pack | tasks | what it is |
|---|---|---|
| **dock-v1** (`tasks/dock-v1-eval`, `tasks/dock-v1-train`) | 50 eval + 1,050 train | the hard pack: 1-3 weeks, real crane fleets, movement limits, gales, outages, emergencies, priority cargo |
| berth-v1 (`tasks/berth-v1`) | 100 (80 train, 20 test) | the first pack: one week, sections and hours only, 25 each easy / medium / hard / expert |

### dock-v1

**Rules on every task** (sources in `data/barcelona/SOURCE.md` and `berth_core/generate.py`):

| rule | value | grounded in |
|---|---|---|
| quay cranes | BEST 13, quay 24B 9 (its share of APM's 14 on 1,515 m) | the terminals' 2024 fleets |
| cranes per ship | 1-7 by length (one per 50 m); standard 1/2/3/4 below 180/260/330 m and above | calibrated so every 2024 fortnight fits the real fleets (peaks 12-13 and 8-9) |
| work | container moves at 28 moves per crane-hour; hours alongside = ceil(moves / (28 × cranes)) | European crane productivity; recorded stays |
| movements | at most 3 (36A) / 2 (24B) ships dock or leave per hour | the 2024 record's maximum; pilot rules |
| wind | ships ≥300 m may not move above 25 kn, nobody above 30 kn; finished ships wait alongside | Port of Barcelona traffic ordinance (2023) |

**Disruptions**, drawn per tier and seed: closures, late ships, bunching (several delayed ships arriving together, as
in the 2024 Red Sea diversions), crane outages, gales, the other quay's traffic diverted here (36A), unscheduled
calls, genuine emergencies (a reefer failure: dock by a deadline or pay per hour) and priority cargo (lateness counts
triple).

| tier | weeks | eval | train | ships (eval, median) | adds |
|---|---|---:|---:|---|---|
| standard | 1 | 9 | 266 | 15-32 (24) | a closure, late ships, maybe an outage, an emergency or a priority ship |
| busy | 1-2 | 15 | 260 | 14-55 (27) | more of each, bunching, sometimes a gale |
| storm | 2 | 13 | 262 | 23-59 (32) | a gale every time, an emergency, priority cargo, sometimes diverted traffic |
| extreme | 2-3 | 13 | 262 | 24-59 (35) | everything stacked: two gales, diverted traffic, several emergencies and priorities |

**Splits never share a week.** Eval windows lie inside 10 held-out ISO weeks (5-7, 10, 15-17, 35-37); train windows
avoid them. Train tasks reuse their real weeks with different seeds, so the same ships face different disruptions. The
pack was built with 100 eval tasks over 20 held-out weeks; to keep evals affordable the eval set was cut to 50 by moving
whole week groups (20, 25-27, 30, 40, 45-47, 50) and their 50 tasks to train, so the separation still holds. Eval tiers:
9 standard / 15 busy / 13 storm / 13 extreme, 27 at quay 36A and 23 at 24B; train: 266 / 260 / 262 / 262.

**A task is kept only if** CP-SAT gets within 1 % of its bound, neither the naive re-plan (reward ≤ 0.6) nor a greedy
heuristic (≤ 0.85) comes close, and it has at least 12 ships. Eval: 100 of 150 candidates kept, 50 of them later moved
to train. Train: 1,000 of 1,573 candidates kept (223 too easy, 337 not solved within 1 %), plus those 50, for 1,050
tasks with 12-90 ships, 555 at quay 36A and 495 at 24B. Every one of the 1,100 references is proven optimal. In
practice the naive re-plan never scores above 0.44 and the greedy heuristic never above 0.82. The train pack ships
gzipped (1.1 MB).

```bash
cd envs/berth_planning/core && uv venv -p 3.12 && uv pip install -e '.[dev]'
.venv/bin/python -m berth_core.build_dock --split eval --n 100 --out ../tasks/dock-v1-eval
.venv/bin/python -m berth_core.build_dock --split train --n 1000 --out ../tasks/dock-v1-train   # resumable
```

## The task

The agent reads the situation (quay, notices, ships already alongside, closed sections, cranes and movement limits,
wind windows, a table of ships with arrival, work, crane range, planned slot, planned departure and notes) and returns
a docking hour, a first section and (dock-v1) a crane count for every ship.

- **Rules:** dock at or after arrival and not inside a wind window that applies; stay on the quay; never share a
  section-hour with another ship, a ship already alongside or a closed section; cranes within the ship's range and,
  hour by hour, within the cranes available; movements per hour within the limit.
- **Cost:** Σ sections × hours late against the planned departure (× 3 for priority cargo), + the emergency rate per
  hour an emergency docks after its deadline, + 5 per scheduled ship moved off its planned sections.
- **Tools:** `get_situation()`, `check_plan(plan)` (violations and cost of the agent's own plan, 10 per episode, never a
  score or a reference), `submit_plan(plan)` (ends the episode, graded once). 24 tool calls per episode.

## The reward

Graded once, on `submit_plan`, as OpenEnv rubrics (`BerthPlanRubric` with children `ships_clean` and `cost_quality`;
every other action scores 0.0) on top of `berth_core.reward`, which the offline tools share.

**dock-v1 (reward v3).** Each task stores the CP-SAT optimum. The grader also computes a provable floor: each ship on
its own, docking at its earliest legal hour with the most cranes it can take (`check.unavoidable_cost`). Mistakes are
measured against the part of the cost a planner controls:

| submitted plan | reward |
|---|---|
| unreadable, malformed or infeasible | 0.2 × clean ships / (ships + entry problems), always below 0.2 |
| feasible | 0.2 + 0.8 × exp(−g / 0.5), g = (cost − optimum) / (optimum − floor + 100) |

Any feasible plan beats any infeasible one, and only the optimum scores 1.0. About three quarters of a typical
optimum is unavoidable delay, which is why the gap is not measured against the optimum itself; measured that way a
plan three times the optimum scored 0.93 on a big task.

**How it is kept honest** (`core/tests/`, 331 tests):
- an independent checker (an hour-by-section occupancy grid written separately) agrees with the grader on feasibility,
  cost and which ships break rules, over 240 perturbed plans per sampled task;
- hostile plans (duplicates, unknown ships, booleans, NaN, huge or negative numbers, zero cranes, 50× repeated
  entries, text inside the JSON) never crash grading and never pass;
- no plan beats a proven optimum, and the floor is below every feasible plan;
- lazy policies stay low on every task: the published plan is infeasible, the naive re-plan scores ≤ 0.6, a greedy
  heuristic ≤ 0.85, and docking ships one after another sits at the 0.2 floor.


## Results

All 50 eval tasks, six models, one rollout each, 12 turns and 32k output tokens per turn, reward v3
(`results/rollouts/dock-eval50/`). Open models ran through the Hugging Face router on the fastest provider for each;
Qwen3.8-27B fell back from Cerebras to OVHcloud when Cerebras returned 5xx.

| Model | n | Mean reward (95% CI) | standard | busy | storm | extreme | Feasible | Optimal | Ended without submit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| openai:gpt-6.1-sol | 50 | **0.888** (0.833-0.940) | 0.998 | 0.900 | 0.864 | 0.822 | 100% | 52% | 0 |
| anthropic:claude-sonnet-5-5 | 50 | **0.782** (0.711-0.849) | 0.921 | 0.797 | 0.675 | 0.774 | 98% | 20% | 1 |
| hf:zai-org/GLM-5.3-Flash:baseten | 50 | **0.470** (0.357-0.588) | 0.767 | 0.500 | 0.460 | 0.240 | 50% | 18% | 14 |
| hf:Qwen/Qwen3.8-2.4T-A95B:together | 50 | **0.380** (0.271-0.493) | 0.580 | 0.449 | 0.328 | 0.215 | 46% | 10% | 16 |
| hf:zai-org/GLM-5.3:together | 50 | **0.313** (0.198-0.440) | 0.624 | 0.342 | 0.224 | 0.154 | 32% | 20% | 33 |
| hf:Qwen/Qwen3.8-27B:cerebras\|ovhcloud | 50 | **0.211** (0.115-0.318) | 0.397 | 0.274 | 0.219 | 0.003 | 26% | 6% | 34 |

Every model scores lower on extreme weeks than on standard ones, and even the best matches the proven optimum on only
26 of 50 tasks, so there is plenty of headroom. The open models lose most of their score by never submitting a plan.
They spend their 32k-token turns reasoning (3.1-4.2 M output tokens over the 50 tasks, against 0.41 M for GPT-6.1
Sol) and end without calling `submit_plan`, or submit plans that break the crane pool, movement limit or wind rules.
GPT-6.1 Sol used 41k input and 8k output tokens per task on average, with a median of 2 min 15 s per task.

The first pack, berth-v1 (`tasks/berth-v1`, one week, no cranes), and a 10-task prototype of the harder rules
(`tasks/berth-frontier-v1`) are kept for reference; their runs are not included here.

## Run it locally

```bash
cd envs/berth_planning/openenv
uv venv -p 3.12 && uv pip install -e '.[agents,dev]'
.venv/bin/uvicorn berth_openenv.server:app --port 8011
```

The server serves dock-v1 by default (`eval` 50 and `train` 1,050). Set `BERTH_TASKS_DIR` to a colon-separated list
of pack directories to serve others, for example `tasks/berth-v1`.

- `http://127.0.0.1:8011/web` is OpenEnv's web UI. **Try Environment** has `reset(split=)` and `reset(index=)`, then
  the editor: drag ships in time and along the quay, set cranes, watch the 3D quay follow, and call the same MCP tools
  the agent has (`get_situation`, `check_plan`, `submit_plan`), graded by the env's rubric. OpenEnv's MCP playground
  is the other tab.
- `http://127.0.0.1:8011/viewer/` is the full viewer, including the explorer of tasks and rollouts that the eval Space
  serves (point `BERTH_RUNS_DIR` at `results/rollouts`).
- `/reset`, `/step`, `/ws`, `/mcp` and the Task API under `/berth_planning/` are OpenEnv's.

Run the eval yourself (this is how `dock-eval50` was produced):

```bash
cd 07-simulation-environments/portsim-v1
envs/berth_planning/openenv/.venv/bin/python eval/run_eval.py --run my-run --max-tokens 32000 \
    --models anthropic:claude-sonnet-5-5 hf:Qwen/Qwen3.8-27B:cerebras --split eval --limit 10
envs/berth_planning/core/.venv/bin/python eval/rescore.py my-run --tasks envs/berth_planning/tasks/dock-v1-eval   # after a reward change
```

Publish everything (bucket, dataset, environment Space, eval Space, collection, article) with
`envs/berth_planning/tools/publish_all.sh`; it takes a step number to resume.

## The 3D view

**The 3D scene is a digital twin of the port** (`web/twin/`, built by `tools/twin/build_twin.py` from open data):
the real coastline, quays and breakwaters, land cover, 46,913 building footprints with tagged or estimated heights, 582 tanks, Montjuïc
and the rise to Collserola from terrain tiles, and the skyline (Sagrada Família, Torre Glòries, Hotel Arts, the Fira
towers, the Collserola tower). Each quay is placed where it is (BEST 1,526 m at bearing 37.6°, APM's face 1,162 m at
27°) and built the way the terminal is: BEST's 34 automated stacking blocks from OSM with two blue stacking cranes
each, shuttle carriers, its on-dock rail terminal and white-and-carmine STS cranes; APM's straddle-carrier yard, its
yellow STS cranes and the three light-blue 2025 "Triple-E" cranes, the CLH tank farm and Montjuïc behind it, cruise
ships at the Moll Adossat opposite. Ships wear their line's livery (MSC, Maersk, CMA CGM, ONE, Hapag-Lloyd, ...),
come in through the south entrance, wait in separated offshore anchorage slots, move laterally alongside with tugs, and turn in the entrance basin after backing clear of the quay. Sources and
licences are in `web/twin/SOURCES.md` (© OpenStreetMap contributors, ODbL).

The 3D replay is a visualisation of the submitted schedule and never changes the cost or the reward.

- **Traffic.** A deterministic kinematic controller moves ships and tugs through the harbour. It checks hull and tug
  envelopes against the shoreline, moored cruise ships and other traffic, respects wind windows and closures, and
  holds a ship at anchor or alongside, with a reason, when the plan leaves no safe path. It is not a hydrodynamic
  solver.
- **Cargo.** Cranes work real bay positions. Each visible discharge moves one specific container from the deck to a
  carrier and on to a reserved yard slot. This is representative handling, not a container manifest.
- **Controls.** Camera presets (Overview, Harbour, Quayside, Overhead), Mediterranean or Golden-hour light, playback
  from real time to 16 h/s, Cargo (follow one transfer), Focus (follow a ship) and Cinema (full window).

Viewer regression checks (Node 22+, no npm install):

```bash
cd envs/berth_planning/openenv
node --test tests/viewer/*.test.mjs
```

These cover both quays and all 50 optimal eval replays (1,692 vessels including ships already alongside), sampling
hull, escort, cruise-ship and shoreline separation throughout manoeuvres, along with wind, closures, completion, deterministic seeking,
invalid plans, anchorage overflow and crane allocation. The two densest training tasks also run through the same checks.
Targeted collision regressions cover thin piers, shoreline holes, between-frame crossings, escort side changes,
blocked departures and unplanned anchorage obstacles; busy APM and BEST plans also receive independent three-second audits.
Cargo checks cover ownership conservation, continuous handoffs, top-down stack access, preserved orientation,
hoist/trolley/carrier speed limits, wind deferral, unique storage, route separation, pausing and deterministic seeking.

## Layout

```
07-simulation-environments/portsim-v1/
├── data/barcelona/          the 2024 container calls (CC BY-SA 4.0, see SOURCE.md)
├── envs/berth_planning/
│   ├── core/                berth_core: tasks, checker, naive + CP-SAT references, reward, prompts, pack builders
│   │                        (dock.py + build_dock.py for dock-v1; baselines.py for the lazy-policy checks)
│   ├── tasks/dock-v1-eval/  50 eval tasks (tasks.jsonl, manifest.json, build_log.jsonl)
│   ├── tasks/dock-v1-train/ 1,050 train tasks (tasks.jsonl.gz)
│   ├── tasks/berth-v1/      the first 100-task pack
│   ├── tools/               twin/ (the 3D digital twin from open data), build_dataset.py, deploy_space.py, publish_all.sh
│   └── openenv/             berth_openenv: environment, rubric, server, viewer (web/), agent harness, rollout.py
├── eval/                    run_eval.py (rollouts through the server), report.py, rescore.py
└── results/rollouts/        one directory per run: index.json, board.md, one JSON per episode
```

## Data and licence

Port of Barcelona open data, CC BY-SA 4.0 ("Contains data from the Port de Barcelona open data portal"); the task
pack is a derived database under the same licence. The 3D twin uses OpenStreetMap data (© OpenStreetMap contributors,
ODbL 1.0) and Terrain Tiles on AWS; see `web/twin/SOURCES.md`. Code is Apache-2.0 like the rest of the repository.
The idea of the viewer owes a lot to [Portwise](https://github.com/alexngdev99/rork-seaport-logistics-3d) (MIT), a
fictional Singapore terminal visualisation.
