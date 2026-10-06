<div align="center">

<h1>PortSimEnv v1</h1>

<h3>Plan who docks where and when at a Port of Barcelona container quay, on real 2024 data</h3>

<p>A multi-turn OpenEnv environment: one to three real weeks at a quay, the terminal's real crane fleet and the port's own rules, stacked disruptions, one graded submit against the proven CP-SAT optimum. 1,050 train and 50 eval tasks, with a playable 3D dock planner on a digital twin of the port and an explorer for rollouts.</p>

</div>

---

## What this is

A container terminal's quay is a row of numbered sections. Every ship needs a run of consecutive sections for its
time alongside, and the dock planner (the industry says *berth planner*) decides where and when each one docks and
how many quay cranes work it. The Port of Barcelona publishes every 2024 call with the sections the port actually
assigned, so the plans here are real: 1,784 container calls at quay 36A (Terminal Catalunya, BEST, sections 2-30) and
quay 24B (APM Terminals, sections 2-22).

The recorded plan is almost conflict-free (ETA is effectively docking time), so an undisturbed week has nothing to
optimise. Each task takes real weeks and applies what dock planners deal with, then asks for a new plan.

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

**A task is kept only if** CP-SAT gets within 1 % of its bound (every eval task is proven optimal), neither the naive
re-plan (reward ≤ 0.6) nor a greedy heuristic (≤ 0.85) comes close, and it has at least 12 ships. Eval: 100 of 150 candidates kept (50 now in eval, 50 moved to train).
Train: 1,000 of 1,573 kept (223 too easy, 337 not solved within 1 %), 12-90 ships, 529 at quay 36A and 471 at 24B,
52 / 86 / 36 / 52 distinct real windows per tier. Every one of the 1,100 references is proven optimal. The train pack
ships gzipped (1.1 MB).

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

**How it is kept honest** (`core/tests/`, 288 tests):
- an independent checker (an hour-by-section occupancy grid written separately) agrees with the grader on feasibility,
  cost and which ships break rules, over 240 perturbed plans per sampled task;
- hostile plans (duplicates, unknown ships, booleans, NaN, huge or negative numbers, zero cranes, 50× repeated
  entries, text inside the JSON) never crash grading and never pass;
- no plan beats a proven optimum, and the floor is below every feasible plan;
- lazy policies stay low on every task: the published plan is infeasible, the naive re-plan scores ≤ 0.6, a greedy
  heuristic ≤ 0.85, and docking ships one after another sits at the 0.2 floor.

berth-v1 keeps its original banded reward (anchored on the naive re-plan), so its board stays comparable.

## Results

### dock-v1 eval (50 tasks)

All 50 eval tasks, six models, 12 turns, 32k tokens per turn, reward v3 (`results/rollouts/dock-eval50/`; open models
through the Hugging Face router on the fastest provider found for each, Qwen3.8-27B falling back from Cerebras to
OVHcloud when Cerebras returned 5xx).

| Model | n | Mean reward (95% CI) | standard | busy | storm | extreme | Feasible | Optimal | Ended without submit |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| openai:gpt-6.1-sol | 50 | **0.888** (0.833-0.940) | 0.998 | 0.900 | 0.864 | 0.822 | 100% | 52% | 0 |
| anthropic:claude-sonnet-5-5 | 50 | **0.782** (0.711-0.849) | 0.921 | 0.797 | 0.675 | 0.774 | 98% | 20% | 1 |
| hf:zai-org/GLM-5.3-Flash:baseten | 50 | **0.470** (0.357-0.588) | 0.767 | 0.500 | 0.460 | 0.240 | 50% | 18% | 14 |
| hf:Qwen/Qwen3.8-2.4T-A95B:together | 50 | **0.380** (0.271-0.493) | 0.580 | 0.449 | 0.328 | 0.215 | 46% | 10% | 16 |
| hf:zai-org/GLM-5.3:together | 50 | **0.313** (0.198-0.440) | 0.624 | 0.342 | 0.224 | 0.154 | 32% | 20% | 33 |
| hf:Qwen/Qwen3.8-27B:cerebras\|ovhcloud | 50 | **0.211** (0.115-0.318) | 0.397 | 0.274 | 0.219 | 0.003 | 26% | 6% | 34 |

Every model falls from standard to extreme weeks, and even the best matches the proven optimum on only half the
tasks, so the set is far from saturated. The open models lose most of their score by never submitting a plan: they
spend the 32k-token turns reasoning (2-4 M output tokens over the 50 tasks, against 0.4 M for GPT-6.1 Sol) and end
without calling `submit_plan`, or submit plans that break the crane pool, movement limit or wind rules. Cost per task
was about $0.16 for GPT-6.1 Sol (41k input / 8k output tokens, median 2.3 min) and $0.44 for Sonnet 5.5 (75k / 29k,
median 3 min). An earlier 10-task check on the 100-task eval set is in `results/rollouts/dock-eval-probe/`.

### berth-v1

Five models on all 100 tasks (`results/rollouts/board/`), 12 turns, 32k tokens per turn, reward as defined above.
36 Qwen3.8-27B episodes that ended on provider connection errors are being re-run; its row will move slightly.

| Model | n | Mean reward (95% CI) | easy | medium | hard | expert | Feasible | Beat naive | Optimal | Median turns |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openai:gpt-6-astra@low | 100 | **0.975** (0.959-0.988) | 0.947 | 0.985 | 0.991 | 0.976 | 100% | 98% | 62% | 3 |
| anthropic:claude-sonnet-5-5 | 100 | **0.965** (0.951-0.978) | 0.942 | 0.977 | 0.971 | 0.969 | 100% | 100% | 51% | 2 |
| hf:zai-org/GLM-5.3 | 100 | **0.642** (0.557-0.725) | 0.755 | 0.741 | 0.749 | 0.322 | 70% | 63% | 33% | 3 |
| hf:Qwen/Qwen3.8-27B:deepinfra | 100 | **0.488** (0.403-0.575) | 0.604 | 0.554 | 0.500 | 0.295 | 52% | 45% | 20% | 4 |
| hf:openai/gpt-oss-120b | 100 | **0.452** (0.383-0.527) | 0.555 | 0.517 | 0.498 | 0.238 | 65% | 30% | 12% | 6 |

Frontier models nearly saturate this set; the spread is between frontier and open models, which is the useful range
for RL on open models. On tasks with 27+ ships the frontier models hit the exact optimum only 9-18 % of the time,
but the reward band anchored on the naive plan hides that.

**Harder tier (prototype, `tasks/berth-frontier-v1`, 10 tasks):** two-week horizons (20-56 ships) with quay cranes at
the terminals' real 2024 counts (BEST 13, quay 24B 9), workload in container moves at 28 moves per crane-hour, the
record's hourly movement limit, a gale under the port's 2023 wind rules, a crane outage and diverted traffic; reward
v2 (gap to the optimum).

| Model | n | Mean reward (95% CI) | easy | medium | hard | expert | Feasible | Beat naive | Optimal | Median turns |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openai:gpt-6-astra@low | 10 | **0.982** (0.959-0.997) | - | - | - | - | 100% | 100% | 60% | 3 |
| anthropic:claude-sonnet-5-5 | 10 | **0.885** (0.764-0.966) | - | - | - | - | 100% | 100% | 30% | 3 |

Even so, GPT-6 Astra finds the proven optimum on 6 of 10. Realistic constraints make the planning richer but not
frontier-hard while all information is given up front; the next step is a live simulation where information arrives
over time (see DESIGN-live.md once written).

## Run it

```bash
cd envs/berth_planning/openenv
uv venv -p 3.12 && uv pip install -e '.[agents,dev]'
.venv/bin/uvicorn berth_openenv.server:app --port 8011
```

The server serves dock-v1 by default (splits `eval` 50 and `train` 1,050); set `BERTH_TASKS_DIR` to a
colon-separated list of pack directories to serve others (for example `tasks/berth-v1`).

- `http://127.0.0.1:8011/web` — OpenEnv's web UI. The **Dock planner** tab is where a person plays an episode: pick a
  task through the Task API, start an OpenEnv session over `/ws`, edit the plan on the dock chart (drag ships in time
  and along the quay, set cranes, 3D quay alongside, with crane and movement strips and wind windows), and call the
  same MCP tools the agent has — `get_situation`, `check_plan`, `submit_plan` — with the reward coming from the env's
  rubric. No model rollouts or reference plans on it. OpenEnv's own Playground tab sits next to it.
- `http://127.0.0.1:8011/viewer/` — the **Explorer**, full page: every task in 3D with the published, naive and
  optimal plans, the model board, every rollout step by step, and live episodes.

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
**Traffic and visual replay.** Both the OpenEnv planner and evaluation explorer use the same deterministic
kinematic traffic controller. It checks oriented hull and tug envelopes against complete shoreline edges,
moored cruise ships, waiting vessels and other moving convoys. Clearance samples adapt to speed and turning
rate so no envelope corner travels more than two metres between checks. Cruise-ship positions and tug offsets
are shared by the renderer and controller. The approach admits one manoeuvre at a time and respects wind
windows and maintenance pockets. APM traffic stays on the container-terminal side of the cruise basin. BEST's route
passes north of the inner breakwater before turning into the basin. Anchorage exit aisles keep departing vessels
clear of ships waiting offshore. Acceleration, deceleration, speed-dependent wakes, restrained water motion and
tug-assisted lateral movement make transitions legible; this is a kinematic visualisation, not a hydrodynamic solver.

The chart, plan costs, RL observations and reward still describe the **submitted schedule**. If physical traffic
needs more clearance than the hourly plan allows, the 3D replay waits and reports a visual traffic delay; the scrubber
extends through the final departure. Impossible spatial assignments, early arrivals and wind-blocked docking
instructions stay at anchor with a reason, while their original violations remain in the chart. Ships already
alongside may finish clearing a berth in the first hour of a closure; its workboat waits until the entire convoy
has cleared. An obstructed departure holds safely alongside with a reason. Tug offsets remain on the same side
through turns, and interpolated headings avoid abrupt yaw changes along rounded routes.
Cranes work real cargo-bay positions, avoid the accommodation block, respect the requested count and park spare
booms raised.

Container handling uses an explicit, deterministic transfer sequence. Each visible discharge removes a specific
topmost deck container only when the spreader locks onto it. Its colour, markings and dimensions follow it through
vertical hoisting, a clearance-height trolley crossing, lowering and release on the quay. A carrier waits for the
STS spreader to clear, collects that same box and sets it down in a reserved yard slot. Containers remain there
after delivery; pausing freezes handling, and seeking reconstructs both deck and yard inventory. Four hoist ropes,
open spreader frames, lock/unlock dwell, acceleration, speed limits and carrier-route reservations replace the
independent colour-changing crane and yard loops. BEST uses shuttle carriers and block-end transfer slots; APM
uses taller straddle carriers and reserved slots in the mapped yard rows.

This is **representative deck handling**, not a container manifest or a rigid-body physics solver. The tasks only
provide handling hours/crane workload, so the replay allocates a finite sample of actual modelled containers to
available yard spaces. It does not invent loading manifests or change the task's work, costs or reward.
**Cargo** frames the next transfer, plays it at **10× real time**, and pauses after delivery and the carrier's return.
Use **Real time** to inspect twistlocks and hoisting, or change the speed to resume ordinary continuous playback.

Use **Overview**, **Harbour**, **Quayside** or **Overhead** to inspect the port. **3 min/s** and **15 min/s** make manoeuvres easier
to follow. **Harbour** frames the wider port, Montjuïc and surrounding city. Select a ship and use **Focus** to inspect its current position, including its anchorage slot.

**Rendering.** Switch between **Mediterranean** daylight and **Golden hour**; these are presentation presets,
independent of the simulation's UTC clock. Sky, sun, environment lighting, shadows and water reflections change
together. **Cinema** expands the scene; press Escape or **Exit cinema** to return. The scene uses absorbing teal
water with wind-driven ripples, corrugated container normals, smooth plated hulls with waterline wear, reflective
bridge glazing, textured paving, and detailed tank farms, warehouse facades and pine canopies. Small architectural
fixtures and foliage are procedural details on the mapped footprints. Textures are generated locally with fixed
seeds; no external model or texture downloads are required. Desktop rendering uses contact shadows, antialiasing
and planar reflections; compact/touch devices retain the lighter rendering path.
The surrounding city includes the local street network and a georeferenced surface atlas, so neighbourhoods
remain visible beyond the detailed building range. Roads follow the rendered terrain triangles; parking areas
and boulevard trees are placed around mapped buildings and carriageways. The source and rebuild instructions
for the scenery supplement are in `web/twin/SOURCES.md`.

Viewer regression checks (Node 22+; no npm install needed):

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

- `/reset`, `/step`, `/ws`, `/mcp`, and the Task API under `/berth_planning/` are OpenEnv's.

Rollouts:

```bash
cd 07-simulation-environments/portsim-v1
envs/berth_planning/openenv/.venv/bin/python eval/run_eval.py --run my-run \
    --models anthropic:claude-sonnet-5-5 hf:Qwen/Qwen3.8-27B:deepinfra --split eval --limit 10
envs/berth_planning/core/.venv/bin/python eval/rescore.py my-run --tasks envs/berth_planning/tasks/dock-v1-eval   # after a reward change
```

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
