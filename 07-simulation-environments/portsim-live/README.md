<div align="center">

<a href="https://huggingface.co/spaces/earakely-scale/PortSimEnv-AgentEnv"><img src="./assets/banner.jpg" alt="PortSimEnv live: GPT-6.1 Sol's week at the Port of Barcelona on AgentEnv, graded at its last watch on PortSimEnv's 3D quay" width="100%"></a>

<h1>PortSimEnv v2–v5: live, marine, wind and lines</h1>

<h3>Run a quay at the Port of Barcelona through a week as it unfolds</h3>

<p>PortSimEnv's weeks played watch by watch on AgentEnv's virtual clock: the news arrives late, pilots and tugs run
short, and the wind is the wind that blew.</p>

<a href="https://huggingface.co/spaces/earakely-scale/PortSimEnv-AgentEnv"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Replays%20in%203D-FFD21E?style=for-the-badge&labelColor=1a1a1a" alt="Replays in 3D" height="32"></a>
<a href="https://huggingface.co/datasets/earakely-scale/PortSimEnv-AgentEnv"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Tasks%20%26%20runs-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Dataset" height="32"></a>
<a href="https://github.com/scaleapi/agentenv-framework"><img src="https://img.shields.io/badge/framework-AgentEnv-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="AgentEnv" height="32"></a>
<a href="https://github.com/earakely-scale/agentenv-portsim-plugin"><img src="https://img.shields.io/badge/GitHub-agentenv--portsim%20plugin-181717?style=for-the-badge&logo=github&labelColor=1a1a1a" alt="The agentenv-portsim plugin on GitHub" height="32"></a>

<sub>GPT-6.1 Sol ends a live week at the hindsight optimum; every run replays in the Space. Twin © OpenStreetMap
contributors (<a href="https://www.openstreetmap.org/copyright">ODbL</a>) · terrain: Terrain Tiles on AWS.</sub>

</div>

---

## The task

[PortSimEnv v1](../portsim-v1/) hands the agent a broken week at one quay and grades one plan for all of it. Here the
same quay and the same 2024 weeks are played as they unfold. The week starts at Monday 00:00 with only what the port
knows then. The rest arrives in 6-hour bulletins, the watches: ships report delays and unscheduled calls, the harbour
master issues gale warnings and emergencies, terminal ops announce crane outages, the line desk flags priority cargo.
At each watch the agent confirms berth windows; a window about to start is frozen. When it calls `advance`, the clock
jumps to the next bulletin and the ships berth and sail on what it confirmed. The week is graded once,
deterministically, on the confirmed windows against the week as it really happened.

There are three envs, each adding to the one before:

| env | the week | graded against |
|---|---|---|
| `portsim-live` | 15 one-week eval weeks of dock-v1, news watch by watch | the hindsight optimum CP-SAT proves |
| `portsim-marine` | the same weeks, and every berthing and departure takes a pilot and tugs from pools shared with the port's other 2024 traffic | the marine hindsight optimum |
| `portsim-wind` | 15 marine weeks moved into real Barcelona weather weeks of 2023-2025; an ECMWF forecast at every watch, graded on the wind that blew | the anchor: the lower of the hindsight optimum and a forecast-following re-planner's cost |

| tool | what it does |
|---|---|
| `get_situation()` | the watch, its time, the week as known now, the confirmed windows and the ships without one |
| `check_plan(plan)` | rule breaks and cost of windows merged over the confirmed ones; confirms nothing |
| `confirm_berths(plan)` | sets or changes windows; each refusal carries the harbour master's reason |
| `advance()` | ends the watch; after the last one the rest of the week runs and is graded |

`check_plan` and `confirm_berths` share 3 calls a watch. A valid week scores `0.2 + 0.8·e^(−gap/0.5)`, so the optimum
(on the wind weeks, the anchor) scores 1.0; a week that breaks a rule scores below 0.2. News that lands on a frozen
window is excused, not charged.

These envs run on AgentEnv only: they need its gateway's virtual clock, which the env moves at each `advance`, and its
triggers, one per watch, which deliver the news through a tool hidden from the agent. There is no OpenEnv port.
[DESIGN.md](DESIGN.md) explains the watches, the clock and triggers, the rules and the grade. The agentenv-portsim
plugin and its Hub dataset call live, marine and wind v2, v3 and v4.

## Results

Twelve models on the 15 weeks of each env, one run each, 540 runs:

| Model | live | marine | wind | All 45 weeks | Feasible weeks |
|---|---:|---:|---:|---:|---:|
| GPT-6 Astra | 0.984 | 0.956 | 0.896 | **0.945** | 44 of 45 |
| Claude Opus 5.5 | 0.976 | 0.951 | 0.906 | **0.944** | 45 of 45 |
| Claude Sonnet 5.5 | 0.904 | 0.932 | 0.888 | **0.908** | 45 of 45 |
| GPT-6.1 Sol | 0.957 | 0.905 | 0.745 | **0.869** | 41 of 45 |
| Claude Haiku 5.5 | 0.694 | 0.639 | 0.583 | **0.639** | 44 of 45 |
| Qwen3.8-2.4T | 0.686 | 0.519 | 0.662 | **0.622** | 28 of 45 |
| DeepSeek V4.1 Flash | 0.754 | 0.598 | 0.480 | **0.611** | 28 of 45 |
| GLM-5.3 | 0.748 | 0.593 | 0.468 | **0.603** | 28 of 45 |
| GLM-5.3-Flash | 0.470 | 0.461 | 0.328 | **0.420** | 18 of 45 |
| Kimi K3 | 0.465 | 0.446 | 0.249 | **0.386** | 18 of 45 |
| GPT-6 Luna | 0.335 | 0.287 | 0.320 | **0.314** | 36 of 45 |
| Qwen3.8-27B | 0.055 | 0.073 | 0.090 | **0.073** | 0 of 45 |

On the same weeks the rolling re-planner (on the wind weeks, the forecast-following one) scores 1.000 on every env,
and the naive policy 0.202, 0.187 and 0.183. GPT-6 Astra and Claude Opus 5.5 share the top. The three best open models
average within noise of Claude Haiku 5.5, but each breaks a rule in 17 of its 45 weeks, and Haiku in one. Qwen3.8-27B
never finds a valid plan, the clearest headroom for training. These scores are not comparable with v1's eval: the
agent re-plans as the week unfolds, with 47 turns a week (52 on the wind weeks), and plays 15 standard and busy weeks
per env rather than 50 across four tiers. [results/board.md](results/board.md) has each env's table with intervals,
and how the runs were made.

## Start here

| You want to… | Open |
|---|---|
| Play a week on AgentEnv | [envs/port_week/agentenv](envs/port_week/agentenv/) |
| Read how the watches, clock, triggers, rules and grade work | [DESIGN.md](DESIGN.md) |
| See every model's score | [results/board.md](results/board.md) |
| Watch any run in 3D | [earakely-scale/PortSimEnv-AgentEnv](https://huggingface.co/spaces/earakely-scale/PortSimEnv-AgentEnv) |
| Get the weeks and every recorded run, as the plugin publishes them | [datasets/earakely-scale/PortSimEnv-AgentEnv](https://huggingface.co/datasets/earakely-scale/PortSimEnv-AgentEnv) |
| Browse the Space, dataset and runs together | [the collection](https://huggingface.co/collections/earakely-scale/portsimenv-on-agentenv) |
| Plan the whole week in one go | [PortSimEnv v1](../portsim-v1/) |
| Run sweeps, film runs or publish to the Hub | [agentenv-portsim plugin](https://github.com/earakely-scale/agentenv-portsim-plugin) |

## Run it locally

Each env plays through `rollout.py`, which stages a task bundle and runs it with AgentEnv on local Docker:

```bash
cd envs/port_week/agentenv
uv run --frozen --extra dev pytest tests -q                    # the envs, the task steps and six recorded weeks
uv run python rollout.py --smoke                               # no model or key: a wiring week scores 0, audit ok
HF_TOKEN=hf_... uv run python rollout.py --env marine          # GLM-5.3-Flash on the Hugging Face router
uv run python rollout.py --model anthropic/claude-sonnet-5-5 --tasks 3   # through LITELLM_* or .agentenv/config.toml
```

The week model (`port_week_core`) has its own tests, run through the same project:

```bash
uv run --frozen --extra dev pytest ../core/tests -q             # real-wind cases skip until the wind build has run
```

**The wind weeks are built on your machine.** They derive from the readings of Meteocat's station Y7, which this
repository doesn't redistribute, so it holds only the ECMWF forecast values, the weeks' identifiers and sha256 pins. One
command fetches the readings, builds the weeks and checks them against the pins:

```bash
cd envs/port_week
uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/build_wind.py
cd agentenv && uv run python rollout.py --env wind
```

The full build took 24 minutes here: 15 to fetch the readings and 9 to solve on 14 processes. A rerun reuses its
cache. Until it has run, `--env wind` stops and points to `tools/build_wind.py`, and so does the server for
`portsim-wind`.

## Layout

```
07-simulation-environments/portsim-live/
├── README.md, DESIGN.md     this page; how the watches, clock, triggers, rules and grade work
├── assets/banner.jpg        the banner: the last watch of a live week in the Space's 3D replay
├── data/
│   ├── barcelona/           the port's 2024 calls of every ship type, for the marine pack (pinned, not committed)
│   ├── ecmwf/               ECMWF HRES 10 m wind at the port, 2023-2026 (points.jsonl.gz, CC BY 4.0)
│   └── meteocat/            pins of the Y7 readings and of the built wind weeks; no readings
├── envs/port_week/
│   ├── core/                port_week_core: the week in watches, the live, marine and wind worlds, packs, prompts
│   ├── agentenv/            port_week_agentenv: the three envs' server, task steps, verifier, rollout.py, Dockerfile
│   ├── tasks/dock-v1-marine/  the 18 marine weeks (tasks.jsonl, manifest.json)
│   ├── tasks/live/, marine/ the rolling and naive references (references.jsonl)
│   ├── tasks/dock-v1-wind/, wind/  the wind weeks and references, built by tools/build_wind.py (git-ignored)
│   └── tools/               the data tools: build_wind.py, wind_weather.py and the three *_references.py
└── results/board.md         twelve models on the live, marine and wind weeks
```

The live weeks are portsim-v1's `dock-v1-eval` tasks, read through `berth_core`, and the agent is portsim-v1's
`portsim-llm` ([../portsim-v1/envs/berth_planning/agentenv/agent](../portsim-v1/envs/berth_planning/agentenv/agent/)).

## Data and licence

Port of Barcelona open data, CC BY-SA 4.0 ("Contains data from the Port de Barcelona open data portal"): the marine
pack and the live and marine references are derived databases under the same licence
([data/barcelona/SOURCE.md](data/barcelona/SOURCE.md)). The forecasts contain modified ECMWF open data, CC BY 4.0, with
ECMWF's full statement in [data/ecmwf/SOURCE.md](data/ecmwf/SOURCE.md). Nothing derived from Meteocat's readings is
committed: the wind weeks are built on your machine under the portal's terms, which this repository doesn't relicense
([data/meteocat/SOURCE.md](data/meteocat/SOURCE.md)). Check those terms before you share the built weeks, the results
and transcripts of wind runs (they repeat the observed windows), or an image or dataset made with them. The plugin's
Hub dataset is published by the plugin under the terms on its card and isn't part of this repository. Code is
Apache-2.0 like the rest of the repository.

Contributed from the agentenv-portsim plugin (https://github.com/earakely-scale/agentenv-portsim-plugin) by Edgar
Arakelyan (Scale AI), built on AgentEnv (https://github.com/scaleapi/agentenv-framework).
