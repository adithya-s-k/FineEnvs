# AirOpsEnv v1–v2 · AgentEnv

The agentenv-airops plugin's two envs as [AgentEnv](https://github.com/scaleapi/agentenv-framework) environments,
with the same tools and their text, the same limits, the same rewards and the same packs, so every published run
replays here byte for byte. Each env is an MCP server on port 18765 behind AgentEnv's gateway:

| env | server | image | agent | verifier |
|---|---|---|---|---|
| `airops-day` (v1) | `airops_agentenv/live.py` | `Dockerfile` | `airops-llm` (`agents/airops-llm/`) | `airops-verifier` (`verifiers/airops-verifier/`) |
| `airops-occ` (v2) | `airops_agentenv/occ/live.py` | `Dockerfile.occ` | `airops-desk` (`agents/airops-desk/`), one per desk | `airops-occ-verifier` (`verifiers/airops-occ-verifier/`) |

`airops_agentenv/tasks.py` and `occ/tasks.py` write each day's task steps (load the day, hide `airops_notice`, give
each desk's role its tools, register a trigger per watch, deploy the agents, arm the clock, play, end the day, grade),
the same files the published runs played. The day logic is `airops_core` (`../core`); [DESIGN.md](../../../DESIGN.md)
says how it works.

## Run

First fetch the days the envs play, from `envs/airops`:

```bash
uv run -p 3.12 --with pyarrow python tools/fetch_packs.py      # 11 MB from the dataset at v0.3.3, checked by sha256
```

Then, from here:

```bash
uv run -p 3.12 python rollout.py --smoke                              # v1, no model or key: wiring-noplay
uv run -p 3.12 python rollout.py --env occ --smoke                    # v2, no model: the crew system alone
uv run -p 3.12 python rollout.py --env occ --script                   # v2, no model: the scripted desks, both modes
uv run -p 3.12 python rollout.py --model anthropic/claude-haiku-5-5 --tasks 2   # through LITELLM_* or [model]
HF_TOKEN=hf_... uv run -p 3.12 python rollout.py --model zai-org/GLM-5.3-Flash:baseten   # v1 on the HF router
uv run -p 3.12 python rollout.py --env occ --model openai/gpt-6-luna --mode one --task-id b6-20250111-0400
```

`rollout.py` stages a bundle in `runs/bundle` and runs it with agent-env in its own process. It needs Docker; agent-env
builds the env and agent images on first use, starts a registry on `127.0.0.1:5000` and keeps its stores under
`~/.local/state/agent-env`. Each run writes `runs/<UTC>/results.jsonl` and each play step's transcript under
`runs/<UTC>/transcripts/`. The zero-spend runs check themselves: `--smoke` must cost what the pack's reference says
(v1's naive day, v2's idle day), and `--script` the pack's scripted references, with the audit passing and no spend.
Flags: `--env day|occ`, `--model`, `--tasks N` (v1 by fewest watches, v2 by date), `--task-id ID` (repeatable), `--mode
joint|one|both`, `--episode-cap-usd` (2; v1 the episode's, v2 each desk's; the v2 board ran at 5 a desk for two desks
and 8 for one agent, and one board desk spent $2.68), `--hf-bill-to ORG`, `--keep`. Only v1's agent prices the Hugging
Face router's models; v2's desks take Anthropic and OpenAI models through LiteLLM. A v1 day takes a few minutes; a
two-desk v2 day with GPT-6.1 Sol took a median of 37 minutes and $2, one agent 20 minutes and $1.

The agents play only the models in their `PRICES`; when `AIROPS_MAX_COST_USD` is set they stop before a request that
could take the episode (v2: the desk) past it. `airops-desk` with `AIROPS_DESK_POLICY=script` plays a fixed script with
no model, which `--script` uses.

## Tests

```bash
uv run -p 3.12 --frozen --extra dev pytest ../core/tests -q        # offline, on committed fixtures
uv run -p 3.12 --frozen --extra dev pytest tests -q                 # the same; the minute-long replays are marked slow
uv run -p 3.12 --frozen --extra dev pytest tests -q -m slow         # three follower days and the recorded v2 day
AIROPS_IMAGE=airops-day-agentenv AIROPS_OCC_IMAGE=airops-occ-agentenv \
    uv run --frozen --extra dev pytest tests/env/test_recorded_runs.py tests/occ/test_occ_server.py -q   # the images
uv run --frozen --extra dev ruff check . ../core ../tools
```

The fixtures, all committed: four v1 days and three v2 dates cut from the packs (`tests/fixtures/b6-v1/`,
`../core/tests/fixtures/b6-v2/`), the plugin's fixture v1 days, real windows without their news
(`../core/tests/fixtures/tasks.jsonl`), a slice of the raw inputs for the pack builders
(`../core/tests/pack/fixtures/`), the dataset's task files for those days (`tests/fixtures/bundles/`), and two recorded
runs (`tests/fixtures/runs/`): v1's `b6-bos-20250216-1500` with the day its verifier read, and the v2 board's one-agent
day of 2025-01-11. No test needs the fetched packs, the network, Docker or a model; with `AIROPS_IMAGE` and
`AIROPS_OCC_IMAGE` set, the server tests start a container of each image instead. CI's test job runs the first two lines
and ruff; its image job runs the slow replays and the image tests.

## Docker

```bash
cd ..   # envs/airops, with the packs fetched
docker build -f agentenv/Dockerfile -t airops-day-agentenv .
docker build -f agentenv/Dockerfile.occ -t airops-occ-agentenv .
docker build -t airops-llm agentenv/agents/airops-llm
docker build -t airops-desk agentenv/agents/airops-desk
```

Each env image holds its own pack only; `Dockerfile.dockerignore` and `Dockerfile.occ.dockerignore` keep the rest of
the context out.

## Parity with the plugin

`../core/airops_core` is the plugin's `src/agentenv_airops` at v0.3.3, file for file, without the parts the envs don't
use (the stopped flow model, the CLI, the sweep and trial-run runners, the viewer and the Hub tooling): every module is
byte for byte the plugin's except `occ/pack.py`, where the pack's default path changed, and the two package docstrings
(`__init__.py`, `occ/__init__.py`). The env servers here differ from the plugin's only in their imports and the v1
pack's path, the verifiers not at all, and the agents by two checks: a task whose time limit is within the 1,200 seconds
the harness keeps for its last turn, or runs out before the first turn, now fails, where the plugin's agents ended the
day at turn 0 and let it be graded. Replayed through this code, all 5 published v1 runs give every tool output byte for
byte and the graded day their verifier read, and all 55 recorded v2 attempts (the board's 48, its retry and the 6 runs
of the trial sweep before it) give every call's result, the grade, the audit and the digest. The whole record the env
served matches too, but for one detail on one date when the replay runs on macOS: on 2026-06-14 (one agent) the
end-of-day crew problem has two equal-cost ways to send two cabin units home, and HiGHS picks the other one there, so
their shares come out 67.7 passenger-minutes apart from the record's, the day's cost the same. Replayed in the
`airops-occ` image, on Linux as the board ran, that record matches in full too (`../tools/replay_runs.py` over the
dataset's `runs/`):

```bash
cd ..   # envs/airops
uv run -p 3.12 --with huggingface-hub hf download earakely-scale/AirOpsEnv-AgentEnv --repo-type dataset \
    --revision v0.3.3 --include 'runs/*' 'raw/*' --local-dir cache/dataset
uv run --project agentenv python tools/replay_runs.py cache/dataset/runs --raw cache/dataset/raw/airops.jsonl \
    --workers 8
```

## Rebuilding the packs from the public records

The packs are built from the public BTS, FAA and IEM files in `../data/`, which the tools download into `cache/` (not
in git) and check against the manifests' pins. They need `pyarrow` (`--extra dev`) and a large machine for v2.

| step | tool | what |
|---|---|---|
| v1 pack | `tools/build_pack.py` | the b6-v1 days from the pinned inputs; `--keep-references` carries the references over; a rerun writes the same bytes. Its manifest keeps the plugin's words (`scripts/build_pack.py`, `cache/inputs/`), so a rebuild writes the published file |
| v1 references | `tools/references.py references tasks/b6-v1/tasks.jsonl` | hindsight, floor, the follower, the blind follower, naive and the airline, into each task line |
| v2 world, without crews | `tools/world_check.py --out cache/world` | every storm date of the 19 months as a whole-network task, its anchors and the world's checks against the record |
| v2 crews | `tools/crew_check.py --out cache/crews --tasks cache/world` | the crews: trips, the record replay, the calm day, the best plan and its bound per date (2 h 34 min on 96 workers) |
| v2 pack | `tools/build_occ_pack.py --out cache/occ --run cache/crews --split tasks/b6-v2-split.json --tasks cache/world` | each kept date's units frozen at 04:00 and its references; `--assemble cache/occ --pack tasks/b6-v2 --scripted cache/scripted` writes the pack |
| v2 scripted references | `tools/scripted_desks.py --out cache/scripted` | the scripted desks' costs, and the checks the references rest on: idle, the best plan confirmed at 04:00 scoring 1.0, the audit, a leak test, the page sizes, a repeat run (about 5 minutes a date) |

`tasks/b6-v2-split.json` is the published split: the seed-0 draw over the dates the crew filter kept
(`programs/network.py`, `split`), each eval date with the storm hub it was drawn for, and the train dates.
