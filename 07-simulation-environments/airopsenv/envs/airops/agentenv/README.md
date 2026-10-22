# AirOpsEnv v1–v3 · AgentEnv

The agentenv-airops plugin's three envs as [AgentEnv](https://github.com/scaleapi/agentenv-framework) environments,
with the same tools and their text, the same limits, the same rewards and the same packs, so every published run
replays here byte for byte. Each env is an MCP server on port 18765 behind AgentEnv's gateway:

| env | server | image | agent | verifier |
|---|---|---|---|---|
| `airops-day` (v1) | `airops_agentenv/live.py` | `Dockerfile` | `airops-llm` (`agents/airops-llm/`) | `airops-verifier` (`verifiers/airops-verifier/`) |
| `airops-occ` (v2) | `airops_agentenv/occ/live.py` | `Dockerfile.occ` | `airops-desk` (`agents/airops-desk/`), one per desk | `airops-occ-verifier` (`verifiers/airops-occ-verifier/`) |
| `airops-occ3` (v3) | `airops_agentenv/occ/live3.py` | `Dockerfile.occ3` | `airops-desk`, one per desk, the customer desk in role `customer` | `airops-occ3-verifier` (`verifiers/airops-occ3-verifier/`) |

`airops_agentenv/tasks.py`, `occ/tasks.py` and `occ/tasks3.py` write each day's task steps (load the day, hide
`airops_notice`, give each desk's role its tools, register a trigger per watch, deploy the agents, arm the clock, play,
end the day, grade), the same files the published runs played. The day logic is `airops_core` (`../core`);
[DESIGN.md](../../../DESIGN.md) says how it works.

## Run

First fetch the days the envs play, from `envs/airops`:

```bash
uv run -p 3.12 --with pyarrow python tools/fetch_packs.py      # 21 MB from the dataset at v0.4.2, checked by sha256
```

Then, from here:

```bash
uv run -p 3.12 python rollout.py --smoke                              # v1, no model or key: wiring-noplay
uv run -p 3.12 python rollout.py --env occ --smoke                    # v2, no model: the crew system alone
uv run -p 3.12 python rollout.py --env occ --script                   # v2, no model: the scripted desks, both modes
uv run -p 3.12 python rollout.py --model anthropic/claude-haiku-5-5 --tasks 2   # through LITELLM_* or [model]
HF_TOKEN=hf_... uv run -p 3.12 python rollout.py --model zai-org/GLM-5.3-Flash:baseten   # v1 on the HF router
uv run -p 3.12 python rollout.py --env occ --model openai/gpt-6-luna --mode one --task-id b6-20250111-0400
uv run -p 3.12 python rollout.py --env occ3 --smoke                   # v3, no model: the crew and passenger systems
uv run -p 3.12 --extra dev python rollout.py --env occ3 --script      # v3, no model: the scripted desks, both modes
uv run -p 3.12 python rollout.py --env occ3 --model openai/gpt-6.1-sol --mode joint --episode-cap-usd 3
```

`rollout.py` stages a bundle in `runs/bundle` and runs it with agent-env in its own process. It needs Docker; agent-env
builds the env and agent images on first use, starts a registry on `127.0.0.1:5000` and keeps its stores under
`~/.local/state/agent-env`. Each run writes `runs/<UTC>/results.jsonl` and each play step's transcript under
`runs/<UTC>/transcripts/`. The zero-spend runs check themselves, with the audit passing and no spend: on v1 and v2,
`--smoke` must cost what the pack's reference says (v1's naive day, v2's idle day) and `--script` the pack's scripted
references. b6-v3 keeps no idle or scripted reference, so on v3 each must cost what the same day costs played here in
process: no desk, or the agent's own scripts at every desk (`Script` in `agents/airops-desk/agent.py`, which is why v3's
`--script` needs the dev extra). Flags: `--env day|occ|occ3`, `--model`, `--tasks N` (v1 by fewest watches, v2 and v3
by date), `--task-id ID` (repeatable), `--mode joint|one|both`, `--episode-cap-usd` (2; v1 the episode's, v2 and v3
each desk's; the v2 board ran at 5 a desk for two desks and 8 for one agent, and one board desk spent $2.68; the v3
board at 3 a desk for three desks and 6 for one agent), `--hf-bill-to ORG`, `--keep`. Only v1's agent prices the
Hugging Face router's models; the desks take Anthropic and OpenAI models through LiteLLM. A v1 day takes a few
minutes; a two-desk v2 day with GPT-6.1 Sol took a median of 37 minutes and $2, one agent 20 minutes and $1; a
three-desk v3 day 70 minutes and $2.79, one agent 44 minutes and $1.52.

The agents play only the models in their `PRICES`; when `AIROPS_MAX_COST_USD` is set they stop before a request that
could take the episode (v2 and v3: the desk) past it. `airops-desk` with `AIROPS_DESK_POLICY=script` plays a fixed
script with no model, which `--script` uses.

## Tests

```bash
uv run -p 3.12 --frozen --extra dev pytest ../core/tests -q        # offline, on committed fixtures
uv run -p 3.12 --frozen --extra dev pytest tests -q                 # the same; the minute-long replays are marked slow
uv run -p 3.12 --frozen --extra dev pytest tests -q -m slow         # three follower days, the recorded v2 and v3 days
AIROPS_IMAGE=airops-day-agentenv AIROPS_OCC_IMAGE=airops-occ-agentenv AIROPS_OCC3_IMAGE=airops-occ3-agentenv \
    uv run --frozen --extra dev pytest tests/env/test_recorded_runs.py tests/occ/test_occ_server.py \
    tests/occ/test_occ3_server.py -q                                # the images
uv run --frozen --extra dev ruff check . ../core ../tools
```

The fixtures, all committed: four v1 days, three v2 dates and two v3 dates cut from the packs (`tests/fixtures/b6-v1/`,
`../core/tests/fixtures/b6-v2/`, `../core/tests/fixtures/b6-v3/`), the plugin's fixture v1 days, real windows without
their news (`../core/tests/fixtures/tasks.jsonl`), a slice of the raw inputs for the pack builders
(`../core/tests/pack/fixtures/`), the dataset's task files for those days (`tests/fixtures/bundles/`), and three
recorded runs (`tests/fixtures/runs/`): v1's `b6-bos-20250216-1500` with the day its verifier read, the v2 board's
one-agent day of 2025-01-11 and the v3 board's three-desk day of the same date. v3's passenger tests read the survey
extracts in `../tasks/pax/`. No test needs the fetched packs, the network, Docker or a model; with `AIROPS_IMAGE`,
`AIROPS_OCC_IMAGE` and `AIROPS_OCC3_IMAGE` set, the server tests start a container of each image instead. CI's test job
runs the first two lines and ruff; its image job runs the slow replays and the image tests.

## Docker

```bash
cd ..   # envs/airops, with the packs fetched
docker build -f agentenv/Dockerfile -t airops-day-agentenv .
docker build -f agentenv/Dockerfile.occ -t airops-occ-agentenv .
docker build -f agentenv/Dockerfile.occ3 -t airops-occ3-agentenv .
docker build -t airops-llm agentenv/agents/airops-llm
docker build -t airops-desk agentenv/agents/airops-desk
```

Each env image holds its own pack only; `Dockerfile.dockerignore`, `Dockerfile.occ.dockerignore` and
`Dockerfile.occ3.dockerignore` keep the rest of the context out.

## Parity with the plugin

`../core/airops_core` is the plugin's `src/agentenv_airops` at v0.4.2, file for file, without the parts the envs don't
use (the stopped flow model, the CLI, the sweep and trial-run runners, the viewer and the Hub tooling): every module is
byte for byte the plugin's except `occ/pack.py` and `programs/pax.py`, where the packs' and the survey extracts' default
paths changed, and the two package docstrings (`__init__.py`, `occ/__init__.py`). The env servers here differ from the
plugin's only in their imports, the v1 pack's path and, for v3's task steps, the verifier's path and the bundle README's
deploy line (the v1 and v2 task modules also drop the plugin's click CLI); the verifiers not at all; and the agents by
two checks, for every desk: a task whose time limit is within the 1,200 seconds the harness keeps for its last turn, or
runs out before the first turn, now fails, where the plugin's agents ended the day at turn 0 and let it be graded. With
no passengers v3's code leaves every v2 output byte for byte as it was (a test checks the digests). Replayed through
this code, all 5 published v1 runs give every tool output byte for byte and the graded day their verifier read; all 55
recorded v2 attempts (the board's 48, its retry and the 6 runs of the trial sweep before it) and all 48 v3 board
attempts give every call's result, the grade, the audit and the digest. The whole record the env served matches too
(v3's passengers included), but for one detail on two dates when the replay runs on macOS: on v2's 2026-06-14 (one
agent) and v3's 2025-12-14 (three desks) the end-of-day crew problem has two equal-cost ways to send two cabin units
home, and HiGHS picks the other one there, so their shares come out 67.7 passenger-minutes apart from the record's, the
day's cost the same. Replayed on Linux, as the boards ran, the v2 record matches in full in the `airops-occ` image, and
so do all 48 v3 records in the `airops-occ3` image (`../tools/replay_runs.py` over the dataset's `runs/`):

```bash
cd ..   # envs/airops
uv run -p 3.12 --with huggingface-hub hf download earakely-scale/AirOpsEnv-AgentEnv --repo-type dataset \
    --revision v0.4.2 --include 'runs/*' 'raw/*' --local-dir cache/dataset
uv run --project agentenv python tools/replay_runs.py cache/dataset/runs --raw cache/dataset/raw/airops.jsonl \
    --workers 8                                      # --only v3 for the v3 board alone, about 5 minutes a day
```

## Rebuilding the packs from the public records

The packs are built from the public BTS, DOT, FAA and IEM files `../../../data/` describes, which the tools download
into `cache/` (not in git) and check against the manifests' pins. They need `pyarrow` (`--extra dev`) and a large
machine for v2 and v3.

| step | tool | what |
|---|---|---|
| v1 pack | `tools/build_pack.py` | the b6-v1 days from the pinned inputs; `--keep-references` carries the references over; a rerun writes the same bytes. Its manifest keeps the plugin's words (`scripts/build_pack.py`, `cache/inputs/`), so a rebuild writes the published file |
| v1 references | `tools/references.py references tasks/b6-v1/tasks.jsonl` | hindsight, floor, the follower, the blind follower, naive and the airline, into each task line |
| v2 world, without crews | `tools/world_check.py --out cache/world` | every storm date of the 19 months as a whole-network task, its anchors and the world's checks against the record |
| v2 crews | `tools/crew_check.py --out cache/crews --tasks cache/world` | the crews: trips, the record replay, the calm day, the best plan and its bound per date (2 h 34 min on 96 workers) |
| v2 pack | `tools/build_occ_pack.py --out cache/occ --run cache/crews --split tasks/b6-v2-split.json --tasks cache/world` | each kept date's units frozen at 04:00 and its references; `--assemble cache/occ --pack tasks/b6-v2 --scripted cache/scripted` writes the pack |
| v2 scripted references | `tools/scripted_desks.py --out cache/scripted` | the scripted desks' costs, and the checks the references rest on: idle, the best plan confirmed at 04:00 scoring 1.0, the audit, a leak test, the page sizes, a repeat run (about 5 minutes a date) |
| v3 survey extracts | `tools/survey_shares.py 2025Q1=<DB1B Market CSV> 2025-07=<DB1C Market parquet> ...` | `tasks/pax/`: JetBlue's B6-to-B6 transfers per period and station, DB1C's connection times and the median fare per period, from the DOT's Origin and Destination Survey (DB1B to 2025 Q2, DB1C after), downloaded by hand from the URLs in its docstring |
| v3 aircraft plans | `tools/passenger_plans.py --out cache/v3/plans --split all --seeds 4 --time-limit 1800` | per date the passengers booked (`pax.book` on `tasks/pax` and the following days' BTS schedules and T-100 loads) and the passenger-aware aircraft plans under the 3 h line; on the eval dates, step 0's premise (G1) |
| v3 references | `tools/passenger_references.py --out cache/v3/references --aircraft cache/v3/plans` | per date every candidate played with crews and passengers under P, the capped repair, the reference and how it replays, and the generator checks P1 to P4 (with the plans, about 2.5 h on 192 vCPUs) |
| v3 bound | `tools/passenger_bound.py --reference cache/v3/references --out cache/v3/bound --limit 1800` | the capacity-aware lower bound under P, every played day checked at or above it |
| v3 pack | `tools/build_occ_pack.py --v3 cache/v3/references --bound cache/v3/bound` | b6-v3: b6-v2's dates (only read) with their passengers, references under P and best plans |
| v3 checks | `tools/customer_desk_check.py --out cache/v3/customer --reference cache/v3/references`; `tools/scripted_desks3.py --out cache/scripted3 --split train` | stage 2b (the premise with crews, and the scripted customer desk against two scripted desks); the env's scripted desks with the audit, the replay and the leak checks |

`tasks/b6-v2-split.json` is the published split: the seed-0 draw over the dates the crew filter kept
(`programs/network.py`, `split`), each eval date with the storm hub it was drawn for, and the train dates.
`tasks/b6-v3-split.json` is b6-v3's: b6-v2's, date for date, with the two calm controls.

## Data

The packs, the survey extracts in `../tasks/pax/` (from the DOT's Origin and Destination Survey, DB1B and DB1C), the
test fixtures and the recorded runs derive from U.S. Government works in the public domain (BTS, DOT, FAA, and NWS
products archived by the Iowa Environmental Mesonet, which asks for attribution, given here); every pack date and
fixture date copies airportsdata's airport rows, so they carry its MIT notice; the recorded runs also hold the model's
own words (requests, replies, notes), as the dataset publishes them under its card's terms
([../../../data/airportsdata/SOURCE.md](../../../data/airportsdata/SOURCE.md)). Each source's terms are in
[../../../data/](../../../data/), and the card's [Data and licence](../../../README.md#data-and-licence) section has the
whole list.
