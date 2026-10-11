# PortSimEnv live · AgentEnv

One week at one Port of Barcelona quay, played live in watches, served as three
[AgentEnv](https://github.com/scaleapi/agentenv-framework) environments from one MCP server on port 18765:

| env | weeks | what it adds |
|---|---|---|
| `portsim-live` | the 15 qualifying one-week dock-v1-eval weeks (any one-week dock-v1 week loads) | the news arrives watch by watch |
| `portsim-marine` | the 15 qualifying dock-v1-marine weeks | the port's pilots and tugs, shared with its other 2024 traffic |
| `portsim-wind` | the 15 qualifying wind weeks, built locally | a real weather week, with the forecast at every watch |

`ENVIRONMENT_NAME` picks the env (`portsim-live` when unset). The week model and the grade are `../core`
(`port_week_core`); this folder is the server (`port_week_agentenv`), its task steps, `portsim-live-verifier`
(`verifier/verify.py`) and `rollout.py`. The agent is portsim-v1's `portsim-llm`
(`../../../../portsim-v1/envs/berth_planning/agentenv/agent`).

## Tools

| tool | does |
|---|---|
| `get_situation()` | the hour, the week as known now, the confirmed windows, the ships without one, new messages |
| `check_plan(plan)` | checks windows over the confirmed ones without confirming; one of the watch's 3 planning calls |
| `confirm_berths(plan)` | confirms windows; ships left out keep theirs; one of the watch's 3 planning calls |
| `advance()` | ends the watch; after the last one the rest of the week runs and is graded |
| `port_notice(event_id, name, text)` | delivers one party's notice; hidden from the agent, called by the gateway's triggers |

Extensions: `urn:portsim:live-load/v1` loads a week (one per deploy), `urn:portsim:end-week/v1` runs the rest of the
week and grades it, `urn:agentenv:clock/v1` follows the gateway's virtual clock. `data/get` reports the week and never
its reference plans.

## Watches, triggers and the clock

A week's news is revealed a fixed lead before the hour it is about and grouped into 6-hour bulletins, the watches. The
task registers one gateway trigger per watch after the first: when `advance` answers that watch, the trigger calls
`port_notice` once per notice before the agent's next call, and the audit in `data/get` checks it did.

The task syncs the gateway clock frozen at the week's start. Each `advance` sets it to the next bulletin's time, so
time passes only when the agent advances. The verifier scores the week's reward and refuses a week that never finished
or failed the audit. [DESIGN.md](../../../DESIGN.md) has the rules and the grade of each env.

## Run

```bash
uv run python rollout.py --smoke                                         # no model or key: wiring-noplay scores 0
uv run python rollout.py --smoke --env marine
HF_TOKEN=hf_... uv run python rollout.py                                 # GLM-5.3-Flash on the HF router, 1 week
uv run python rollout.py --env marine --model anthropic/claude-sonnet-5-5 --tasks 3   # through LITELLM_* or [model]
uv run python rollout.py --env wind --task-id dock-36A-w06x1-standard-0-e07           # after the wind build
```

`rollout.py` stages a bundle in `runs/bundle` and runs it with agent-env in its own process. It needs Docker; agent-env
builds the env and agent images and the gateway on first use, starts a registry on `127.0.0.1:5000` and keeps its
stores under `~/.local/state/agent-env`. Each run writes `runs/<UTC>/results.jsonl` and
`runs/<UTC>/transcripts/<task>.json`. Flags: `--env live|marine|wind`, `--tasks N` (qualifying weeks, fewest watches
first), `--task-id ID` (repeatable), `--episode-cap-usd` (5), `--hf-bill-to ORG`, `--keep`. Every week gets 47 turns
(52 on wind), so the turn budget doesn't reveal how many watches are coming.

## The wind weeks

The wind weeks are derived from Meteocat readings and are not committed
([../../../data/meteocat/SOURCE.md](../../../data/meteocat/SOURCE.md)). Build them on your machine, from `../`:

```bash
uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/build_wind.py
```

Until then `--env wind` stops and points to `tools/build_wind.py`, and the server exits the same way when
`ENVIRONMENT_NAME=portsim-wind`.

## Tests

```bash
uv run --frozen --extra dev pytest tests -q                              # offline
uv run --frozen --project . --extra dev pytest ../core/tests -q          # the week model, through this project
```

`tests/fixtures/runs` holds six recorded weeks (two live weeks played by Sonnet and GPT, scripted marine and wind runs,
the wind one on synthetic weather); `test_recorded_runs.py` replays every recorded tool call through the env and
compares each output byte for byte, and the graded week with the recording. The cards, the task steps and the tool
outputs equal those of the agentenv-portsim plugin at v0.4.5.

## Docker

```bash
cd ../../../..   # 07-simulation-environments
docker build -f portsim-live/envs/port_week/agentenv/Dockerfile -t portsim-live-agentenv .
PORTSIM_LIVE_IMAGE=portsim-live-agentenv uv run --frozen --project portsim-live/envs/port_week/agentenv --extra dev \
    pytest portsim-live/envs/port_week/agentenv/tests -q    # the server tests, one container of the image per week
docker run -d --name portsim-marine -p 127.0.0.1:18765:18765 -e ENVIRONMENT_NAME=portsim-marine portsim-live-agentenv
```

The image never holds the wind weeks. To serve `portsim-wind` from it, mount them read-only:

```bash
docker run -d --name portsim-wind -p 127.0.0.1:18766:18765 -e ENVIRONMENT_NAME=portsim-wind \
    -v "$PWD/portsim-live/envs/port_week/tasks/dock-v1-wind:/app/port_week/tasks/dock-v1-wind:ro" portsim-live-agentenv
```

`rollout.py --env wind` stages a build context with them, so the image it builds does hold them: keep it on your
machine.

Contributed from the agentenv-portsim plugin (https://github.com/earakely-scale/agentenv-portsim-plugin) by Edgar
Arakelyan (Scale AI), built on AgentEnv (https://github.com/scaleapi/agentenv-framework).

Data: Port de Barcelona open data, CC BY-SA 4.0; the forecasts contain modified ECMWF open data, CC BY 4.0
([../../../data/ecmwf/SOURCE.md](../../../data/ecmwf/SOURCE.md)); the wind weeks are built locally under Meteocat's
terms, which this repository doesn't relicense ([../../../data/meteocat/SOURCE.md](../../../data/meteocat/SOURCE.md)).
