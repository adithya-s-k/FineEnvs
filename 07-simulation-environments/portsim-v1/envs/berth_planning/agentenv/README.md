# PortSimEnv v1 · AgentEnv

The same env as `../openenv`, served as an [AgentEnv](https://github.com/scaleapi/agentenv-framework) environment:
the same three tools and their text, the same limits (10 checks, 24 tool calls), the same reward and the dock-v1
packs (1,050 train, 50 eval). The env is an MCP server on port 18765 with two extensions, `urn:portsim:load-task/v1`
and `urn:portsim:submit-plan/v1`; `data/get` reports the episode and never the task's reference plans. The
`portsim-llm` agent (`agent/`) plays an episode, and `portsim-verifier` (`verifier/verify.py`) scores it with the
reward the env gave the submitted plan.

## Run

```bash
uv run python rollout.py --smoke                                         # no model or key
HF_TOKEN=hf_... uv run python rollout.py                                 # GLM-5.3-Flash on the HF router, 1 task
uv run python rollout.py --model anthropic/claude-sonnet-5-5 --tasks 3   # through LITELLM_* or [model]
uv run python rollout.py --model hf:Qwen/Qwen3.8-27B:ovhcloud            # ../openenv's model spellings work too
```

`rollout.py` stages a bundle in `runs/bundle` and runs it with agent-env in its own process. It needs Docker; agent-env
builds the env and agent images on first use, starts a registry on `127.0.0.1:5000` and keeps its stores under
`~/.local/state/agent-env`. Each run writes `runs/<UTC>/results.jsonl` and `runs/<UTC>/transcripts/<task>.json`.
Flags: `--tasks N`, `--task-id ID` (repeatable), `--split eval|train`, `--max-turns` (12), `--episode-cap-usd` (5),
`--hf-bill-to ORG`, `--keep`.

## Tools

| tool | returns |
|---|---|
| `get_situation()` | quay, notices, ships already alongside, closed sections, ships to berth |
| `check_plan(plan)` | violations per ship, departure/delay/moved/cost per ship, the plan's cost; 10 per episode |
| `submit_plan(plan)` | ends the episode; the plan is graded once |

The 24th tool call, failed ones included, ends the episode with reward 0.

## Reward

`berth_core.grade`, as in `../openenv`: see [DESIGN.md](../../../DESIGN.md#the-reward).

## Smoke test

```bash
uv run --frozen --extra dev pytest tests -q                  # offline
uv run --frozen --extra dev pytest tests -q -m network       # the published rollouts, from the Hugging Face Hub
uv run python rollout.py --smoke                             # through AgentEnv: scores 1.0, 0.176471 and 0
```

## Docker

```bash
cd ..   # envs/berth_planning
docker build -f agentenv/Dockerfile -t portsim-v1-agentenv .
docker run -d --name portsim-v1 -p 127.0.0.1:18765:18765 portsim-v1-agentenv
PORTSIM_URL=http://127.0.0.1:18765 uv run --frozen --project agentenv --extra dev \
    pytest agentenv/tests/golden agentenv/tests/replay -m 'network or not network'   # against the container
docker rm -f portsim-v1
```

## Parity with the OpenEnv results

The 830 tool calls of the 300 published dock-eval50 episodes replay byte for byte (one pinned exception), the 202
submitted plans regrade to their published reward and grade, and every stored optimal plan scores 1.0
(`tests/replay`). `portsim-llm` sends the requests `../openenv`'s harness sends on five published episodes
(`tests/golden/harness.json`). Played through it on ten of the eval tasks, GPT-6.1 Sol scored 0.907 against its
published 0.872 on the same tasks and Claude Sonnet 5.5 0.778 against 0.839, one run each, both within noise
([parity.md](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.4.5/results/parity.md)).

## Agent

`portsim-llm` is `../openenv`'s harness loop as an A2A agent, calling the model through agent-env's model endpoint:
Anthropic models on the Messages API, OpenAI models on the Responses API, the rest on chat completions. It plays only
the models in its `PRICES`, and when `PORTSIM_MAX_COST_USD` is set it stops before a request that could take the
episode past it; its summary also gives the cost without the failed requests (`priced_cost_usd`), which may not have
been billed, and how many failed. Peered with other agents (`urn:agentenv:peer-agents/v1`), it gives the model a
`message_agent` tool that sends one of them structured requests.

`line_agent/` is `portsim-line`, the ships' agent of PortSimEnv live's lines weeks
([../../../../portsim-live](../../../../portsim-live/)): one per firm, answering the planner's messages from the env
through `ship_status` and `agree_arrival`, with no model.

Contributed from the agentenv-portsim plugin (https://github.com/earakely-scale/agentenv-portsim-plugin) by Edgar
Arakelyan (Scale AI), built on AgentEnv (https://github.com/scaleapi/agentenv-framework).

Data: Port de Barcelona open data, CC BY-SA 4.0.
