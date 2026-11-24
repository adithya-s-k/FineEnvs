# The model board: twelve models on the live, marine, wind and lines weeks

Between 7 and 9 October 2026 twelve models played every live, marine and wind week once: 15 weeks each, 540 runs,
through the agentenv-portsim plugin at
[v0.4.5](https://github.com/earakely-scale/agentenv-portsim-plugin/tree/v0.4.5), whose env servers, task steps,
verifier and agent now live here, with the same tool outputs, cards and rewards. Every run, with its transcript, is in
the plugin's dataset
[earakely-scale/PortSimEnv-AgentEnv](https://huggingface.co/datasets/earakely-scale/PortSimEnv-AgentEnv) (published
under the terms on its card, not part of this repository) and replays in 3D in the
[Space](https://huggingface.co/spaces/earakely-scale/PortSimEnv-AgentEnv). The plugin's sweep reports, which count
every attempt and go week by week, are
[live.md](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.4.5/results/live.md),
[marine.md](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.4.5/results/marine.md) and
[wind.md](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.4.5/results/wind.md); its
[board](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.4.5/results/board.md) adds the rules broken,
how often a model heeds its own checks, and the spend.

On 12 and 13 October 2026 the same twelve played every lines week (v5) once, 180 runs, through the plugin at
[v0.5.3](https://github.com/earakely-scale/agentenv-portsim-plugin/tree/v0.5.3), whose lines env, ships' agent and
`message_agent` now live here too; the plugin's sweep report is
[lines.md](https://github.com/earakely-scale/agentenv-portsim-plugin/blob/v0.5.3/results/lines.md).

**Not comparable with PortSimEnv v1's eval** ([../../portsim-v1/](../../portsim-v1/)). The agent re-plans the week as
it unfolds, with four tools, at most 3 planning calls a watch and 47 turns a week (52 on the wind weeks, 72 with
`message_agent` on the lines weeks), where v1 plans the whole week in one go with 12 turns, 10 checks and one
`submit_plan`. Each env plays 15 one-week standard and busy eval weeks on which a rolling re-planner reaches the
optimum, where v1's eval plays 50 weeks across four tiers.
Rankings don't carry over: on v1 GLM-5.3-Flash scores above GLM-5.3, here below.

## Summary

Mean reward on each env and over all 60 weeks (each week weighs the same), best first:

| Model | Provider | live | marine | wind | lines | All 60 weeks | Feasible weeks |
|---|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | Anthropic | 0.976 | 0.951 | 0.906 | 0.593 | **0.857** | 59 of 60 |
| GPT-6 Astra | Azure OpenAI | 0.984 | 0.956 | 0.896 | 0.568 | **0.851** | 58 of 60 |
| GPT-6.1 Sol | Azure OpenAI | 0.957 | 0.905 | 0.745 | 0.536 | **0.786** | 55 of 60 |
| Claude Sonnet 5.5 | Anthropic | 0.904 | 0.932 | 0.888 | 0.387 | **0.778** | 60 of 60 |
| Claude Haiku 5.5 | Anthropic | 0.694 | 0.639 | 0.583 | 0.258 | **0.543** | 57 of 60 |
| Qwen3.8-2.4T | Fireworks | 0.686 | 0.519 | 0.662 | 0.291 | **0.539** | 36 of 60 |
| DeepSeek V4.1 Flash | Fireworks | 0.754 | 0.598 | 0.480 | 0.168 | **0.500** | 34 of 60 |
| GLM-5.3 | Fireworks | 0.748 | 0.593 | 0.468 | 0.153 | **0.490** | 31 of 60 |
| GLM-5.3-Flash | Fireworks | 0.470 | 0.461 | 0.328 | 0.261 | **0.380** | 24 of 60 |
| Kimi K3 | Fireworks | 0.465 | 0.446 | 0.249 | 0.231 | **0.348** | 23 of 60 |
| GPT-6 Luna | Azure OpenAI | 0.335 | 0.287 | 0.320 | 0.234 | **0.294** | 49 of 60 |
| Qwen3.8-27B | Groq | 0.055 | 0.073 | 0.090 | 0.143 | **0.090** | 1 of 60 |

- Claude Opus 5.5 and GPT-6 Astra share the top, their intervals overlapping on every env. Claude Sonnet 5.5 is level
  with them on the marine and wind weeks and below them on the live and lines ones. Opus and Sonnet broke no rule in
  any of their live, marine and wind weeks, Astra in one.
- The three best open models on live, marine and wind, Qwen3.8-2.4T, DeepSeek V4.1 Flash and GLM-5.3, average 0.603
  to 0.622 there, within noise of Claude Haiku 5.5's 0.639. Haiku keeps 44 of those 45 weeks feasible; each of them
  keeps 28.
- Eight of the twelve models score lower at each step from live to marine to wind; GPT-6.1 Sol falls the most, from
  0.957 to 0.745. The envs' weeks differ (the wind weeks reuse 11 of the marine weeks' schedules), so this is a pattern
  across the sets, not a paired comparison. The lines weeks are the wind weeks with the ships' agents, and 11 of the
  12 models score lower on them.
- GPT-6 Luna's plans are mostly valid (49 of 60 weeks) but cost several times the optimum, hence 0.294.
- Qwen3.8-27B, the one model small enough to fine-tune cheaply, found a valid plan in 1 of 60 weeks.

## live

On the same weeks the rolling re-planner scores 1.000 and the naive policy 0.202.

| Model | Provider | Weeks | Mean reward (95% CI) | Standard | Busy | Reached the end | Feasible | Optimal | Median turns | Cost per episode |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra | Azure OpenAI | 15 of 15 | **0.984** (0.965 to 0.998) | 0.981 | 0.986 | 15 | 15 | 11 | 15 | $0.40 |
| Claude Opus 5.5 | Anthropic | 15 of 15 | **0.976** (0.954 to 0.993) | 0.981 | 0.971 | 15 | 15 | 10 | 10 | $0.43 |
| GPT-6.1 Sol | Azure OpenAI | 15 of 15 | **0.957** (0.916 to 0.991) | 0.963 | 0.952 | 15 | 15 | 10 | 13 | $0.06 |
| Claude Sonnet 5.5 | Anthropic | 15 of 15 | **0.904** (0.854 to 0.945) | 0.901 | 0.906 | 15 | 15 | 2 | 8 | $0.32 |
| DeepSeek V4.1 Flash | Fireworks | 15 of 15 | **0.754** (0.573 to 0.913) | 0.895 | 0.631 | 14 | 12 | 3 | 16 | $0.14 |
| GLM-5.3 | Fireworks | 15 of 15 | **0.748** (0.563 to 0.920) | 0.856 | 0.653 | 14 | 11 | 5 | 18 | $0.44 |
| Claude Haiku 5.5 | Anthropic | 15 of 15 | **0.694** (0.550 to 0.821) | 0.652 | 0.730 | 15 | 15 | 0 | 10 | $0.03 |
| Qwen3.8-2.4T | Fireworks | 15 of 15 | **0.686** (0.484 to 0.856) | 0.603 | 0.759 | 14 | 11 | 5 | 18 | $0.83 |
| GLM-5.3-Flash | Fireworks | 15 of 15 | **0.470** (0.307 to 0.648) | 0.518 | 0.428 | 15 | 7 | 1 | 14 | $0.04 |
| Kimi K3 | Fireworks | 15 of 15 | **0.465** (0.290 to 0.654) | 0.643 | 0.308 | 15 | 7 | 2 | 14 | $1.03 |
| GPT-6 Luna | Azure OpenAI | 15 of 15 | **0.335** (0.231 to 0.466) | 0.231 | 0.426 | 15 | 12 | 1 | 18 | $0.02 |
| Qwen3.8-27B | Groq | 15 of 15 | **0.055** (0.037 to 0.077) | 0.067 | 0.045 | 15 | 0 | 0 | 21 | $0.49 |

## marine

On the same weeks the rolling re-planner scores 1.000 and the naive policy, which knows nothing of pilots and tugs,
0.187.

| Model | Provider | Weeks | Mean reward (95% CI) | Standard | Busy | Reached the end | Feasible | Optimal | Median turns | Cost per episode |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra | Azure OpenAI | 15 of 15 | **0.956** (0.910 to 0.993) | 0.970 | 0.943 | 15 | 15 | 10 | 17 | $0.53 |
| Claude Opus 5.5 | Anthropic | 15 of 15 | **0.951** (0.914 to 0.983) | 0.944 | 0.957 | 15 | 15 | 8 | 11 | $0.56 |
| Claude Sonnet 5.5 | Anthropic | 15 of 15 | **0.932** (0.892 to 0.967) | 0.914 | 0.949 | 15 | 15 | 5 | 11 | $0.45 |
| GPT-6.1 Sol | Azure OpenAI | 15 of 15 | **0.905** (0.790 to 0.977) | 0.950 | 0.865 | 15 | 14 | 7 | 14 | $0.08 |
| Claude Haiku 5.5 | Anthropic | 15 of 15 | **0.639** (0.507 to 0.765) | 0.610 | 0.664 | 15 | 15 | 0 | 12 | $0.04 |
| DeepSeek V4.1 Flash | Fireworks | 15 of 15 | **0.598** (0.392 to 0.786) | 0.678 | 0.528 | 12 | 9 | 2 | 19 | $0.24 |
| GLM-5.3 | Fireworks | 15 of 15 | **0.593** (0.368 to 0.804) | 0.561 | 0.621 | 11 | 10 | 3 | 18 | $0.47 |
| Qwen3.8-2.4T | Fireworks | 15 of 15 | **0.519** (0.296 to 0.746) | 0.364 | 0.655 | 12 | 7 | 5 | 15 | $0.83 |
| GLM-5.3-Flash | Fireworks | 15 of 15 | **0.461** (0.299 to 0.652) | 0.589 | 0.349 | 15 | 6 | 3 | 14 | $0.07 |
| Kimi K3 | Fireworks | 15 of 15 | **0.446** (0.256 to 0.636) | 0.568 | 0.339 | 15 | 6 | 0 | 16 | $1.28 |
| GPT-6 Luna | Azure OpenAI | 15 of 15 | **0.287** (0.199 to 0.407) | 0.190 | 0.372 | 15 | 11 | 0 | 19 | $0.02 |
| Qwen3.8-27B | Groq | 15 of 15 | **0.073** (0.047 to 0.100) | 0.070 | 0.076 | 15 | 0 | 0 | 24 | $0.81 |

## wind

On the same weeks the forecast-following re-planner scores 1.000, the forecast-blind one 0.237 and the naive policy
0.183. A wind week scores 1.0 at its anchor ([DESIGN.md](../DESIGN.md#the-anchor)), so the table counts weeks at the
anchor rather than at the optimum.

| Model | Provider | Weeks | Mean reward (95% CI) | Standard | Busy | Reached the end | Feasible | At the anchor | Median turns | Cost per episode |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | Anthropic | 15 of 15 | **0.906** (0.845 to 0.961) | 0.926 | 0.902 | 15 | 15 | 5 | 17 | $0.94 |
| GPT-6 Astra | Azure OpenAI | 15 of 15 | **0.896** (0.780 to 0.979) | 0.923 | 0.889 | 15 | 14 | 9 | 25 | $0.95 |
| Claude Sonnet 5.5 | Anthropic | 15 of 15 | **0.888** (0.807 to 0.953) | 0.802 | 0.909 | 15 | 15 | 3 | 17 | $0.80 |
| GPT-6.1 Sol | Azure OpenAI | 15 of 15 | **0.745** (0.576 to 0.890) | 0.869 | 0.714 | 15 | 12 | 4 | 23 | $0.15 |
| Qwen3.8-2.4T | Fireworks | 15 of 15 | **0.662** (0.460 to 0.837) | 0.926 | 0.596 | 14 | 10 | 5 | 30 | $1.30 |
| Claude Haiku 5.5 | Anthropic | 15 of 15 | **0.583** (0.460 to 0.708) | 0.373 | 0.636 | 15 | 14 | 1 | 18 | $0.05 |
| DeepSeek V4.1 Flash | Fireworks | 15 of 15 | **0.480** (0.256 to 0.701) | 0.903 | 0.374 | 10 | 7 | 3 | 22 | $0.24 |
| GLM-5.3 | Fireworks | 15 of 15 | **0.468** (0.277 to 0.661) | 0.457 | 0.471 | 13 | 7 | 3 | 29 | $1.02 |
| GLM-5.3-Flash | Fireworks | 15 of 15 | **0.328** (0.209 to 0.460) | 0.309 | 0.332 | 15 | 5 | 0 | 21 | $0.09 |
| GPT-6 Luna | Azure OpenAI | 15 of 15 | **0.320** (0.220 to 0.455) | 0.261 | 0.335 | 15 | 13 | 0 | 32 | $0.03 |
| Kimi K3 | Fireworks | 15 of 15 | **0.249** (0.159 to 0.374) | 0.457 | 0.197 | 15 | 5 | 0 | 23 | $1.34 |
| Qwen3.8-27B | Groq | 15 of 15 | **0.090** (0.064 to 0.119) | 0.139 | 0.078 | 13 | 0 | 0 | 33 | $1.38 |

## lines

The anchor is the lower of hindsight's cost and the follower's best: hindsight on 13 weeks, the follower on 2. On the
same weeks the follower, which re-plans at every watch and requests arrival times through the ships' agents, scores
0.548 in its first solver configuration, the follower that never asks 0.242 and the naive policy 0.184.

| Model | Provider | Weeks | Mean reward (95% CI) | Standard | Busy | Reached the end | Feasible | At the anchor | Median turns | Cost per episode |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | Anthropic | 15 of 15 | **0.593** (0.453 to 0.725) | 0.606 | 0.590 | 15 | 14 | 1 | 33 | $2.42 |
| GPT-6 Astra | Azure OpenAI | 15 of 15 | **0.568** (0.420 to 0.722) | 0.529 | 0.578 | 15 | 14 | 3 | 46 | $2.16 |
| GPT-6.1 Sol | Azure OpenAI | 15 of 15 | **0.536** (0.392 to 0.693) | 0.469 | 0.552 | 15 | 14 | 3 | 39 | $0.32 |
| Claude Sonnet 5.5 | Anthropic | 15 of 15 | **0.387** (0.269 to 0.517) | 0.354 | 0.395 | 15 | 15 | 0 | 27 | $1.55 |
| Qwen3.8-2.4T | Fireworks | 15 of 15 | **0.291** (0.171 to 0.423) | 0.607 | 0.212 | 11 | 8 | 0 | 30 | $2.26 |
| GLM-5.3-Flash | Fireworks | 15 of 15 | **0.261** (0.171 to 0.367) | 0.517 | 0.197 | 11 | 6 | 0 | 31 | $0.18 |
| Claude Haiku 5.5 | Anthropic | 15 of 15 | **0.258** (0.203 to 0.322) | 0.375 | 0.228 | 15 | 13 | 0 | 22 | $0.10 |
| GPT-6 Luna | Azure OpenAI | 15 of 15 | **0.234** (0.199 to 0.284) | 0.312 | 0.215 | 15 | 13 | 0 | 34 | $0.06 |
| Kimi K3 | Fireworks | 15 of 15 | **0.231** (0.172 to 0.308) | 0.167 | 0.247 | 15 | 5 | 0 | 30 | $2.02 |
| DeepSeek V4.1 Flash | Fireworks | 15 of 15 | **0.168** (0.089 to 0.269) | 0.373 | 0.116 | 6 | 6 | 0 | 16 | $0.30 |
| GLM-5.3 | Fireworks | 15 of 15 | **0.153** (0.029 to 0.307) | 0.266 | 0.124 | 5 | 3 | 0 | 2 | $0.75 |
| Qwen3.8-27B | Groq | 15 of 15 | **0.143** (0.118 to 0.164) | 0.151 | 0.141 | 15 | 1 | 0 | 39 | $1.63 |

- Claude Opus 5.5, GPT-6 Astra and GPT-6.1 Sol lead within noise of each other: they agree an arrival for 86% to 90%
  of a week's ships and keep 14 of 15 weeks feasible, and Opus and Astra score above the follower.
- The port berths a ship that misses its window at the first free stretch of quay, so a model that rarely asks stays
  feasible and pays in hours at anchor: Claude Sonnet 5.5 agrees 40% of a week's ships and keeps all 15 weeks
  feasible; Claude Haiku 5.5 and GPT-6 Luna agree 12% and 18% and score about what the silent follower does.
- Qwen3.8-2.4T, GLM-5.3-Flash and Kimi K3 agree about 43% and keep 5 to 8 weeks feasible. GLM-5.3 and DeepSeek V4.1
  Flash end 10 and 9 of their weeks by reasoning past the 32,000-token reply limit twice in a row, which ends the
  episode; the week is scored on the windows confirmed, none in all 10 of GLM-5.3's.

## How it was run

- **Harness:** [`portsim-llm`](../../portsim-v1/envs/berth_planning/agentenv/agent/), with the same prompts, tools,
  nudges and limits for every model: 47 turns a week (52 on the wind weeks, 72 on the lines weeks), at most 3
  planning calls a watch, 32,000 output tokens a reply (16,384 for Qwen3.8-27B, Groq's limit), the provider's default
  reasoning (effort `medium` on the Responses API), one run per model and week, on local Docker.
- **Routes:** all through a LiteLLM proxy. Claude on Anthropic's API; GPT on the Responses API, served by Azure OpenAI;
  Kimi K3, GLM-5.3, GLM-5.3-Flash, Qwen3.8-2.4T and DeepSeek V4.1 Flash on Fireworks; Qwen3.8-27B on Groq. Four of the
  open models are those of v1's eval, which ran them on other providers through the Hugging Face router.
- **Cost:** each episode stops before a request that could take it past its cap, set per model from its price, from
  $1.50 (GPT-6.1 Sol) to $12 (Kimi K3). Cost per episode is the model spend at the prices pinned in the agent.
- **Failures:** an attempt that failed outside the episode (a deploy, a provider error, a lost agent) was played again
  from the start, up to three attempts in all; the plugin's board lists the few weeks played again in a later sweep.
  On the lines weeks a turn that would start within 20 minutes of the episode's 2-hour limit isn't started and the
  week is graded as played; no run reached it.

## Caveats

- One run per model and week: the intervals, a bootstrap over the weeks, are wide, and neither the order of the top
  three nor that of the open models is settled.
- The open models ran on Fireworks and Groq, not on the providers v1's eval used; the same weights can score
  differently on another provider.
- The weeks are simulated pressure on real data, and the wind weeks move real weather onto 2024 schedules with ECMWF's
  forecast standing in for the port's own; the dataset card's
  [Limitations](https://huggingface.co/datasets/earakely-scale/PortSimEnv-AgentEnv#limitations) sets these out.
