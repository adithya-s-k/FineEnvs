# The board: GPT-6.1 Sol on the storm days and at the control center

On 14 October 2026 GPT-6.1 Sol played 5 of v1's 20 eval days once each (the sweep `sol-v1`): a first look at 3 JFK and 2
Boston days picked by hand, not a random sample, and the plugin didn't record how they were chosen. On 19 and 20 October
it played every one of v2's 24 eval dates once in each mode, 48 runs (`occ-board-sol`), all through the agentenv-airops
plugin, whose env servers, task steps, verifiers and agents now live here with the same tool outputs and rewards: every
run replays through this code byte for byte and grades to the same day. Every run, with its transcript, is in the
plugin's dataset [earakely-scale/AirOpsEnv-AgentEnv](https://huggingface.co/datasets/earakely-scale/AirOpsEnv-AgentEnv)
at v0.3.3 (published under the terms on its card, not part of this repository) and replays in 3D in the
[Space](https://huggingface.co/spaces/earakely-scale/AirOpsEnv-AgentEnv). The eval cards, `eval-v1.json` and
`eval-v2.json`, recompute from the dataset's runs and the packs.

**v1 and v2 are not comparable.** v1 is one hub's desk in its 3 to 6 storm hours with aircraft only, graded against a
proven optimum; v2 is the whole network for the whole day with crews, run by two desks, graded against the best plan
found and its bound. Neither is comparable with PortSimEnv's evals.

## v1: one desk at one hub

| Model | Provider | Days | Mean reward (95% CI) | Follower | Doing nothing | JetBlue | Valid days | Mean cost over hindsight | Cost per episode |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| GPT-6.1 Sol | Azure OpenAI | 5 | **0.419** (0.319 to 0.547) | 0.730 | 0.247 | 0.206 | 5 of 5 | +18.9% | $0.60 |

The references are on the same 5 days; over all 20 eval days the follower scores 0.729, the blind follower 0.626,
doing nothing 0.297 and JetBlue's own decisions 0.210.

| Day | Flights | Watches | GPT-6.1 Sol | Over hindsight | Follower | Blind follower | Doing nothing | JetBlue | Turns | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `b6-bos-20250216-1500` | 81 | 12 | **0.672** | +11.3% | 0.937 | 0.937 | 0.366 | 0.224 | 40 | $0.33 |
| `b6-jfk-20230714-1400` | 91 | 13 | **0.441** | +10.4% | 0.782 | 0.229 | 0.212 | 0.200 | 39 | $0.35 |
| `b6-jfk-20240804-0700` | 149 | 20 | **0.382** | +23.6% | 0.901 | 0.627 | 0.213 | 0.202 | 63 | $0.85 |
| `b6-jfk-20250709-1500` | 76 | 12 | **0.324** | +24.1% | 0.614 | 0.743 | 0.224 | 0.200 | 31 | $0.23 |
| `b6-bos-20240107-0600` | 188 | 19 | **0.276** | +24.9% | 0.417 | 0.436 | 0.220 | 0.202 | 75 | $1.23 |

- Sol beats doing nothing and JetBlue's own decisions on every day it played, and stays below the follower on every
  one. A day's every plan is valid (the env never strands a flight), so "valid days" counts the days that finished and
  passed the verifier's audit.
- JetBlue's real day scores near the reward's floor because it cancelled far more than the passenger-centric cost
  favours, with crews and information this model leaves out; it is shown, never the anchor.
- Five runs on five days picked by hand are a first look, not a board, and not an estimate over the 20: doing nothing
  scores 0.247 on them and 0.297 over all 20.

## v2: the ops control center

Fixed before the first run, as the plugin's FIDELITY.md records: the code, the pack's hash, the dates, their order, the
caps ($5 a desk in a two-desk run, $8 for one agent) and the budget.

| Mode | Dates | Mean reward (95% CI) | reward_L (95% CI) | Over the best plan, median | Avoidable cost saved, median | Valid days | Spend |
|---|---:|---|---|---:|---:|---:|---:|
| GPT-6.1 Sol, two desks, talking | 24 | **0.598** (0.545 to 0.658) | 0.523 (0.484 to 0.564) | 12.3% | 60% | 24 of 24 | $48.33 |
| GPT-6.1 Sol, one agent at both desks | 24 | **0.550** (0.497 to 0.609) | 0.487 (0.450 to 0.526) | 13.8% | 57% | 24 of 24 | $24.54 |
| Standard crew desk (B0) | 24 | 0.308 | | | 0% | | |

On the same dates the crew system alone, nobody at the desks, scores 0.305, the scripted desks 0.308, one scripted
desk at both 0.295, and JetBlue's own decisions, replayed with the generated crews, 0.209.

- **Two desks against one agent, paired by date:** +0.049 reward (95% CI 0.024 to 0.075), two desks higher on 18 of
  24 dates; their days cost 2.9% of the best plan less (CI 1.4 to 4.6%).
- **Both far above standard practice, both far from the best plan:** at the median a Sol day saves about 60% of what
  B0 leaves on the table and costs 12 to 14% more than the best plan.
- **It doesn't show that talking helps.** The two desks used about twice the compute: 2.0× the spend, 2.0× the input
  tokens and 2.5× the output tokens, more on every date; no desk came near its cap. Before any model ran, scripted desks
  measured whether coordination is worth at least 5 points of the avoidable cost on 60% of eval dates: it was on 3 of
  24, so that gate failed, and v2 shipped with the result stated. A run with the channel off would separate the two.
- A valid day is one whose audit passed, with a legal crew on every flight it flew; every run scored.

| Date | Storm hubs | Flights | Under FAA programs | JetBlue cancelled | Gap, best to bound | Two desks | One agent | Two desks, reward_L | One agent, reward_L | Spend, two desks | Spend, one agent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-12-07 | JFK, BOS | 767 | 32 | 6 | 4.7% | **0.465** | 0.408 | 0.426 | 0.385 | $1.43 | $1.22 |
| 2023-06-06 | BOS | 774 | 5 | 2 | 5.8% | **0.710** | 0.595 | 0.525 | 0.471 | $0.88 | $0.36 |
| 2023-06-16 | JFK, BOS | 753 | 31 | 47 | 5.7% | **0.576** | 0.510 | 0.518 | 0.469 | $2.51 | $1.45 |
| 2023-07-08 | JFK, BOS | 693 | 27 | 26 | 4.7% | **0.594** | 0.495 | 0.556 | 0.472 | $3.52 | $1.01 |
| 2023-07-09 | JFK, BOS | 733 | 87 | 211 | 4.4% | 0.471 | **0.482** | 0.453 | 0.463 | $2.71 | $1.54 |
| 2023-07-13 | JFK, BOS | 750 | 12 | 15 | 13.8% | **1.000** \* | 0.967 | 0.578 | 0.537 | $1.38 | $1.06 |
| 2023-07-29 | JFK, BOS | 698 | 73 | 70 | 5.8% | **0.569** | 0.548 | 0.520 | 0.504 | $3.01 | $1.77 |
| 2023-12-10 | JFK, BOS | 665 | 44 | 0 | 4.7% | **0.754** | 0.579 | 0.676 | 0.537 | $2.07 | $1.02 |
| 2023-12-18 | BOS | 752 | 48 | 41 | 5.2% | **0.412** | 0.356 | 0.397 | 0.350 | $3.17 | $1.68 |
| 2024-06-23 | JFK, BOS | 662 | 112 | 57 | 4.6% | **0.477** | 0.402 | 0.445 | 0.386 | $2.75 | $1.20 |
| 2024-07-13 | BOS | 612 | 8 | 10 | 12.2% | **0.806** | 0.657 | 0.699 | 0.589 | $1.84 | $0.99 |
| 2024-07-15 | JFK, BOS | 664 | 15 | 12 | 8.3% | **0.484** | 0.462 | 0.410 | 0.399 | $1.76 | $0.99 |
| 2024-07-22 | JFK | 668 | 33 | 21 | 8.5% | **0.545** | 0.438 | 0.462 | 0.397 | $1.94 | $0.99 |
| 2025-01-11 | BOS | 494 | 20 | 6 | 5.4% | 0.805 | **0.819** | 0.740 | 0.751 | $1.23 | $0.56 |
| 2025-07-08 | JFK, BOS | 637 | 71 | 43 | 4.4% | 0.601 | **0.652** | 0.550 | 0.589 | $2.16 | $1.09 |
| 2025-07-12 | JFK | 647 | 17 | 23 | 4.1% | 0.528 | **0.542** | 0.479 | 0.489 | $1.59 | $0.77 |
| 2025-12-02 | BOS | 619 | 18 | 0 | 5.7% | **0.466** | 0.345 | 0.398 | 0.332 | $1.10 | $0.56 |
| 2025-12-14 | JFK | 667 | 41 | 89 | 3.2% | **0.530** | 0.427 | 0.509 | 0.417 | $3.47 | $1.12 |
| 2025-12-19 | JFK, BOS | 708 | 80 | 51 | 13.9% | 0.482 | **0.526** | 0.393 | 0.411 | $2.40 | $1.01 |
| 2025-12-30 | JFK | 689 | 11 | 10 | 7.6% | **0.799** | 0.718 | 0.677 | 0.621 | $1.24 | $0.97 |
| 2026-06-14 | JFK, BOS | 722 | 49 | 51 | 3.1% | **0.608** | 0.583 | 0.565 | 0.545 | $2.06 | $1.00 |
| 2026-07-01 | BOS | 705 | 36 | 8 | 2.4% | **0.668** | 0.597 | 0.636 | 0.573 | $1.15 | $0.84 |
| 2026-07-09 | JFK, BOS | 764 | 85 | 57 | 2.8% | 0.464 | **0.545** | 0.444 | 0.512 | $2.16 | $0.86 |
| 2026-07-15 | JFK | 739 | 16 | 10 | 5.0% | **0.544** | 0.538 | 0.491 | 0.486 | $0.81 | $0.47 |

The higher of the two modes in bold. B0 scores 0.308 on every date. "Under FAA programs": JetBlue flights the programs
at JFK and Boston controlled or held. \* Under the best plan by 1.5%, so 1.0 and flagged; the bound is 13.8% lower, and
reward_L, 0.578, is the fairer reading. 2026-07-09, two desks: the registered retry, which replaces a first attempt
whose crew desk lost its connection at 17:00 and was scored as if it had stood down; the harness now records that as
an error, and the dataset keeps both attempts, the first flagged as superseded.

## How it was run

- **Harness:** v1's planner [`airops-llm`](../envs/airops/agentenv/agents/airops-llm/) with 3 planning calls a
  watch and 5 turns a watch plus 4, 32,000 output tokens a reply, an episode cap of $3; v2's desks
  [`airops-desk`](../envs/airops/agentenv/agents/airops-desk/), each its own A2A agent in its role, reasoning effort
  `high`, 6 model turns a watch plus 10 for a desk of a two-desk run and 8 plus 10 for one agent, 32,000 output tokens
  a reply, capped at $5 a desk in a two-desk run and $8 for one agent. One run per date and mode, on local Docker.
- **Route:** GPT-6.1 Sol on the Responses API, served by Azure OpenAI, through a LiteLLM proxy. Spend is the harness's
  tokens times the prices pinned in the agent.
- **Spend:** v1 $2.99. v2's board $73.94 with the superseded attempt; the 6-run trial sweep before it $6.88.
- **Checks:** every scored run replays exactly through the env's calls and passes the audit, and an independent
  recheck recomputed every reward and reward_L from the graded costs and the pack's best plan, bound and B0.

## Caveats

- One model, one run per date and mode: the intervals, a bootstrap over the dates, are wide, and five v1 days are
  not a board.
- The crews are generated and the crew filter leans the eval dates toward moderate storms; the plugin's
  [FIDELITY.md](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.3.3/FIDELITY.md) sets out every
  layer's checks and misses.
