# The board: GPT-6.1 Sol on the storm days and at the control center, six models with its passengers

On 14 October 2026 GPT-6.1 Sol played 5 of v1's 20 eval days once each (the sweep `sol-v1`): a first look at 3 JFK and 2
Boston days picked by hand, not a random sample, and the plugin didn't record how they were chosen. On 19 and 20 October
it played every one of v2's 24 eval dates once in each mode, 48 runs (`occ-board-sol`), and on 21 October every one of
v3's 24 eval dates once in each mode, 48 runs (`occ3-board-sol`). On 22 October five more models played v3's 24 eval
dates with three desks, once a date each, 120 runs (`occ3-board-astra`, `occ3-board-opus`, `occ3-board-sonnet`,
`occ3-board-terra` and `occ3-board-kimi`). All ran through the agentenv-airops plugin, whose env servers, task steps,
verifiers and agents now live here with the same tool outputs and rewards: every call of every v1, v2 and Sol v3 run
replays through this code exactly and grades to the same day; on Linux, as the boards ran, the whole record matches too
(on macOS two crews' end-of-day shares can differ on one v2 and one v3 run,
[DESIGN.md](../DESIGN.md#determinism-and-replay)). The five other models' 120 runs replay through this code the same way
on Linux, every record matching in full. Every run, with its transcript, is in the plugin's dataset
[earakely-scale/AirOpsEnv-AgentEnv](https://huggingface.co/datasets/earakely-scale/AirOpsEnv-AgentEnv) at v0.5.0, each
file as first published (v1's and v2's at v0.3.3, Sol's v3 runs at v0.4.2; published under the terms on its card, not
part of this repository), and replays in 3D in the
[Space](https://huggingface.co/spaces/earakely-scale/AirOpsEnv-AgentEnv). The eval cards, `eval-v1.json`, `eval-v2.json`
and `eval-v3.json`, recompute from the dataset's runs and the packs.

**v1, v2 and v3 are not comparable.** v1 is one hub's desk in its 3 to 6 storm hours with aircraft only, graded against
a proven optimum; v2 is the whole network for the whole day with crews, run by two desks, graded against the best plan
found and its bound; v3 is v2's day with its passengers on board and a third desk, graded under another cost, the
passenger cost P, against the best plan found under P. None is comparable with PortSimEnv's evals.

## v3: passengers and the customer desk

Six models ran JetBlue's control center with three desks talking (the crew scheduler, the customer desk and the ops
controller, each its own agent) on the same 24 eval dates, once a date each, at reasoning effort `high`, on the same
env, pack and grading: GPT-6.1 Sol on its board of 21 October, which also played every date with one agent at all three
desks ([below](#gpt-61-sol-three-desks-against-one-agent)), and on 22 October GPT-6 Astra, Claude Opus 5.5, Claude
Sonnet 5.5, GPT-5.6 Terra and Kimi K3.

| | Model | Mean reward (95% CI) | reward_L (95% CI) | Over the best plan, median | Avoidable cost saved, median | Under the best plan | Valid days | Spend | Cap a desk | Median day |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GPT-6 Astra | 0.742 (0.673 to 0.813) | 0.377 (0.356 to 0.406) | 6.0% | 75% | 6 | 24 of 24 | $544.24 | $23.50 | 58 min, $21.33 |
| 2 | GPT-6.1 Sol | 0.721 (0.633 to 0.811) | 0.372 (0.351 to 0.398) | 7.7% | 71% | 7 | 24 of 24 | $71.38 | $3 | 70 min, $2.79 |
| 3 | Claude Opus 5.5 | 0.626 (0.544 to 0.711) | 0.357 (0.340 to 0.376) | 10.4% | 60% | 3 | 24 of 24 | $261.46 | $9.50 | 43 min, $10.40 |
| 4 | Claude Sonnet 5.5 | 0.556 (0.476 to 0.644) | 0.347 (0.330 to 0.367) | 14.0% | 44% | 2 | 24 of 24 | $72.28 | $5 | 27 min, $3.23 |
| 5 | GPT-5.6 Terra | 0.515 (0.443 to 0.594) | 0.337 (0.324 to 0.353) | 14.1% | 44% | 1 | 24 of 24 | $84.41 | $5 | 52 min, $3.70 |
| 6 | Kimi K3 | 0.426 (0.366 to 0.491) | 0.326 (0.310 to 0.342) | 23.1% | 31% | 0 | 24 of 24 | $150.51 | $7.50 | 57 min, $6.01 |
| | GPT-6.1 Sol, one agent at all three desks | 0.677 (0.592 to 0.766) | 0.366 (0.346 to 0.391) | 7.7% | 72% | 5 | 24 of 24 | $35.66 | $6 | 44 min, $1.52 |
| | Standard crew desk (B0) | 0.386 | | | 0% | | | | | |
| | v2's best plan, blind to passengers | 0.602 | | | | | | | | |

Rows 1 to 6 are each model's three desks, ranked by their mean. The 95% CIs are a bootstrap over the 24 dates (10,000
resamples). "Under the best plan": the runs that came in under the best plan found, which score 1.0 and are flagged
`below_best`. "Valid days": the runs whose audit passed. "Spend": the model spend of the 24 scored runs. "Median day":
a day's wall time and spend at the median. The avoidable cost is B0's excess over the best plan, so its median is over
the 23 dates where B0 isn't the best plan.

- **GPT-6 Astra has the highest mean, but the board doesn't separate it from GPT-6.1 Sol or Claude Opus 5.5:** their
  95% intervals overlap Astra's (its lower end is 0.673; Sol's upper end 0.811, Opus's 0.711), and Astra's lead over
  Sol is 0.021. Claude Sonnet 5.5, GPT-5.6 Terra and Kimi K3 score below Astra: their intervals lie wholly under its
  own. `reward_L`, which the best plan doesn't cap, ranks the six in the same order.
- **One run a date for each model.** A model's interval is over the dates, not over repeated runs, and nothing is
  paired across models; as registered, the models are compared by whether their intervals overlap the leader's.
- **The best plan's ceiling compresses the top.** A run under the best plan found scores 1.0 whichever model played it
  and however far under it came: 24 of the 168 runs, all on the 7 dates where Sol's three desks did (Sol's 12 over both
  modes, Astra's 6, Opus's 3, Sonnet's 2, Terra's 1 and Kimi K3's none). `reward_L` is the fairer reading there.
- **Spend doesn't follow the ranking.** Astra's board cost $544.24, 7.6 times Sol's $71.38, for 0.021 more; Opus's
  $261.46, 3.7 times; Kimi K3's $150.51, 2.1 times Sol's, for the lowest mean; Sonnet's ($72.28) and Terra's ($84.41)
  are near Sol's.

**How the five ran.** As the plugin's
[README](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.5.0/README.md#the-v3-board) records, the
models, the dates, the caps, the retries and how the models would be compared were registered before the first of them
ran, and the five boards ran at its commit 6e5ec17, which priced GPT-5.6 Terra and Kimi K3 in the desk agent. Sol
wasn't run again: its three-desk runs are its board's 24.

- **Caps.** A desk's cap buys what $3 bought a Sol desk: Sol's median desk's tokens at the model's prices, rounded up to
  the next $0.50, so $23.50 a desk for GPT-6 Astra, $9.50 for Claude Opus 5.5, $7.50 for Kimi K3 and $5 for Claude
  Sonnet 5.5 and GPT-5.6 Terra. The caps stopped 5 of the five boards' 360 desks, all crew desks: Astra's on 2023-12-18,
  Opus's on 2023-07-08, 2023-07-29 and 2023-12-10, and Kimi K3's on 2025-12-14. Each of those runs scored.
- **Retries.** GPT-5.6 Terra's 24 runs took 31 attempts: 7 first attempts lost a desk to the provider's token rate
  limit (`provider_error`), so their audits failed and they weren't scored; each was retried, as registered, and scored
  on its retry. The dataset keeps every attempt: `v3_episodes` has 175 rows on the eval split, the 168 scored runs and
  Terra's 7 superseded attempts.
- **Spend.** $1,125.30 on the models for the five boards, Terra's 7 unscored attempts ($12.39) included; the table's
  spend is each model's 24 scored runs.
- **Checks.** All 120 runs passed their audits, and every reward and `reward_L` recomputes from the run's graded cost
  and the pack's best plan, bound and B0 (within 1e-6). The plugin replayed all 168 board runs exactly through its env
  on Linux. This port's code replays all 120 the same way on Linux (54,036 calls): every call's result, the grade,
  the audit, the digest and the passengers.

Each model's three-desk run, by date:

| Date | GPT-6 Astra | GPT-6.1 Sol | Claude Opus 5.5 | Claude Sonnet 5.5 | GPT-5.6 Terra | Kimi K3 |
|---|---:|---:|---:|---:|---:|---:|
| 2022-12-07 | **0.484** | 0.477 | 0.378 | 0.296 | 0.288 | 0.249 |
| 2023-06-06 | 0.775 | **0.848** | 0.696 | 0.594 | 0.821 | 0.668 |
| 2023-06-16 | **0.686** | 0.548 | 0.481 | 0.434 | 0.620 | 0.432 |
| 2023-07-08 | **0.555** | 0.528 | 0.471 | 0.385 | 0.462 | 0.321 |
| 2023-07-09 | **0.488** | 0.468 | 0.441 | 0.378 | 0.270 | 0.330 |
| 2023-07-13 | 1.000 \* | 1.000 \* | 0.558 | 0.741 | 0.696 | 0.580 |
| 2023-07-29 | 1.000 \* | 1.000 \* | 1.000 \* | 1.000 \* | 0.644 | 0.543 |
| 2023-12-10 | 0.772 | **0.825** | 0.708 | 0.563 | 0.419 | 0.416 |
| 2023-12-18 | **0.655** | 0.406 | 0.260 | 0.355 | 0.304 | 0.302 |
| 2024-06-23 | 1.000 \* | 1.000 \* | 0.798 | 1.000 \* | 0.853 | 0.203 |
| 2024-07-13 | 1.000 \* | 1.000 \* | 0.657 | 0.713 | 0.593 | 0.564 |
| 2024-07-15 | 1.000 \* | 1.000 \* | 1.000 \* | 0.576 | 1.000 \* | 0.201 |
| 2024-07-22 | 1.000 \* | 1.000 \* | 1.000 \* | 0.933 | 0.296 | 0.803 |
| 2025-01-11 | 0.816 | 0.853 | **0.868** | 0.725 | 0.612 | 0.579 |
| 2025-07-08 | **0.815** | 0.795 | 0.751 | 0.780 | 0.358 | 0.413 |
| 2025-07-12 | **0.601** | 0.587 | 0.561 | 0.418 | 0.462 | 0.303 |
| 2025-12-02 | 0.534 | 0.393 | **0.541** | 0.329 | 0.467 | 0.415 |
| 2025-12-14 | **0.518** | 0.452 | 0.424 | 0.357 | 0.316 | 0.381 |
| 2025-12-19 | 0.598 | **0.650** | 0.462 | 0.510 | 0.474 | 0.407 |
| 2025-12-30 | 0.834 | **1.000** \* | 0.887 | 0.349 | 0.443 | 0.717 |
| 2026-06-14 | **0.658** | 0.628 | 0.430 | 0.397 | 0.324 | 0.291 |
| 2026-07-01 | 0.673 | **0.708** | 0.653 | 0.544 | 0.529 | 0.401 |
| 2026-07-09 | **0.641** | 0.595 | 0.496 | 0.462 | 0.535 | 0.328 |
| 2026-07-15 | **0.705** | 0.538 | 0.512 | 0.494 | 0.585 | 0.375 |

The day's highest in bold, none where several runs share 1.0. \* Under the best plan found, so 1.0 and flagged
`below_best`. Sol's dates, with their storm hubs, flights, passengers and B0, are in its table below.

`reward_L`, against the bound:

| Date | GPT-6 Astra | GPT-6.1 Sol | Claude Opus 5.5 | Claude Sonnet 5.5 | GPT-5.6 Terra | Kimi K3 |
|---|---:|---:|---:|---:|---:|---:|
| 2022-12-07 | 0.329 | 0.329 | 0.319 | 0.306 | 0.304 | 0.294 |
| 2023-06-06 | 0.345 | 0.348 | 0.341 | 0.336 | 0.347 | 0.340 |
| 2023-06-16 | 0.366 | 0.351 | 0.342 | 0.335 | 0.359 | 0.335 |
| 2023-07-08 | 0.371 | 0.366 | 0.354 | 0.333 | 0.352 | 0.313 |
| 2023-07-09 | 0.364 | 0.359 | 0.352 | 0.334 | 0.290 | 0.317 |
| 2023-07-13 | 0.383 | 0.380 | 0.343 | 0.357 | 0.354 | 0.345 |
| 2023-07-29 | 0.344 | 0.339 | 0.329 | 0.311 | 0.301 | 0.298 |
| 2023-12-10 | 0.388 | 0.393 | 0.380 | 0.362 | 0.337 | 0.336 |
| 2023-12-18 | 0.356 | 0.327 | 0.293 | 0.319 | 0.307 | 0.307 |
| 2024-06-23 | 0.340 | 0.337 | 0.304 | 0.323 | 0.305 | 0.246 |
| 2024-07-13 | 0.634 | 0.586 | 0.469 | 0.489 | 0.444 | 0.433 |
| 2024-07-15 | 0.332 | 0.323 | 0.332 | 0.304 | 0.320 | 0.241 |
| 2024-07-22 | 0.351 | 0.359 | 0.344 | 0.323 | 0.292 | 0.319 |
| 2025-01-11 | 0.483 | 0.493 | 0.496 | 0.459 | 0.427 | 0.417 |
| 2025-07-08 | 0.435 | 0.431 | 0.424 | 0.429 | 0.328 | 0.346 |
| 2025-07-12 | 0.345 | 0.344 | 0.342 | 0.327 | 0.332 | 0.307 |
| 2025-12-02 | 0.330 | 0.319 | 0.330 | 0.311 | 0.325 | 0.321 |
| 2025-12-14 | 0.356 | 0.344 | 0.339 | 0.323 | 0.311 | 0.329 |
| 2025-12-19 | 0.350 | 0.355 | 0.335 | 0.341 | 0.337 | 0.327 |
| 2025-12-30 | 0.370 | 0.383 | 0.374 | 0.318 | 0.333 | 0.362 |
| 2026-06-14 | 0.355 | 0.352 | 0.331 | 0.326 | 0.312 | 0.304 |
| 2026-07-01 | 0.406 | 0.413 | 0.402 | 0.380 | 0.376 | 0.342 |
| 2026-07-09 | 0.365 | 0.360 | 0.347 | 0.341 | 0.352 | 0.314 |
| 2026-07-15 | 0.355 | 0.341 | 0.338 | 0.336 | 0.345 | 0.321 |

### GPT-6.1 Sol: three desks against one agent

Sol's board, which the plugin published in v0.4.0, also played every date with one agent at all three desks; it is
here as it was.

Registered before the first run, as the plugin's
[README](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.4.2/README.md#the-v3-board) records: the
code and the pack with it, the dates, the caps ($3 a desk in a three-desk run, $6 for one agent) and the budget. Sol
ran at reasoning effort `high`, once per date in each mode: three desks talking (the crew scheduler, the customer desk
and the ops controller, each its own agent), and one agent at all three desks.

| Mode | Dates | Mean reward (95% CI) | reward_L (95% CI) | Over the best plan, median | Avoidable cost saved, median | Under the best plan | Valid days | Spend |
|---|---:|---|---|---:|---:|---:|---:|---:|
| GPT-6.1 Sol, three desks, talking | 24 | **0.721** (0.633 to 0.811) | 0.372 (0.351 to 0.398) | 7.7% | 71% | 7 | 24 of 24 | $71.38 |
| GPT-6.1 Sol, one agent at all three desks | 24 | **0.677** (0.592 to 0.766) | 0.366 (0.346 to 0.391) | 7.7% | 72% | 5 | 24 of 24 | $35.66 |
| Standard crew desk (B0) | 24 | 0.386 | | | 0% | | | |
| v2's best plan, blind to passengers | 24 | 0.602 | | | | | | |

On the same dates doing nothing (the planned crews by the trip rule, no desk at all) scores 0.329, and JetBlue's own
decisions, replayed with the generated crews and passengers, 0.239. B0 scores 0.308 on 20 of the 24 dates and more on
the 4 where it comes within 10% of the best plan: 0.907 on 2023-07-29, 1.0 on 2024-06-23 (where B0 is the best plan),
0.697 on 2024-07-15 and 0.498 on 2024-07-22. The avoidable cost is B0's excess over the best plan, so its median is
over the other 23 dates.

- **Three desks against one agent, paired by date:** +0.044 reward (95% CI 0.007 to 0.083); three desks higher on 14 of
  24 dates, lower on 5, and level on the 5 where both came in under the best plan; their days cost 1.8% of the best
  plan less (CI 0.5 to 3.1%), and `reward_L` is +0.006 (0.002 to 0.011).
- **A small margin.** A two-sided sign test on the 19 dates where the two differ gives p = 0.064, and leaving out the 7
  dates with a run under the best plan gives +0.043 (CI −0.002 to 0.090) over the other 17.
- **It doesn't show that the third desk or talking helps.** The three desks used about twice the compute: 2.0× the
  spend, 2.0× the input tokens and 2.4× the output tokens, more spend on all 24 dates. There is no two-desk arm under
  P, so the board doesn't measure what the customer desk adds by itself either. Before any model ran, a scripted
  customer desk built from published practice saved at least 2% against two scripted desks on 2 of the 24 eval dates,
  against 12 needed, so that gate failed and v3 shipped with the result stated.
- **Both well above standard practice:** at the median a Sol day saves 71% (three desks) or 72% (one agent) of what B0
  leaves on the table and costs 7.7% more than the best plan, and both modes' means are above v2's best plan played
  with its passengers (0.602).
- **Flagged: 12 of the 48 runs came in under the best plan found,** by 0.8% to 8.9%: both modes on 2023-07-13,
  2023-07-29, 2024-06-23, 2024-07-15 and 2024-07-22, and three desks alone on 2024-07-13 and 2025-12-30. They score
  1.0 and are flagged `below_best`, with `reward_L`, 0.323 to 0.586, the fairer reading. On 4 of those 5 dates B0
  itself comes within 10% of the best plan, which is weak there. As registered before the board, the best plan isn't
  updated from played runs, so no run's score depends on another's.
- A valid day is one whose audit passed: every watch's news on time, every call in its desk's turn, no decision reaching
  back, each desk in its role, a legal crew on every flight, the cost not under the bound, and every desk present to the
  end or stood down. Every run scored. Ops and crew stood down on 20 of the 24 three-desk days and the customer desk on
  15 (it finished the day on the other 9); one agent stood down on 21 of 24; no desk ended on an error.

**The customer desk's requests.** Over the 24 three-desk days the customer desk sent ops 1,303 requests, a median of
56 a day (27 to 78):

| Request | Sent | Accepted by ops | Declined | Lapsed, expired or replaced |
|---|---:|---:|---:|---:|
| HOLD (hold a departure for its connecting passengers) | 887 | 463 | 357 | 67 |
| PROTECT (don't cancel a flight) | 184 | 177 | 5 | 2 |
| SUGGEST_CANCEL (cancel a flight) | 76 | 24 | 33 | 19 |
| TEXT | 156 | | | |

An accepted HOLD is ops' hold; an accepted PROTECT or SUGGEST_CANCEL is ops' word, which it carries out, if at all,
with its own tools. Ops texted the customer desk 67 times. One agent has no one to ask.

**Spend.** The three-desk days cost $71.38 (the crew desks $40.05, the customer desks $13.88, the ops desks $17.45),
the one-agent days $35.66: a median three-desk day took 70 minutes and $2.79, one agent 44 minutes and $1.52. Two crew
desks reached 90% or more of their $3 cap (96% and 92%), and none was stopped by it.

| Date | Storm hubs | Flights | Passengers | Connecting | Gap, best to bound | B0 | Three desks | One agent | Three desks, reward_L | One agent, reward_L | Spend, three desks | Spend, one agent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-12-07 | JFK, BOS | 767 | 86,500 | 3.5% | 78.7% | 0.308 | **0.477** | 0.465 | 0.329 | 0.327 | $2.33 | $1.71 |
| 2023-06-06 | BOS | 774 | 93,810 | 2.9% | 64.5% | 0.308 | **0.848** | 0.587 | 0.348 | 0.335 | $1.46 | $0.75 |
| 2023-06-16 | JFK, BOS | 753 | 91,328 | 2.9% | 59.6% | 0.308 | **0.548** | 0.515 | 0.351 | 0.347 | $2.92 | $1.38 |
| 2023-07-08 | JFK, BOS | 693 | 86,237 | 2.8% | 54.2% | 0.308 | **0.528** | 0.440 | 0.366 | 0.347 | $4.76 | $1.93 |
| 2023-07-09 | JFK, BOS | 733 | 90,919 | 2.8% | 62.8% | 0.308 | **0.468** | 0.411 | 0.359 | 0.344 | $4.94 | $1.85 |
| 2023-07-13 | JFK, BOS | 750 | 92,275 | 2.9% | 68.4% | 0.308 | 1.000 \* | 1.000 \* | 0.380 | 0.385 | $3.34 | $1.64 |
| 2023-07-29 | JFK, BOS | 698 | 87,063 | 2.8% | 69.8% | 0.907 | 1.000 \* | 1.000 \* | 0.339 | 0.334 | $4.24 | $1.56 |
| 2023-12-10 | JFK, BOS | 665 | 75,737 | 2.4% | 71.6% | 0.308 | **0.825** | 0.663 | 0.393 | 0.375 | $2.56 | $1.52 |
| 2023-12-18 | BOS | 752 | 86,058 | 2.4% | 73.7% | 0.308 | 0.406 | **0.470** | 0.327 | 0.337 | $4.60 | $2.10 |
| 2024-06-23 | JFK, BOS | 662 | 81,982 | 2.6% | 66.0% | 1.000 | 1.000 \* | 1.000 \* | 0.337 | 0.324 | $2.69 | $1.47 |
| 2024-07-13 | BOS | 612 | 78,055 | 2.8% | 69.2% | 0.308 | **1.000** \* | 0.937 | 0.586 | 0.563 | $2.42 | $1.67 |
| 2024-07-15 | JFK, BOS | 664 | 84,523 | 2.9% | 65.9% | 0.697 | 1.000 \* | 1.000 \* | 0.323 | 0.338 | $2.85 | $1.10 |
| 2024-07-22 | JFK | 668 | 85,342 | 2.9% | 67.3% | 0.498 | 1.000 \* | 1.000 \* | 0.359 | 0.350 | $3.14 | $1.44 |
| 2025-01-11 | BOS | 494 | 59,412 | 2.6% | 65.3% | 0.308 | 0.853 | **0.865** | 0.493 | 0.496 | $3.02 | $1.38 |
| 2025-07-08 | JFK, BOS | 637 | 81,279 | 2.3% | 65.6% | 0.308 | **0.795** | 0.655 | 0.431 | 0.405 | $2.89 | $1.75 |
| 2025-07-12 | JFK | 647 | 83,103 | 2.2% | 60.5% | 0.308 | **0.587** | 0.560 | 0.344 | 0.342 | $2.61 | $1.37 |
| 2025-12-02 | BOS | 619 | 76,562 | 2.0% | 60.2% | 0.308 | 0.393 | **0.398** | 0.319 | 0.319 | $1.78 | $1.06 |
| 2025-12-14 | JFK | 667 | 81,685 | 2.2% | 59.7% | 0.308 | **0.452** | 0.386 | 0.344 | 0.330 | $3.84 | $1.58 |
| 2025-12-19 | JFK, BOS | 708 | 87,356 | 2.0% | 69.7% | 0.308 | **0.650** | 0.508 | 0.355 | 0.341 | $2.65 | $1.33 |
| 2025-12-30 | JFK | 689 | 85,650 | 2.0% | 76.0% | 0.308 | **1.000** \* | 0.740 | 0.383 | 0.363 | $2.74 | $1.52 |
| 2026-06-14 | JFK, BOS | 722 | 89,741 | 2.0% | 73.5% | 0.308 | **0.628** | 0.591 | 0.352 | 0.349 | $2.39 | $1.73 |
| 2026-07-01 | BOS | 705 | 91,080 | 2.4% | 63.9% | 0.308 | **0.708** | 0.664 | 0.413 | 0.405 | $1.73 | $1.14 |
| 2026-07-09 | JFK, BOS | 764 | 96,759 | 3.0% | 58.3% | 0.308 | 0.595 | **0.719** | 0.360 | 0.374 | $3.45 | $1.73 |
| 2026-07-15 | JFK | 739 | 93,021 | 3.0% | 76.6% | 0.308 | 0.538 | **0.669** | 0.341 | 0.352 | $2.04 | $0.96 |

The higher of the two modes in bold. \* Under the best plan found, so 1.0 and flagged `below_best`; `reward_L` is the
fairer reading there. "Passengers": the day's passenger trips; "Connecting": the share of them that connect. The gap
from the best plan to the bound is mostly the bound's weakness (by the plugin's FIDELITY.md, HiGHS stopped at its time
limit on every date), so `reward_L` reads low throughout.

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

## How it was run

- **Harness:** v1's planner [`airops-llm`](../envs/airops/agentenv/agents/airops-llm/) with 3 planning calls a
  watch and 5 turns a watch plus 4, 32,000 output tokens a reply, an episode cap of $3; v2's desks
  [`airops-desk`](../envs/airops/agentenv/agents/airops-desk/), each its own A2A agent in its role, reasoning effort
  `high`, 6 model turns a watch plus 10 for a desk of a two-desk run and 8 plus 10 for one agent, 32,000 output tokens
  a reply, capped at $5 a desk in a two-desk run and $8 for one agent; v3's desks, the same agent in role `crew`,
  `customer`, `ops` or `both`, reasoning effort `high`, 6 model turns a watch plus 10 for a desk of a three-desk run
  (up to 232) and 10 plus 10 for one agent (up to 380), 32,000 output tokens a reply, capped at $3 a desk in a
  three-desk run and $6 for one agent; the five other models' desks, the same agent with three desks only, capped at
  $23.50 a desk for GPT-6 Astra, $9.50 for Claude Opus 5.5, $7.50 for Kimi K3 and $5 for Claude Sonnet 5.5 and GPT-5.6
  Terra. One agent at v3's three desks sees 16 of their 22 tools: no requests or replies, and no `crew_program`. One
  run per date and mode, on local Docker; the five other models' boards on five machines at once, one a model.
- **Route:** GPT-6.1 Sol on the Responses API, served by Azure OpenAI, through a LiteLLM proxy. The five other models
  through the same agent, on the API it uses for each provider: GPT-6 Astra and GPT-5.6 Terra on the Responses API,
  served by Azure OpenAI; Claude Opus 5.5 and Claude Sonnet 5.5 on the Messages API, served by Anthropic; Kimi K3 on
  OpenAI-compatible chat, served by Fireworks. Spend is the harness's tokens times the prices pinned in the agent.
- **Spend:** v1 $2.99. v2's board $73.94 with the superseded attempt; the 6-run trial sweep before it $6.88. v3: Sol's
  board $107.04, and the plugin's FIDELITY.md gives $6.20 for the 4-run pilot before it, which the dataset doesn't
  hold; the five other models' boards $1,125.30, GPT-5.6 Terra's 7 unscored attempts ($12.39) included.
- **Checks:** every scored run replays exactly through the env's calls and passes the audit, and an independent
  recheck recomputed every reward and reward_L from the graded costs and the pack's best plan, bound and B0 (all 168 of
  v3's board runs within 1e-6).

## Caveats

- One run per date and mode, and one model on v1 and v2: the intervals, a bootstrap over the dates, are wide; v3's six
  models are compared by whether their intervals overlap the leader's, with nothing paired across models; and five v1
  days are not a board.
- The crews are generated and the crew filter leans the eval dates toward moderate storms; the plugin's
  [FIDELITY.md](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.3.3/FIDELITY.md) sets out every
  layer's checks and misses.
- v3's passengers are generated too, from route-month loads and a ticket survey, and its best plan is the best of a
  fixed set of candidates, not an optimum; its bound sits 54% to 79% under it. The plugin's
  [FIDELITY.md at v0.5.0](https://github.com/earakely-scale/agentenv-airops-plugin/blob/v0.5.0/FIDELITY.md) sets out
  v3's checks and misses, the five other models' boards among them.
