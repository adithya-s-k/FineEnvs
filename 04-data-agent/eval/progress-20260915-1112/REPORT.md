# Evaluation and reward snapshot — 2026-09-15T11:13:22.806509+00:00

Same 250 test tasks per harness: 33 easy, 118 medium, 99 hard. Metric: pass@1.

Checkpoint200 has 978/1000 graded cells. Its full coverage, harness-version and TiTO final audit remains pending.

| Harness | Base | Checkpoint100 | Checkpoint200 — provisional |
| --- | ---: | ---: | ---: |
| OpenCode | 10.8% (27/250) | 24.4% (61/250) | 30.6% (76/248) |
| Claude Code | 16.8% (42/250) | 27.6% (69/250) | 30.9% (71/230) |
| Codex | 16.4% (41/250) | 28.0% (70/250) | 26.4% (66/250) |
| Mini-SWE-Agent | 14.4% (36/250) | 19.2% (48/250) | 18.4% (46/250) |
| Overall | 14.6% (146/1000) | 24.8% (248/1000) | 26.5% (259/978) |

| Harness | Difficulty | Base | Checkpoint100 | Checkpoint200 — provisional |
| --- | --- | ---: | ---: | ---: |
| OpenCode | easy | 33.3% (11/33) | 51.5% (17/33) | 51.5% (17/33) |
| OpenCode | medium | 8.5% (10/118) | 31.4% (37/118) | 34.7% (41/118) |
| OpenCode | hard | 6.1% (6/99) | 7.1% (7/99) | 18.6% (18/97) |
| Claude Code | easy | 42.4% (14/33) | 60.6% (20/33) | 63.6% (21/33) |
| Claude Code | medium | 18.6% (22/118) | 32.2% (38/118) | 34.3% (37/108) |
| Claude Code | hard | 6.1% (6/99) | 11.1% (11/99) | 14.6% (13/89) |
| Codex | easy | 42.4% (14/33) | 57.6% (19/33) | 63.6% (21/33) |
| Codex | medium | 16.9% (20/118) | 35.6% (42/118) | 29.7% (35/118) |
| Codex | hard | 7.1% (7/99) | 9.1% (9/99) | 10.1% (10/99) |
| Mini-SWE-Agent | easy | 42.4% (14/33) | 42.4% (14/33) | 54.5% (18/33) |
| Mini-SWE-Agent | medium | 13.6% (16/118) | 22.9% (27/118) | 22.9% (27/118) |
| Mini-SWE-Agent | hard | 6.1% (6/99) | 7.1% (7/99) | 1.0% (1/99) |

| Difficulty, all harnesses | Base | Checkpoint100 | Checkpoint200 — provisional |
| --- | ---: | ---: | ---: |
| easy | 40.2% (53/132) | 53.0% (70/132) | 58.3% (77/132) |
| medium | 14.4% (68/472) | 30.5% (144/472) | 30.3% (140/462) |
| hard | 6.3% (25/396) | 8.6% (34/396) | 10.9% (42/384) |

On the same 978 completed task–harness pairs: base: 14.7% (144/978); 100: 25.2% (246/978); 200: 26.5% (259/978).

## Training reward

| Steps | Mean logged reward | Nonzero-gradient updates |
| --- | ---: | ---: |
| 54–100 | 0.3784 | 34/47 |
| 101–150 | 0.4115 | 31/50 |
| 151–200 | 0.2571 | 28/50 |
| 201–260 | 0.2495 | 37/60 |
| 221–240 | 0.3294 | 13/20 |
| 241–260 | 0.2375 | 16/20 |

Unweighted mean of logged per-update reward; not unique-rollout or fixed-task pass@1. Training task/harness mix varies. Plot starts at bounded-admission recipe step54.

Current trainer79083: step260, 39/64 nonzero-gradient updates; 0 nonfinite updates; observed maximum staleness4.0; 4.0 whole-rollout stale rejections (67.0 rows), at steps[218]; 0 oversized-row rejections.

Monitor/Trackio timestamps, TiTO evidence, exclusions, matched counts, source-chain cutoffs and all plotted values are in snapshot.json and the CSV files.
