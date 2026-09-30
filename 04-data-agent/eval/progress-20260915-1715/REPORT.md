# Multi-harness progress — 2026-09-15T17:15:12.383455+00:00

Trainer **79083** is at **step 504** on hopper-prod. Qwen3.5-2B; E2B; OpenCode, Claude Code, Codex and Mini-SWE-Agent. LR 3e-6, eight generations per task, max staleness four. Saves every 50 steps plus hourly recovery saves; independent eval every 100 steps.

The fixed test set contains 250 tasks: 33 easy, 118 medium and 99 hard. Each complete checkpoint evaluation has 1,000 pass@1 cells. Partial scores remain provisional until coverage, token and harness-version audits pass.

| Harness | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenCode | 10.8% (27/250) | 24.4% (61/250) | 30.4% (76/250) | 29.6% (74/250) | 32.8% (82/250) |
| Claude Code | 16.8% (42/250) | 27.6% (69/250) | 30.0% (75/250) | 33.2% (83/250) | 36.8% (92/250) |
| Codex | 16.4% (41/250) | 28.0% (70/250) | 26.4% (66/250) | 29.6% (74/250) | 34.8% (87/250) |
| Mini-SWE-Agent | 14.4% (36/250) | 19.2% (48/250) | 18.4% (46/250) | 22.0% (55/250) | 28.8% (72/250) |
| Overall | 14.6% (146/1000) | 24.8% (248/1000) | 26.3% (263/1000) | 28.6% (286/1000) | 33.3% (333/1000) |

| Difficulty, all harnesses | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 40.2% (53/132) | 53.0% (70/132) | 58.3% (77/132) | 64.4% (85/132) | 73.5% (97/132) |
| medium | 14.4% (68/472) | 30.5% (144/472) | 30.1% (142/472) | 32.8% (155/472) | 39.6% (187/472) |
| hard | 6.3% (25/396) | 8.6% (34/396) | 11.1% (44/396) | 11.6% (46/396) | 12.4% (49/396) |

| Harness | Difficulty | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenCode | easy | 33.3% (11/33) | 51.5% (17/33) | 51.5% (17/33) | 60.6% (20/33) | 72.7% (24/33) |
| OpenCode | medium | 8.5% (10/118) | 31.4% (37/118) | 34.7% (41/118) | 33.1% (39/118) | 38.1% (45/118) |
| OpenCode | hard | 6.1% (6/99) | 7.1% (7/99) | 18.2% (18/99) | 15.2% (15/99) | 13.1% (13/99) |
| Claude Code | easy | 42.4% (14/33) | 60.6% (20/33) | 63.6% (21/33) | 66.7% (22/33) | 75.8% (25/33) |
| Claude Code | medium | 18.6% (22/118) | 32.2% (38/118) | 33.1% (39/118) | 39.0% (46/118) | 44.9% (53/118) |
| Claude Code | hard | 6.1% (6/99) | 11.1% (11/99) | 15.2% (15/99) | 15.2% (15/99) | 14.1% (14/99) |
| Codex | easy | 42.4% (14/33) | 57.6% (19/33) | 63.6% (21/33) | 69.7% (23/33) | 72.7% (24/33) |
| Codex | medium | 16.9% (20/118) | 35.6% (42/118) | 29.7% (35/118) | 33.1% (39/118) | 42.4% (50/118) |
| Codex | hard | 7.1% (7/99) | 9.1% (9/99) | 10.1% (10/99) | 12.1% (12/99) | 13.1% (13/99) |
| Mini-SWE-Agent | easy | 42.4% (14/33) | 42.4% (14/33) | 54.5% (18/33) | 60.6% (20/33) | 72.7% (24/33) |
| Mini-SWE-Agent | medium | 13.6% (16/118) | 22.9% (27/118) | 22.9% (27/118) | 26.3% (31/118) | 33.1% (39/118) |
| Mini-SWE-Agent | hard | 6.1% (6/99) | 7.1% (7/99) | 1.0% (1/99) | 4.0% (4/99) | 9.1% (9/99) |

On the same cells completed by the latest checkpoint: base: 14.6% (146/1000); 100: 24.8% (248/1000); 200: 26.3% (263/1000); 300: 28.6% (286/1000); 400: 33.3% (333/1000).

Infrastructure attempts without a graded result are retained separately and may be retried. A scored zero is never replaced by a retry. Ungraded attempts by checkpoint: {"100": 49, "200": 81, "300": 29, "400": 49}.

## Training signal and reliability

| Steps | Mean logged reward | Nonzero-gradient updates | Mean step time |
| --- | ---: | ---: | ---: |
| 54–100 | 0.378 | 34/47 | 99.8s |
| 101–150 | 0.411 | 31/50 | 99.0s |
| 151–200 | 0.257 | 28/50 | 90.5s |
| 201–250 | 0.234 | 28/50 | 92.1s |
| 251–280 | 0.370 | 24/30 | 90.4s |
| 281–310 | 0.416 | 19/30 | 79.9s |
| 311–504 | 0.374 | 114/194 | 89.7s |

Reward is the unweighted mean of logged optimizer-update reward, not fixed-task pass@1. The changing task/harness mix affects it. Zero within-group reward variance produces no relative-advantage learning signal; high mean reward alone does not guarantee useful updates.

The current continuation has 187/308 nonzero-gradient updates and 0 nonfinite updates. The latest 20 have 9 nonzero gradients and 11 zero-variance updates. Observed staleness is at most 4.0; 12 whole rollouts (75 rows) were rejected by the staleness policy; 0 oversized rows were dropped.

Latest saved checkpoints: 200, 238, 250, 294, 300, 337, 350, 394, 400, 446, 450, 491, 500. The audited captures pass TiTO on 1625/1625 completed rollouts, retaining 4,611,934/4,611,934 eligible tokens. Audit timestamp: 2026-09-15T17:13:02.438285+00:00. This is capture/sequence-assembly evidence, not a claim that every captured rollout reached the optimizer.

Continuation coverage: 203/1,000 unique training tasks. The continuation started with a saved schedule cursor, so this count excludes the parent run’s earlier coverage; optimizer steps are not unique tasks.

Monitor alerts: [] at 2026-09-15T17:13:04.000184+00:00. The latest nonzero-gradient update is step 504; monitor alerts may precede newer updates. Offline logging healthy: True; online Trackio healthy: True at 2026-09-15T17:14:58.595938+00:00.

Source paths, exact counts, resume-chain provenance and timestamps are preserved in the companion JSON snapshot.
