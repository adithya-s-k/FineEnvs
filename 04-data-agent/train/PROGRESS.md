# Multi-harness progress — 2026-09-16T05:23:17.913132+00:00

**Historical snapshot.** All three async trainers and their scheduled checkpoint evaluations
finished on September 17. See [completed results](../results.md) and the
[training/evaluation analysis](../reports/three-run-analysis-20260917/REPORT.md) for current findings.

Trainer **80608** is at **step 943** on hopper-prod. Qwen3.5-2B; E2B; OpenCode, Claude Code, Codex and Mini-SWE-Agent. LR 3e-6, eight generations per task, max staleness four. Saves every 50 steps plus hourly recovery saves; independent eval every 100 steps.

The fixed test set contains 250 tasks: 33 easy, 118 medium and 99 hard. Each complete checkpoint evaluation has 1,000 pass@1 cells. Partial scores remain provisional until coverage, token and harness-version audits pass.

| Harness | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 | Checkpoint 500 | Checkpoint 600 | Checkpoint 684 (partial) | Checkpoint 700 (partial) | Checkpoint 800 (partial) | Checkpoint 900 (partial) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCode | 10.8% (27/250) | 24.4% (61/250) | 30.4% (76/250) | 29.6% (74/250) | 32.8% (82/250) | 32.8% (82/250) | 34.0% (85/250) | 29.6% (73/247) | 21.4% (52/243) | 24.0% (58/242) | pending |
| Claude Code | 16.8% (42/250) | 27.6% (69/250) | 30.0% (75/250) | 33.2% (83/250) | 36.8% (92/250) | 44.8% (112/250) | 30.0% (75/250) | 35.6% (89/250) | 31.6% (79/250) | 32.0% (80/250) | 0.0% (0/2) |
| Codex | 16.4% (41/250) | 28.0% (70/250) | 26.4% (66/250) | 29.6% (74/250) | 34.8% (87/250) | 39.2% (98/250) | 32.4% (81/250) | 33.2% (83/250) | 32.0% (80/250) | 26.8% (67/250) | 100.0% (1/1) |
| Mini-SWE-Agent | 14.4% (36/250) | 19.2% (48/250) | 18.4% (46/250) | 22.0% (55/250) | 28.8% (72/250) | 31.2% (78/250) | 30.8% (77/250) | 30.4% (76/250) | 30.4% (76/250) | 26.2% (65/248) | pending |
| Overall | 14.6% (146/1000) | 24.8% (248/1000) | 26.3% (263/1000) | 28.6% (286/1000) | 33.3% (333/1000) | 37.0% (370/1000) | 31.8% (318/1000) | 32.2% (321/997) | 28.9% (287/993) | 27.3% (270/990) | 33.3% (1/3) |

| Difficulty, all harnesses | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 | Checkpoint 500 | Checkpoint 600 | Checkpoint 684 (partial) | Checkpoint 700 (partial) | Checkpoint 800 (partial) | Checkpoint 900 (partial) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 40.2% (53/132) | 53.0% (70/132) | 58.3% (77/132) | 64.4% (85/132) | 73.5% (97/132) | 72.7% (96/132) | 72.7% (96/132) | 75.8% (100/132) | 61.5% (80/130) | 59.5% (78/131) | pending |
| medium | 14.4% (68/472) | 30.5% (144/472) | 30.1% (142/472) | 32.8% (155/472) | 39.6% (187/472) | 44.3% (209/472) | 37.3% (176/472) | 35.9% (169/471) | 34.0% (159/467) | 33.1% (154/465) | 50.0% (1/2) |
| hard | 6.3% (25/396) | 8.6% (34/396) | 11.1% (44/396) | 11.6% (46/396) | 12.4% (49/396) | 16.4% (65/396) | 11.6% (46/396) | 13.2% (52/394) | 12.1% (48/396) | 9.6% (38/394) | 0.0% (0/1) |

| Harness | Difficulty | Base | Checkpoint 100 | Checkpoint 200 | Checkpoint 300 | Checkpoint 400 | Checkpoint 500 | Checkpoint 600 | Checkpoint 684 (partial) | Checkpoint 700 (partial) | Checkpoint 800 (partial) | Checkpoint 900 (partial) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCode | easy | 33.3% (11/33) | 51.5% (17/33) | 51.5% (17/33) | 60.6% (20/33) | 72.7% (24/33) | 69.7% (23/33) | 66.7% (22/33) | 72.7% (24/33) | 32.3% (10/31) | 37.5% (12/32) | pending |
| OpenCode | medium | 8.5% (10/118) | 31.4% (37/118) | 34.7% (41/118) | 33.1% (39/118) | 38.1% (45/118) | 40.7% (48/118) | 41.5% (49/118) | 30.8% (36/117) | 28.3% (32/113) | 34.5% (39/113) | pending |
| OpenCode | hard | 6.1% (6/99) | 7.1% (7/99) | 18.2% (18/99) | 15.2% (15/99) | 13.1% (13/99) | 11.1% (11/99) | 14.1% (14/99) | 13.4% (13/97) | 10.1% (10/99) | 7.2% (7/97) | pending |
| Claude Code | easy | 42.4% (14/33) | 60.6% (20/33) | 63.6% (21/33) | 66.7% (22/33) | 75.8% (25/33) | 75.8% (25/33) | 72.7% (24/33) | 72.7% (24/33) | 78.8% (26/33) | 66.7% (22/33) | pending |
| Claude Code | medium | 18.6% (22/118) | 32.2% (38/118) | 33.1% (39/118) | 39.0% (46/118) | 44.9% (53/118) | 52.5% (62/118) | 33.9% (40/118) | 41.5% (49/118) | 34.7% (41/118) | 37.3% (44/118) | 0.0% (0/1) |
| Claude Code | hard | 6.1% (6/99) | 11.1% (11/99) | 15.2% (15/99) | 15.2% (15/99) | 14.1% (14/99) | 25.3% (25/99) | 11.1% (11/99) | 16.2% (16/99) | 12.1% (12/99) | 14.1% (14/99) | 0.0% (0/1) |
| Codex | easy | 42.4% (14/33) | 57.6% (19/33) | 63.6% (21/33) | 69.7% (23/33) | 72.7% (24/33) | 75.8% (25/33) | 66.7% (22/33) | 84.8% (28/33) | 72.7% (24/33) | 66.7% (22/33) | pending |
| Codex | medium | 16.9% (20/118) | 35.6% (42/118) | 29.7% (35/118) | 33.1% (39/118) | 42.4% (50/118) | 46.6% (55/118) | 39.8% (47/118) | 36.4% (43/118) | 34.7% (41/118) | 31.4% (37/118) | 100.0% (1/1) |
| Codex | hard | 7.1% (7/99) | 9.1% (9/99) | 10.1% (10/99) | 12.1% (12/99) | 13.1% (13/99) | 18.2% (18/99) | 12.1% (12/99) | 12.1% (12/99) | 15.2% (15/99) | 8.1% (8/99) | pending |
| Mini-SWE-Agent | easy | 42.4% (14/33) | 42.4% (14/33) | 54.5% (18/33) | 60.6% (20/33) | 72.7% (24/33) | 69.7% (23/33) | 84.8% (28/33) | 72.7% (24/33) | 60.6% (20/33) | 66.7% (22/33) | pending |
| Mini-SWE-Agent | medium | 13.6% (16/118) | 22.9% (27/118) | 22.9% (27/118) | 26.3% (31/118) | 33.1% (39/118) | 37.3% (44/118) | 33.9% (40/118) | 34.7% (41/118) | 38.1% (45/118) | 29.3% (34/116) | pending |
| Mini-SWE-Agent | hard | 6.1% (6/99) | 7.1% (7/99) | 1.0% (1/99) | 4.0% (4/99) | 9.1% (9/99) | 11.1% (11/99) | 9.1% (9/99) | 11.1% (11/99) | 11.1% (11/99) | 9.1% (9/99) | pending |

On the cells completed by every displayed checkpoint: base: 0.0% (0/3); 100: 66.7% (2/3); 200: 66.7% (2/3); 300: 33.3% (1/3); 400: 33.3% (1/3); 500: 66.7% (2/3); 600: 0.0% (0/3); 684: 33.3% (1/3); 700: 66.7% (2/3); 800: 0.0% (0/3); 900: 33.3% (1/3).

Infrastructure attempts without a graded result are retained separately and may be retried. A scored zero is never replaced by a retry. Ungraded attempts by checkpoint: {"100": 49, "200": 81, "300": 29, "400": 49, "500": 85, "600": 124, "684": 79, "700": 88, "800": 157, "900": 5}.

## Training signal and reliability

| Steps | Mean logged reward | Nonzero-gradient updates | Mean step time |
| --- | ---: | ---: | ---: |
| 54–100 | 0.378 | 34/47 | 99.8s |
| 101–150 | 0.411 | 31/50 | 99.0s |
| 151–200 | 0.257 | 28/50 | 90.5s |
| 201–250 | 0.234 | 28/50 | 92.1s |
| 251–280 | 0.370 | 24/30 | 90.4s |
| 281–310 | 0.416 | 19/30 | 79.9s |
| 311–943 | 0.393 | 400/633 | 95.9s |

Reward is the unweighted mean of logged optimizer-update reward, not fixed-task pass@1. The changing task/harness mix affects it. Zero within-group reward variance produces no relative-advantage learning signal; high mean reward alone does not guarantee useful updates.

The current continuation has 166/259 nonzero-gradient updates and 0 nonfinite updates. The latest 20 have 16 nonzero gradients and 4 zero-variance updates. Observed staleness is at most 4.0; 4 whole rollouts (9 rows) were rejected by the staleness policy; 0 oversized rows were dropped.

Latest saved checkpoints: 700, 730, 750, 799, 800, 837, 850, 884, 900, 926. The audited captures pass TiTO on 1307/1336 completed rollouts, retaining 8,330,190/8,330,190 eligible tokens. Audit timestamp: 2026-09-16T05:22:53.434075+00:00. This is capture/sequence-assembly evidence, not a claim that every captured rollout reached the optimizer.

Continuation coverage: 166/1,000 unique training tasks. The continuation started with a saved schedule cursor, so this count excludes the parent run’s earlier coverage; optimizer steps are not unique tasks.

Monitor alerts: ["eval_failed:checkpoint-700", "eval_failed:checkpoint-800", "eval_failed:checkpoint-900", "tito_failure:claude-code", "tito_failure:codex", "tito_failure:mini-swe-agent", "tito_failure:opencode"] at 2026-09-16T05:22:55.223692+00:00. The latest nonzero-gradient update is step 943; monitor alerts may precede newer updates. Offline logging healthy: True; latest online Trackio sync successful: True at 2026-09-16T05:22:48.150772+00:00.

Source paths, exact counts, resume-chain provenance and timestamps are preserved in the companion JSON snapshot.

## Checkpoint-500 regression and throughput follow-up

Read-only diagnosis: [EVAL_REGRESSION_500_600.md](EVAL_REGRESSION_500_600.md). The 37.0% → 31.8% decline is concentrated in Claude Code and Codex on medium/hard tasks; the training cause remains unproven.

A 50-update throughput snapshot on 2026-09-16 measured synchronous SETA at 319.4 seconds/update (steps 50–99), native async OpenCode at 31.4 seconds/update (707–756), and multi-harness async at 148.9 seconds/update (897–946). Approximate optimizer-consumed rollouts/hour: 90, 514, and 114, respectively. These are observed workload rates, not an isolated sync/async experiment; hardware, trajectory lengths and rollouts per update differ. Evidence: `experiments/daytona_harness_comparison/logs/hf-20260915/three-run-monitor/status-check-20260916/sync-async-throughput.json`.
