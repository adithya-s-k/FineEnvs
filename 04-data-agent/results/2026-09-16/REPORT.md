# Harbor and OpenCode — consolidated training and pass@1

Updated: 2026-09-16T10:50:30.515660+00:00

[Live Trackio dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio) · [Overview image](comparison.png) · [Snapshot](snapshot.json.gz)

Qwen3.5-2B; 1,000 optimizer-step target per run. Recorded training: Harbor multi-harness: 1000 steps; Native OpenCode: 1000 steps; Harbor OpenCode-only: 74 steps. Every accepted checkpoint has 250 fixed tasks × four harnesses = 1,000 grades. Task difficulty: 33 easy, 118 medium, 99 hard (13.2% / 47.2% / 39.6%). Scores retain first graded attempts; incomplete and failed-audit evaluations are excluded. Missing scores are not estimated.

Baselines are separate measured cohorts: Harbor/E2B 14.6%; Harbor/Daytona 15.9% for the native OpenCode checkpoint evaluator. The standalone native OpenCode 8.4% baseline uses a different harness protocol and is excluded here. Infrastructure and training recipe histories differ; this is an observational comparison, not a controlled causal experiment.

## Overall checkpoint curve

| Checkpoint | Harbor multi-harness | Native OpenCode | Harbor OpenCode-only |
| --- | ---: | ---: | ---: |
| 0 (baseline) | 14.6% | 15.9% | 14.6% |
| 100 | 24.8% | 19.7% | Pending |
| 200 | 26.3% | 22.1% | Pending |
| 300 | 28.6% | 21.6% | Pending |
| 400 | 33.3% | 26.4% | Pending |
| 500 | 37.0% | 23.1% | Pending |
| 600 | 31.8% | 25.1% | Pending |
| 684 (recovery) | 32.1% | Not scheduled | Not scheduled |
| 700 | 28.8% | 23.2% | Pending |
| 800 | 27.0% | 25.6% | Pending |
| 900 | Pending | 25.3% | Pending |
| 1000 | Pending | 29.8% | Pending |

## Harbor multi-harness

### Overall and difficulty

| Checkpoint | Overall | Easy (132 cells) | Medium (472) | Hard (396) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 14.6% | 40.2% | 14.4% | 6.3% |
| 100 | 24.8% | 53.0% | 30.5% | 8.6% |
| 200 | 26.3% | 58.3% | 30.1% | 11.1% |
| 300 | 28.6% | 64.4% | 32.8% | 11.6% |
| 400 | 33.3% | 73.5% | 39.6% | 12.4% |
| 500 | 37.0% | 72.7% | 44.3% | 16.4% |
| 600 | 31.8% | 72.7% | 37.3% | 11.6% |
| 684 | 32.1% | 75.8% | 35.8% | 13.1% |
| 700 | 28.8% | 60.6% | 33.9% | 12.1% |
| 800 | 27.0% | 59.1% | 32.6% | 9.6% |

### Harness × difficulty at every checkpoint

| Checkpoint | Harness | Overall (250) | Easy (33) | Medium (118) | Hard (99) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | opencode | 10.8% | 33.3% (11/33) | 8.5% (10/118) | 6.1% (6/99) |
| 0 | claude-code | 16.8% | 42.4% (14/33) | 18.6% (22/118) | 6.1% (6/99) |
| 0 | codex | 16.4% | 42.4% (14/33) | 16.9% (20/118) | 7.1% (7/99) |
| 0 | mini-swe-agent | 14.4% | 42.4% (14/33) | 13.6% (16/118) | 6.1% (6/99) |
| 100 | opencode | 24.4% | 51.5% (17/33) | 31.4% (37/118) | 7.1% (7/99) |
| 100 | claude-code | 27.6% | 60.6% (20/33) | 32.2% (38/118) | 11.1% (11/99) |
| 100 | codex | 28.0% | 57.6% (19/33) | 35.6% (42/118) | 9.1% (9/99) |
| 100 | mini-swe-agent | 19.2% | 42.4% (14/33) | 22.9% (27/118) | 7.1% (7/99) |
| 200 | opencode | 30.4% | 51.5% (17/33) | 34.7% (41/118) | 18.2% (18/99) |
| 200 | claude-code | 30.0% | 63.6% (21/33) | 33.1% (39/118) | 15.2% (15/99) |
| 200 | codex | 26.4% | 63.6% (21/33) | 29.7% (35/118) | 10.1% (10/99) |
| 200 | mini-swe-agent | 18.4% | 54.5% (18/33) | 22.9% (27/118) | 1.0% (1/99) |
| 300 | opencode | 29.6% | 60.6% (20/33) | 33.1% (39/118) | 15.2% (15/99) |
| 300 | claude-code | 33.2% | 66.7% (22/33) | 39.0% (46/118) | 15.2% (15/99) |
| 300 | codex | 29.6% | 69.7% (23/33) | 33.1% (39/118) | 12.1% (12/99) |
| 300 | mini-swe-agent | 22.0% | 60.6% (20/33) | 26.3% (31/118) | 4.0% (4/99) |
| 400 | opencode | 32.8% | 72.7% (24/33) | 38.1% (45/118) | 13.1% (13/99) |
| 400 | claude-code | 36.8% | 75.8% (25/33) | 44.9% (53/118) | 14.1% (14/99) |
| 400 | codex | 34.8% | 72.7% (24/33) | 42.4% (50/118) | 13.1% (13/99) |
| 400 | mini-swe-agent | 28.8% | 72.7% (24/33) | 33.1% (39/118) | 9.1% (9/99) |
| 500 | opencode | 32.8% | 69.7% (23/33) | 40.7% (48/118) | 11.1% (11/99) |
| 500 | claude-code | 44.8% | 75.8% (25/33) | 52.5% (62/118) | 25.3% (25/99) |
| 500 | codex | 39.2% | 75.8% (25/33) | 46.6% (55/118) | 18.2% (18/99) |
| 500 | mini-swe-agent | 31.2% | 69.7% (23/33) | 37.3% (44/118) | 11.1% (11/99) |
| 600 | opencode | 34.0% | 66.7% (22/33) | 41.5% (49/118) | 14.1% (14/99) |
| 600 | claude-code | 30.0% | 72.7% (24/33) | 33.9% (40/118) | 11.1% (11/99) |
| 600 | codex | 32.4% | 66.7% (22/33) | 39.8% (47/118) | 12.1% (12/99) |
| 600 | mini-swe-agent | 30.8% | 84.8% (28/33) | 33.9% (40/118) | 9.1% (9/99) |
| 684 | opencode | 29.2% | 72.7% (24/33) | 30.5% (36/118) | 13.1% (13/99) |
| 684 | claude-code | 35.6% | 72.7% (24/33) | 41.5% (49/118) | 16.2% (16/99) |
| 684 | codex | 33.2% | 84.8% (28/33) | 36.4% (43/118) | 12.1% (12/99) |
| 684 | mini-swe-agent | 30.4% | 72.7% (24/33) | 34.7% (41/118) | 11.1% (11/99) |
| 700 | opencode | 21.2% | 30.3% (10/33) | 28.0% (33/118) | 10.1% (10/99) |
| 700 | claude-code | 31.6% | 78.8% (26/33) | 34.7% (41/118) | 12.1% (12/99) |
| 700 | codex | 32.0% | 72.7% (24/33) | 34.7% (41/118) | 15.2% (15/99) |
| 700 | mini-swe-agent | 30.4% | 60.6% (20/33) | 38.1% (45/118) | 11.1% (11/99) |
| 800 | opencode | 23.2% | 36.4% (12/33) | 33.1% (39/118) | 7.1% (7/99) |
| 800 | claude-code | 32.0% | 66.7% (22/33) | 37.3% (44/118) | 14.1% (14/99) |
| 800 | codex | 26.8% | 66.7% (22/33) | 31.4% (37/118) | 8.1% (8/99) |
| 800 | mini-swe-agent | 26.0% | 66.7% (22/33) | 28.8% (34/118) | 9.1% (9/99) |

### Training history

| Allocation | First optimizer step | Last optimizer step |
| --- | ---: | ---: |
| 78647 | 1 | 17 |
| 78681 | 18 | 25 |
| 78767 | 26 | 30 |
| 78831 | 31 | 53 |
| 78956 | 54 | 196 |
| 79083 | 197 | 684 |
| 80608 | 685 | 1000 |

### Score provenance

- Step 0: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914/job-78215/canonical_results.json`; SHA256 `8c4f5bced4eff04b0c2e5f41806da9ae1b8a4c0fe356ddf926e1f781c6bb9ac6`.
- Step 100: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/checkpoint-evals/step-000100/scores.json`; SHA256 `1e4f42f54526c9b8b71156f5b1f3673529519201401bc88f88ccff805f291181`.
- Step 200: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000200/scores.json`; SHA256 `699dce549d1002e879c675555ac06142448b0cbeef534a0816616faf669862af`.
- Step 300: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000300/scores.json`; SHA256 `299e5ed3528922d9912a591f3cff0c4d85070fef4de732d37723967934bfd691`.
- Step 400: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000400/scores.json`; SHA256 `f8909af81669f4ec092317c5f9889ef8735754c0dcbc826da82a270d2d92dc7a`.
- Step 500: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000500/scores.json`; SHA256 `86d56b65edbdc2f5a3f5d54151d0888dae83ca2b81753a7a9dbc2e80ee4f6130`.
- Step 600: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000600/scores.json`; SHA256 `02fc5a5e84978c198dc880c143fe223507c30485563b3545cb64b2794e3e60b0`.
- Step 684: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000684/scores.json`; SHA256 `fb54bd52f352463e405bee8067ef24b27147bfd02d41429561f440a916f5d776`.
- Step 700: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-cont-20260915/checkpoint-evals/step-000700/scores.json`; SHA256 `39c2ead8681847c6c398eb6b22ae8919aabaff3ad6b668d81b5961564139f8be`.
- Step 800: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-cont-20260915/checkpoint-evals/step-000800/scores.json`; SHA256 `65130786707d07f68cfa145fcd5ad7308890cb89a305b65382dfbec36ef7a150`.

## Native OpenCode

### Overall and difficulty

| Checkpoint | Overall | Easy (132 cells) | Medium (472) | Hard (396) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 15.9% | 37.9% | 18.0% | 6.1% |
| 100 | 19.7% | 44.7% | 22.5% | 8.1% |
| 200 | 22.1% | 59.8% | 24.6% | 6.6% |
| 300 | 21.6% | 51.5% | 25.6% | 6.8% |
| 400 | 26.4% | 59.1% | 29.4% | 11.9% |
| 500 | 23.1% | 53.0% | 27.3% | 8.1% |
| 600 | 25.1% | 56.1% | 28.8% | 10.4% |
| 700 | 23.2% | 49.2% | 28.8% | 7.8% |
| 800 | 25.6% | 52.3% | 29.7% | 11.9% |
| 900 | 25.3% | 50.0% | 30.3% | 11.1% |
| 1000 | 29.8% | 59.8% | 35.8% | 12.6% |

### Harness × difficulty at every checkpoint

| Checkpoint | Harness | Overall (250) | Easy (33) | Medium (118) | Hard (99) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | opencode | 12.8% | 24.2% (8/33) | 16.9% (20/118) | 4.0% (4/99) |
| 0 | claude-code | 16.8% | 42.4% (14/33) | 18.6% (22/118) | 6.1% (6/99) |
| 0 | codex | 15.2% | 33.3% (11/33) | 16.9% (20/118) | 7.1% (7/99) |
| 0 | mini-swe-agent | 18.8% | 51.5% (17/33) | 19.5% (23/118) | 7.1% (7/99) |
| 100 | opencode | 19.6% | 42.4% (14/33) | 24.6% (29/118) | 6.1% (6/99) |
| 100 | claude-code | 20.4% | 42.4% (14/33) | 22.9% (27/118) | 10.1% (10/99) |
| 100 | codex | 18.0% | 27.3% (9/33) | 21.2% (25/118) | 11.1% (11/99) |
| 100 | mini-swe-agent | 20.8% | 66.7% (22/33) | 21.2% (25/118) | 5.1% (5/99) |
| 200 | opencode | 17.2% | 48.5% (16/33) | 21.2% (25/118) | 2.0% (2/99) |
| 200 | claude-code | 24.0% | 60.6% (20/33) | 26.3% (31/118) | 9.1% (9/99) |
| 200 | codex | 25.6% | 63.6% (21/33) | 28.8% (34/118) | 9.1% (9/99) |
| 200 | mini-swe-agent | 21.6% | 66.7% (22/33) | 22.0% (26/118) | 6.1% (6/99) |
| 300 | opencode | 20.8% | 48.5% (16/33) | 28.0% (33/118) | 3.0% (3/99) |
| 300 | claude-code | 29.2% | 66.7% (22/33) | 30.5% (36/118) | 15.2% (15/99) |
| 300 | codex | 16.8% | 39.4% (13/33) | 20.3% (24/118) | 5.1% (5/99) |
| 300 | mini-swe-agent | 19.6% | 51.5% (17/33) | 23.7% (28/118) | 4.0% (4/99) |
| 400 | opencode | 20.4% | 48.5% (16/33) | 24.6% (29/118) | 6.1% (6/99) |
| 400 | claude-code | 32.4% | 60.6% (20/33) | 35.6% (42/118) | 19.2% (19/99) |
| 400 | codex | 25.6% | 57.6% (19/33) | 28.0% (33/118) | 12.1% (12/99) |
| 400 | mini-swe-agent | 27.2% | 69.7% (23/33) | 29.7% (35/118) | 10.1% (10/99) |
| 500 | opencode | 18.0% | 48.5% (16/33) | 19.5% (23/118) | 6.1% (6/99) |
| 500 | claude-code | 26.4% | 51.5% (17/33) | 33.1% (39/118) | 10.1% (10/99) |
| 500 | codex | 18.8% | 48.5% (16/33) | 22.0% (26/118) | 5.1% (5/99) |
| 500 | mini-swe-agent | 29.2% | 63.6% (21/33) | 34.7% (41/118) | 11.1% (11/99) |
| 600 | opencode | 16.8% | 42.4% (14/33) | 18.6% (22/118) | 6.1% (6/99) |
| 600 | claude-code | 32.4% | 63.6% (21/33) | 37.3% (44/118) | 16.2% (16/99) |
| 600 | codex | 18.4% | 45.5% (15/33) | 23.7% (28/118) | 3.0% (3/99) |
| 600 | mini-swe-agent | 32.8% | 72.7% (24/33) | 35.6% (42/118) | 16.2% (16/99) |
| 700 | opencode | 18.4% | 42.4% (14/33) | 23.7% (28/118) | 4.0% (4/99) |
| 700 | claude-code | 29.6% | 63.6% (21/33) | 33.1% (39/118) | 14.1% (14/99) |
| 700 | codex | 14.4% | 24.2% (8/33) | 22.9% (27/118) | 1.0% (1/99) |
| 700 | mini-swe-agent | 30.4% | 66.7% (22/33) | 35.6% (42/118) | 12.1% (12/99) |
| 800 | opencode | 20.0% | 36.4% (12/33) | 26.3% (31/118) | 7.1% (7/99) |
| 800 | claude-code | 32.4% | 72.7% (24/33) | 35.6% (42/118) | 15.2% (15/99) |
| 800 | codex | 18.0% | 30.3% (10/33) | 21.2% (25/118) | 10.1% (10/99) |
| 800 | mini-swe-agent | 32.0% | 69.7% (23/33) | 35.6% (42/118) | 15.2% (15/99) |
| 900 | opencode | 18.8% | 33.3% (11/33) | 23.7% (28/118) | 8.1% (8/99) |
| 900 | claude-code | 34.0% | 60.6% (20/33) | 42.4% (50/118) | 15.2% (15/99) |
| 900 | codex | 14.4% | 27.3% (9/33) | 17.8% (21/118) | 6.1% (6/99) |
| 900 | mini-swe-agent | 34.0% | 78.8% (26/33) | 37.3% (44/118) | 15.2% (15/99) |
| 1000 | opencode | 20.4% | 42.4% (14/33) | 25.4% (30/118) | 7.1% (7/99) |
| 1000 | claude-code | 33.2% | 66.7% (22/33) | 37.3% (44/118) | 17.2% (17/99) |
| 1000 | codex | 29.6% | 57.6% (19/33) | 35.6% (42/118) | 13.1% (13/99) |
| 1000 | mini-swe-agent | 36.0% | 72.7% (24/33) | 44.9% (53/118) | 13.1% (13/99) |

### Training history

| Allocation | First optimizer step | Last optimizer step |
| --- | ---: | ---: |
| 80626 | 1 | 1000 |

### Score provenance

- Step 0: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/20260915/blackbox/canonical_scores.json`; SHA256 `ece2b0e03c7c7e0e54988eaaa473ba6b53bd6028315b1e99ec39df5137c7632e`.
- Step 100: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80657/canonical_scores.json`; SHA256 `37eab8fe9e97fda23e9803846f965b08c3de2a6e4142363219a4467af41c6f5c`.
- Step 200: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80675/canonical_scores.json`; SHA256 `fa67547d19c7f4e63166fc7c1518ca15eb5e2c28ec8f1f3739445029006617ad`.
- Step 300: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80748/canonical_scores.json`; SHA256 `a0e0cc489e3195827d5ac035945277bfd9ca67e985a015148e5b3c94ca938b0f`.
- Step 400: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80807/canonical_scores.json`; SHA256 `f30bb541e207a2e8b83b2aabd05bf2e3d96eeac40c2fd8226ff1985606397a7b`.
- Step 500: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80861/canonical_scores.json`; SHA256 `b6df2570695c2a15ba43f185719b647dc22319eb82ca1494d56e705572e3f1a2`.
- Step 600: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80902/canonical_scores.json`; SHA256 `d86128c2cae4813a7ae8b99f06d1cd8ca109cefade9944c1cdbebf2b1550e1d9`.
- Step 700: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80956/canonical_scores.json`; SHA256 `7c60cfab333948e63bf44bd01ed4ce3f0a788f76de84c4ceda2177144e9bc9ba`.
- Step 800: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-80993/canonical_scores.json`; SHA256 `b689c9dd76c3e2230bfea49eea393f2d5842fe8f3630f04f59380a131497ae98`.
- Step 900: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-81034/canonical_scores.json`; SHA256 `996f5ea262419b9639fa8f33c1b33fef9b49959c1cbe61e62ba922c0d642985f`.
- Step 1000: `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-81098/canonical_scores.json`; SHA256 `1355a9a2ecc1ec165cf413120dacfc672e5d8d59ef2807b28bcf02322dca142b`.

## Harbor OpenCode-only

### Overall and difficulty

| Checkpoint | Overall | Easy (132 cells) | Medium (472) | Hard (396) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 14.6% | 40.2% | 14.4% | 6.3% |

### Harness × difficulty at every checkpoint

| Checkpoint | Harness | Overall (250) | Easy (33) | Medium (118) | Hard (99) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | opencode | 10.8% | 33.3% (11/33) | 8.5% (10/118) | 6.1% (6/99) |
| 0 | claude-code | 16.8% | 42.4% (14/33) | 18.6% (22/118) | 6.1% (6/99) |
| 0 | codex | 16.4% | 42.4% (14/33) | 16.9% (20/118) | 7.1% (7/99) |
| 0 | mini-swe-agent | 14.4% | 42.4% (14/33) | 13.6% (16/118) | 6.1% (6/99) |

### Training history

| Allocation | First optimizer step | Last optimizer step |
| --- | ---: | ---: |
| 81075 | 1 | 74 |

### Score provenance

- Step 0: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914/job-78215/canonical_results.json`; SHA256 `8c4f5bced4eff04b0c2e5f41806da9ae1b8a4c0fe356ddf926e1f781c6bb9ac6`.

## Dashboard metric guide

Both runs use identical metric names and optimizer-step axes. `eval/pass_at_1` is the overall score; `eval/difficulty/*` aggregates each difficulty; `eval/harness/*` compares each harness; `eval/harness_difficulty/*` contains all twelve intersections. `train/*` preserves recorded loss, reward, learning rate, gradient norm, entropy, KL, staleness, throughput, token, batching and rollout metrics where observed. Missing metrics are not filled with zeros. `train/reward_rolling20` and `train/nonzero_gradient_rolling20` are explicitly derived trailing windows. Raw metrics remain available. Use zero dashboard smoothing for exact checkpoint values.

The independent CPU publisher refreshes every 60 seconds and admits new evaluations only after their full comparison gates pass. It never changes trainer state. Local SQLite backup, event ledger and remote exact-content verification receipts are kept alongside this report.

Storage and deployment follow the [Trackio guide](https://huggingface.co/docs/trackio/quickstart) and [environment configuration](https://huggingface.co/docs/trackio/environment_variables).

- [Overview](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5E%28eval%2Fpass_at_1%7Ctrain%2Freward_rolling20%29%24)
- [Difficulty](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5Eeval%2Fdifficulty%2F)
- [Harness](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5Eeval%2Fharness%2F)
- [Harness × difficulty](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5Eeval%2Fharness_difficulty%2F)
- [Optimizer diagnostics](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5Etrain%2F%28loss%7Cgrad_norm%7Centropy%7Ckl%7Clearning_rate%7Cnonzero_gradient_rolling20%29%24)
- [Throughput and rollout diagnostics](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=%5Etrain%2F%28perf%7Crollout%7Csample%7Cbatch%29%2F)
- [All metrics](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&run_ids=3ae29a23763093285702b71a1f76805a%2Cbeb2604c8f737da4262b1ab19b8b0cbd%2Cc5b445fa337b56b139c1e6b34ae35409&smoothing=0&metric_filter=)

[Download checkpoint scores as CSV](checkpoint_scores.csv)
