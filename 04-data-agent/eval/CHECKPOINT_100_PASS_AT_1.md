# Four-harness checkpoint 100 — completed pass@1

Completed on 2026-09-15 with **1,000/1,000 graded cells**, the same 250 test tasks per harness, matching harness versions and all selected captures passing the exact-token TiTO audit.

| Harness | Base pass@1 | Checkpoint-100 pass@1 | Change |
| --- | ---: | ---: | ---: |
| OpenCode | 27/250 · 10.8% | 61/250 · 24.4% | +13.6 percentage points |
| Claude Code | 42/250 · 16.8% | 69/250 · 27.6% | +10.8 points |
| Codex | 41/250 · 16.4% | 70/250 · 28.0% | +11.6 points |
| mini-swe-agent | 36/250 · 14.4% | 48/250 · 19.2% | +4.8 points |
| **Overall** | **146/1,000 · 14.6%** | **248/1,000 · 24.8%** | **+10.2 points** |

These are pass@1 cell averages, not pass@4. See [the base protocol](BASELINE_PASS_AT_1.md) for model, test-data, sampling, budget and retry definitions. This evaluation uses E2B and is separate from the new Daytona comparison baselines.

The original evaluator job 78987 finished with 999 graded cells. Recovery job 79057 preserved those 999 measurements and recovered the remaining OpenCode test-index-6 cell; cleanup job 79058 completed. There were 49 ungraded attempts across the evaluation history. They were excluded and retried, never scored as zeros and never used to replace an existing graded result.

Evidence: `experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/checkpoint-evals/step-000100/scores.json`, `job-79057/final_tito.json`, and `operations/EVAL100_RECOVERY.json` under the run root. Training job 78956 continued separately throughout recovery.
