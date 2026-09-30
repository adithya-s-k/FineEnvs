---
title: SmolDataEnv RL
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# SmolDataEnv RL

Training and evaluation results in one public Trackio project: **data-agent-rl-comparison**.

## Dashboard navigation

The default page opens six LFM overview charts directly, with section headers hidden and smoothing set to 0. The top navigation provides Qwen results, per-harness scores, tool/token usage, full diagnostics and the separate SFT dashboard. Explicit shared run selections and metric filters are preserved. Use **All RL metrics** for the complete seven-run view; expand a section to see its charts.

LFM training reward includes the efficiency bonus. Qwen RL used correctness only. A score of 0.542 means 54.2%; provisional scores exclude missing cases, which remain visible in coverage charts.

## SFT now has its own dashboard

[SFT training and evaluations](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio) now have a dedicated public CPU-basic Space. This dashboard defaults to the seven existing RL/baseline runs. The previously imported SFT records remain in the raw database for provenance, but are omitted from this Space's run selector. No scalar records were deleted.

Dense training plots are sampled to roughly 800 points per run for browser responsiveness. Every evaluation and runtime-summary record is retained; raw records remain available through the query API. This replaces the previous unlimited plotting response.

## LFM2.5-2.6B: OpenCode-only vs multi-harness

Both runs reached 1,000 optimizer steps. Both final evaluations have 1,000/1,000 graded cases: **52.3% OpenCode-only**, **54.2% multi-harness**. The evaluation set is 250 fixed tasks × four harnesses, pass@1.

[Overview from step 0](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ccombined_reward_normalized%7Ctool_call_savings_pct%29%7Ceval_coverage%2F%28graded_cells%7Cmissing_cells%29%7Ctrain_verified%2Freward_mean50%29%24) · [All LFM metrics](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_efficiency%7Ceval_usage%7Ceval_coverage%7Ctrain_verified%29%2F) · [Harness scores](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Eeval_observed%2Fharness%2F) · [Difficulty](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Eeval_observed%2F%28difficulty%7Charness_difficulty%29%2F) · [Tool and token savings](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Eeval_efficiency%2F) · [Actual token/tool usage](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Eeval_usage%2F%28overall%7Charness%2F%5B%5E%2F%5D%2B%29%2F%28generated_tokens_mean%7Cinput_tokens_mean%7Cnative_tool_calls_mean%29%24) · [Training diagnostics](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Etrain_verified%2F) · [Coverage](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=2710d85285055cff752a1ab35e0ac343%2C7a4d3a85b9b0718a035f6242627ebfc7&smoothing=0&metric_filter=%5Eeval_coverage%2F)

The overview contains the baseline and every available checkpoint, including incomplete early evaluations. Scores use fractions: 0.542 means 54.2%. Keep dashboard smoothing at **0**; the training `reward_mean50` series is already smoothed over 50 updates.

| Metric group | Meaning |
|---|---|
| `eval_observed` | All available graded results, including provisional checkpoints; missing cases are excluded, not counted as failures |
| `eval_coverage` | Graded, missing and correct counts, completeness, native tool-count and token coverage |
| `eval_efficiency` | Positive tool/generated-token/input-token savings versus baseline on tasks both models solved; matched counts included |
| `eval_usage` | Actual average native tool calls, input/generated tokens, turns and duration across graded rollouts |
| `train_verified` | Committed training lineage: reward, loss, gradients, tokens, calls, staleness and performance diagnostics |

The shared pretrained baseline is repeated at step 0 in both training curves for comparison; it is one evaluation, not two independent measurements. Baseline coverage is **998/1,000**. OpenCode checkpoints 100–400 and multi-harness checkpoints 100–200 are also incomplete. See coverage charts or [checkpoint coverage CSV](lfm_checkpoint_coverage.csv). Later checkpoints, including both final results, are complete.

Normalized combined reward is `mean(correctness × (1 + tool bonus)) / 1.1`, with bonus `1.5 / (15 + native tool calls)` only when a positive native count is verified. Efficiency cohorts vary across checkpoints and runs; comparisons are descriptive, not causal. Input tokens include repeated history and are not cache-adjusted.

The raw multi-harness logs preserve 17 superseded pre-resume updates (851–867). `train_verified` excludes them and follows the saved checkpoint lineage used in the final figures. Original logs remain available.

[Overview figure](lfm_final_overview.png) · [Tool and token figure](lfm_final_efficiency.png) · [Overview PDF](lfm_final_overview.pdf) · [Tool and token PDF](lfm_final_efficiency.pdf) · [Verification and source notes](LFM_DATA.md)

## Historical runs

The same project retains Qwen 2B multi-harness, native OpenCode, Harbor OpenCode-only and hard-task continuation runs, plus the LFM pretrained baseline. Clear the run selection to include them. Historical backends, retry policies and reasoning settings differ; compare with their protocol notes.

`eval/pass_at_1` retains completed audited scores; `eval/provisional_pass_at_1` retains partial results. The new `eval_observed` series unifies the available LFM checkpoints into continuous curves while preserving explicit coverage metrics. Original databases and scalar records were retained.

[Historical report](REPORT.md) · [Historical checkpoint CSV](checkpoint_scores.csv) · [Historical provisional scores](PROVISIONAL_EVALS.md)
