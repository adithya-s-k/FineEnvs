# SFT training runtimes and evaluations

Snapshot: 29 September 2026. Six completed two-epoch SFT runs, published in the dedicated SFT project `data-agent-sft-comparison`.

[Overview](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ctool_call_savings_pct%29%7Ctrain_sft%2Floss_mean50%7Ceval_coverage%2Fgraded_cells%29%24) · [Training](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28train_sft%7Ctrain_summary%29%2F) · [Evaluations](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_coverage%7Ceval_usage%7Ceval_efficiency%29%2F)

| Run | Tasks / examples | Train runtime | Mean loss | Loss: first / last 50 | Epoch 1 pass@1 | Epoch 2 pass@1 |
|---|---:|---:|---:|---:|---:|---:|
| LFM 2.6B · SFT bash · 907 tasks | 907 / 907 | 0h 09m 28s | 0.4464 | 0.5437 / 0.3974 | 41.74% (999/1000) | 46.59% (850/1000) |
| LFM 2.6B · SFT bash · 4677 tasks (eval overlap) | 4,677 / 4,677 | 0h 45m 29s | 0.3533 | 0.5188 / 0.3372 | 41.30% (1000/1000) | 40.80% (1000/1000) |
| LFM 2.6B · SFT OpenCode · 801 tasks | 801 / 4,825 | 1h 28m 17s | 0.2906 | 0.3997 / 0.2789 | 45.10% (1000/1000) | 47.50% (1000/1000) |
| Qwen 2B · SFT OpenCode · 801 tasks | 801 / 4,825 | 1h 11m 27s | 0.2270 | 0.2825 / 0.2147 | 28.40% (1000/1000) | 26.50% (1000/1000) |
| LFM 2.6B · SFT four harnesses · 888 tasks | 888 / 17,929 | 6h 26m 50s | 0.5815 | 0.9076 / 0.5854 | 38.30% (1000/1000) | 43.10% (1000/1000) |
| Qwen 2B · SFT four harnesses · 888 tasks | 888 / 17,929 | 5h 18m 45s | 0.3844 | 0.5131 / 0.3797 | 30.90% (1000/1000) | 32.70% (1000/1000) |

Runtime is the final Trainer `train_runtime`: training loop only, excluding queue time, preprocessing and evaluation. Loss is supervised cross entropy; absolute losses across different training datasets are not comparable. Training curves include every logged optimizer update; `loss_mean50` is a trailing mean with shorter windows at the start. Keep dashboard smoothing at 0. Historical ingestion timestamps are not original training timestamps.

## Evaluation interpretation

- Pass@1 uses 250 fixed tasks across OpenCode, Claude Code, Codex and Mini-SWE-Agent. Each harness has 33 easy, 118 medium and 99 hard tasks. Scores are fractions on Trackio.
- Bash 907 epoch 1 has 999/1000 graded; epoch 2 has 850/1000. These are provisional, excluding missing cases. All other SFT epoch evaluations have 1000/1000 graded.
- Bash 4,677 includes 33 notebook overlaps and four matching test questions. It is an overlap-inclusive experiment, not a clean held-out result.
- The LFM pretrained baseline has 998/1000 graded. The Qwen baseline has 1000/1000 graded. A baseline repeated at step 0 is the same shared measurement, not another evaluation.
- The x axis uses each run's optimizer steps; compare epochs as well, since dataset sizes and numbers of updates differ.
- Tool savings = 100 × (1 − checkpoint calls / baseline calls), on task/harness pairs solved by both. Positive means fewer native ATIF tool calls. Matched cohort sizes are logged; cohorts differ between runs. This is descriptive, not a causal estimate.
- SFT has no explicit tool-efficiency reward. Qwen RL used correctness only; LFM RL used correctness plus a tool-efficiency bonus.
- Token means include all captured calls and repeated history; they are not cache-adjusted. Missing counters are omitted, never filled with zero.

## Data and citation

[Training summary CSV](sft_training_summary.csv) · [Training summary JSON](sft_training_summary.json) · [Epoch evaluations, harness breakdowns and efficiency](sft_evaluations.json)

These aggregate exports include SHA-256 hashes of the source metrics and evaluation summaries. The public dashboard contains scalar metrics only. No rollout text, credentials or model weights are included. For a reproducible article citation, use the commit-pinned version of this document and exports rather than relying solely on the live dashboard.

Training datasets: [bash demonstrations](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft), [OpenCode teacher rollouts](https://huggingface.co/datasets/AdithyaSK/qwen38-27b-harbor-rollouts), [multi-harness SFT](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft).

## Plot rendering

Dense training curves are sampled to roughly 800 points per run for browser responsiveness. Every evaluation and runtime-summary point is retained. All 12,782 original training updates remain in the database and in [the full scalar archive](sft_events.json.gz). The archive is restored automatically on a fresh Space start.
