---
title: SmolDataEnv SFT
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# SmolDataEnv SFT

Six SFT runs, two models and epoch-by-epoch evaluation. This public CPU-basic Space is separate from the [RL dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio).

[Overview](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ctool_call_savings_pct%29%7Ctrain_sft%2Floss_mean50%7Ceval_coverage%2Fgraded_cells%29%24) · [Training](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28train_sft%7Ctrain_summary%29%2F) · [Evaluations](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_coverage%7Ceval_usage%7Ceval_efficiency%29%2F)

[Training runtimes, epoch scores and article citation data](SFT_RESULTS.md) · [CSV](sft_training_summary.csv) · [Evaluation breakdowns](sft_evaluations.json)

Includes training loss, token accuracy, gradients, runtime and throughput; pass@1 by harness and difficulty; tool savings and token usage. Keep smoothing at 0 for already-smoothed loss curves. Dense plots retain roughly 800 training points per run; all evaluation points are kept. The full scalar archive and raw database retain every update.

Bash 907 evaluations are incomplete (999 and 850 of 1,000). Bash 4,677 includes test overlap. Other SFT epoch evaluations are complete. SFT uses imitation, not an efficiency reward. Qwen RL used correctness only; LFM RL used correctness plus efficiency. See the results document for definitions and denominators.

Data is restored from the versioned scalar archive on startup, so a restart cannot lose this historical snapshot. No private traces or model weights are hosted here.
