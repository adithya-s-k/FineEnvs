# Public SFT Trackio publication

Published on 2026-09-29T11:36:24.266400+00:00. All six SFT runs now share the existing `data-agent-rl-comparison` project with the RL runs.

- [SFT overview](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=7e03598b74c5944b42002d1759e0184c%2Cd189e77a4f134099c308ed223b840405%2C2419d3a7bf21850a46903d5df20f3d59%2C60cde9f411f365b5c634ee3de1351505%2Ca9a405588dd8646aea613c02bf479c8b%2C736a2514f326ea6119699671e61bb0ef&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ctool_call_savings_pct%29%7Ctrain_sft%2Floss_mean50%7Ceval_coverage%2Fgraded_cells%29%24)
- [Training curves and runtime metrics](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=7e03598b74c5944b42002d1759e0184c%2Cd189e77a4f134099c308ed223b840405%2C2419d3a7bf21850a46903d5df20f3d59%2C60cde9f411f365b5c634ee3de1351505%2Ca9a405588dd8646aea613c02bf479c8b%2C736a2514f326ea6119699671e61bb0ef&smoothing=0&metric_filter=%5E%28train_sft%7Ctrain_summary%29%2F)
- [Evaluation scores, coverage, tools and tokens](https://fineenvs-data-agent-training-comparison-trackio.hf.space/?project=data-agent-rl-comparison&run_ids=7e03598b74c5944b42002d1759e0184c%2Cd189e77a4f134099c308ed223b840405%2C2419d3a7bf21850a46903d5df20f3d59%2C60cde9f411f365b5c634ee3de1351505%2Ca9a405588dd8646aea613c02bf479c8b%2C736a2514f326ea6119699671e61bb0ef&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_coverage%7Ceval_usage%7Ceval_efficiency%29%2F)
- [Commit-pinned article citation](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio/blob/525055784fdd7b8a4aeede4255ee7a8119bedb31/SFT_RESULTS.md)
- [Local runtime and evaluation table](SFT_RESULTS.md)

Exact readback verified 12,806 additive SFT events: 12,782 optimizer updates, six runtime summaries and 18 evaluation records including repeated pretrained baselines. All 11,643 preexisting RL scalar records were preserved unchanged. Source hashes are included in the public exports.

The original SFT runs logged locally. This publication backfilled aggregate scalar measurements; dashboard timestamps indicate ingestion time, not original training time. No rollout text or model weights were uploaded. Runtime excludes queue, preparation and evaluation. The partial bash-907 evaluations and overlap-inclusive bash-4,677 results remain explicitly flagged.
