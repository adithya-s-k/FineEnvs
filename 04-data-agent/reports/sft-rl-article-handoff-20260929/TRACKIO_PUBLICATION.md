# Public SFT Trackio publication

Updated 29 September 2026: SFT has moved to its own public CPU-basic Space. The original comparison Space defaults to RL. Prior SFT imports remain archived in its raw database; no records were deleted.

- [SFT dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio)
- [SFT overview](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ctool_call_savings_pct%29%7Ctrain_sft%2Floss_mean50%7Ceval_coverage%2Fgraded_cells%29%24)
- [Training curves and runtime metrics](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28train_sft%7Ctrain_summary%29%2F)
- [Evaluation scores, tools and tokens](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_coverage%7Ceval_usage%7Ceval_efficiency%29%2F)
- [Commit-pinned article citation](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio/blob/fa45c93c5b987dfc51ee9758d2bc3786e5fdd8f0/SFT_RESULTS.md)
- [RL dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio)
- [Local results table](SFT_RESULTS.md)

All 12,806 SFT scalar records were verified by exact public API readback. This includes 12,782 optimizer updates, six runtime summaries and 18 evaluation records including repeated baselines. Dense plotting responses now retain roughly 800 training points per run; every evaluation and summary point remains visible. Raw data remains complete and can be queried or downloaded. The dedicated SFT Space restores its entire snapshot from a versioned archive on startup.

The original Space was RUNNING with HTTP 200 and no crash in the available logs when checked. Unlimited plotting responses were a plausible browser bottleneck, not a confirmed server crash. Separation and bounded plotting reduce the rendering load.

Runtime excludes queue, preprocessing and evaluation. Historical logging timestamps are ingestion times. Partial bash-907 evaluations and overlap-inclusive bash-4,677 evaluations remain flagged. No private rollout text or model weights were published. See TRACKIO_PUBLICATION_ORIGINAL.md for the earlier combined-dashboard publication audit.

RL dashboard UI update: the default page now opens six LFM overview charts, with navigation for Qwen, harness scores and tool/token usage. A real browser check verified the overview, Qwen and harness views with no JavaScript errors. The former unrestricted view rendered 426 canvases; all diagnostics remain accessible under All RL metrics.
