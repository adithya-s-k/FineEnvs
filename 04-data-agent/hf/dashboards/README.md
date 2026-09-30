# SmolDataEnv dashboards

Pinned source snapshots of the public [SmolDataEnv RL](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) and [SmolDataEnv SFT](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio) Spaces. Each `source_manifest.json` records the deployed revision and source hashes. `SPACE_README.md` preserves the deployed Space card; historical links in that card resolve relative to the original Space.

Both use CPU-basic and Trackio 0.33.0. RL opens a focused six-chart LFM overview, with Qwen and diagnostic navigation. Dense training plots retain roughly 800 points per run; sparse evaluations remain intact. Raw database queries preserve all records.

The RL runtime needs its existing `TRACKIO_DIR=/data/trackio` and `TRACKIO_BUCKET_ID=FineEnvs/data-agent-training-comparison-trackio`, with bucket access configured as a Space secret. No credentials are included here. The SFT runtime restores its versioned `sft_events.json.gz` archive automatically; this small aggregate scalar archive is intentionally included, unlike raw traces, model weights or live SQLite databases.

See [the article evidence packet](../../reports/sft-rl-article-handoff-20260929/HANDOFF.md) for scores, runtime definitions and overlap/coverage caveats.
