# Training and evaluation logging

The durable scalar source is `audit/metrics.jsonl` for async training and `run/metrics.jsonl` for SETA. Optimizer steps are the horizontal axis. Raw captures and token/mask audits are stored separately.

`hf/runtime/logging_sync.py` replays events into Trackio in a separate process. Deterministic log IDs prevent duplicates after a restart. It writes `trackio-events.jsonl`, a consistent `trackio-backup/` database snapshot and `trackio_verified.json`. The asynchronous artifact publisher uploads these alongside run metadata; network requests do not execute in the optimizer callback.

The default reproduction uses local Trackio plus remote artifact storage. A public dashboard is a separate presentation service, not part of an environment Space. The recorded runs use the [shared comparison dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio); `hf/consolidate_async_runs.py` contains the historical collector. Use a distinct project/run identity for a new reproduction.

For offline viewing, download a verified database backup, copy it to a local writable directory, and point `TRACKIO_DIR` there before running `trackio show --project PROJECT_NAME`. Avoid a writable SQLite database on a shared network filesystem. The pinned Trackio version is 0.33.0.

Completed checkpoint scores are replayed only after fixed coverage, TiTO, harness-version and checkpoint checks. Evaluations that finish after training remain in the coordinator's score artifacts for later replay. The scalar payload includes reward, loss, gradient norm, staleness, token throughput, fork/row counts and pass@1 by harness/difficulty. Training reward and held-out pass@1 remain separate metrics.

The current frozen results and full metrics snapshot are in [../results.md](../results.md). Historical operator notes and retired dashboard repair instructions are preserved locally under ignored `temp/historical-notes/`.
