# Training and deployment entry points

- `grpo_nayana.py`: one-GPU LoRA GRPO against a local catalog or OpenEnv URL. `--task-input auto`
  selects the full-corpus block iterator when the server advertises bucket-backed storage.
  `--task-input corpus` requires it explicitly. The evaluation sample is indexed and balanced;
  the full training pass keeps natural language/task proportions within the selected groups.
- `hf_job.py`: fetch a pushed Git commit and use its frozen lockfile. Supports CPU smoke,
  small-window smoke, and training. Pass the full Space with `--env-url` or a bucket manifest
  with `--corpus-manifest`; a job can attach the source bucket at `/corpus`.
- `deploy_space.py`: publish code and a ready full-corpus manifest, remove the old bundled
  preview, and attach the existing bucket read-only. Indexes must already be published.
- `benchmark_corpus.py`: cold random evaluation, warm GRPO-style requests, sustained
  cached capacity, and multi-block prefetch, using the actual image/reward adapter.
- `benchmark_job.py`: run that benchmark at a pushed commit on a CPU Job with a bucket
  mount, returning its complete measured report through the job logs.

See [REPRODUCE.md](../REPRODUCE.md) for exact commands, defaults, and replay boundaries.
The CPU HF Jobs speed test completed; GPU optimizer training remains unverified.
See [SPEED.md](../results/SPEED.md) for timings and the oversized-page readiness blocker.
