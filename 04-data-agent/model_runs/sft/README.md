# SFT experiment source

Frozen source used for the two-epoch LFM2.5-2.6B and Qwen3.5-2B teacher-trace experiments. `source_manifest.json` records the original experiment location and file hashes. This preserves the run implementation; cluster launchers still expect the original workspace layout and prepared manifests.

- `prepare_opencode_messages.py` and `prepare_opencode.py`: convert teacher traces for the student tokenizer and retain completion-only supervision.
- `train.py`: TRL SFT, learning rate 3e-6, batch size 1, accumulation 8, two epochs, checkpoint saves every 50 updates and at epoch boundaries.
- `eval_policy.py`, `watch_evals.py`, `evaluate.py`: evaluate completed epochs on the fixed 250-task, four-harness pass@1 protocol; default concurrency 100.
- `local_logging.py`: local Trackio and durable metric records. Dashboard publication is separate from training.
- `launch.py`, `pipeline.py`, `train_entry.py`: historical Slurm launch and orchestration.

The epoch results, overlap caveats, exact training runtime and public dashboard links are in [the article evidence packet](../../reports/sft-rl-article-handoff-20260929/HANDOFF.md). Data and checkpoints remain outside Git. The portable teacher dataset is [FineEnvs/SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft).
