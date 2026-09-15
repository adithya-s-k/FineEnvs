# Training and deployment entry points

- `grpo_nayana.py`: one-GPU LoRA GRPO against a local snapshot or an OpenEnv URL.
  The notebook calls this same runner. It validates held-out coverage before loading weights.
- `hf_job.py`: fetch a pushed Git commit, prepare a bounded window, run CPU smoke or training
  with the frozen dependency lock, optionally save results to an explicit personal dataset repo.
- `deploy_space.py`: stage the environment and a selected real-data snapshot into an explicitly
  named Docker Space. Does not create a collection, model, or results bucket.

See [REPRODUCE.md](../REPRODUCE.md) for commands, defaults, checkpoint boundaries, and status.
The scripts are implemented; no GPU training or HF Job result has been recorded for 05 yet.
