# Jobs and Spaces runtime

Use [../reproduce.md](../reproduce.md) and `../reproduce.py` as the entry point.

`build.py` downloads a hash-verified, immutable task bundle and exports pinned OpenEnv/TRL commits. It overlays the implementations in this repository, records every packaged file hash, and archives previous outputs in `../temp/build-archive/`. It does not read a sibling `experiments/` checkout.

`deploy.py` uploads that bundle, deploys an explicitly selected Space, or submits a Job. Jobs create two isolated environments from `locks/`: inference/training and environment serving. Long training requires verified baseline and optimizer/save/remote-resume evidence. The CPU coordinator admits only durably published checkpoints to separate A100 evaluation Jobs.

`cluster.py` stages the same bundle for Slurm. It can create its locked venvs or use two explicitly supplied existing venvs. `local_long.py` and `local_followup.py` qualify checkpoint evaluation before admitting a long run. Use a shared filesystem for local checkpoints and a separate GPU allocation for evals.

Environment Spaces have interactive task/model panels and per-session traces. They do not host Trackio. Training writes local metrics/Trackio artifacts and uploads them asynchronously. The shared comparison dashboard is a separate presentation layer.

Advanced operator scripts retain the migration, checkpoint recovery, intentional-stop and dashboard workflows used by the recorded experiments. They are not prerequisites for the beginner reproduction path. Dated results and provenance are in `../results/`; superseded local notes are preserved in ignored `../temp/`.
