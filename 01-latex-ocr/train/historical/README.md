# Recovered multi-model experiment source

These three files are byte-for-byte copies of Adithya's LaTeX OCR experiment code from
`adithya-s-k/trl_prod` commit `4d82cafc0090d480d856a46bdcca0865434ee932` (2026-07-26).
They are included here so reproduction does not depend on access to the private research repository.
[`results/reproduction-sources.json`](../../results/reproduction-sources.json) records file hashes and
the TRL/OpenEnv submodule revisions stored by that commit.

- `train_ablation.py`: full/LoRA tuning, local OpenEnv server, vLLM colocate/server modes, interval evaluation,
  local JSON/JSONL mirrors, and optional Trackio logging.
- `colocate_5k.slurm`: the four-model full-fine-tuning launch configuration, including unstable Gemma.
- `colocate_5k_stable.slurm`: the stabilized configuration; the retained comparison contains its Gemma run.

The Slurm files contain original cluster paths, scheduler settings, and personal Trackio destinations.
Use the portable invocations in [`REPRODUCE.md`](../../REPRODUCE.md), with fresh output/project names.
The Python source expects the original `experiments/vlm_ocr/ablation/` directory layout alongside `OpenEnv/`.
It uses the legacy OpenEnv client API; it is separate from the maintained `train/grpo_latex_ocr.py` runner.

This recovered source is later than the July 23 metric snapshots. In particular, it closes evaluation
sessions and increases server capacity, with a source comment describing an earlier failure around
step 4,000. It is a recovery/replay reference, not proof of the exact code or full dependency set used
for every archived run. No new full GPU training was performed when recovering these files.
