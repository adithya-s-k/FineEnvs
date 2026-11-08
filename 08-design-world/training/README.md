# Training

Supervised fine-tuning and RL of Qwen vision-language models on DesignGym (`envs/designgym`). Main models:
`Qwen/Qwen3.6-35B-A3B` (MoE) and `Qwen/Qwen3.8-27B` (dense).

Every trainer lays out prompts exactly as the evaluation harness (`envs/designgym/rollout.py`) does: the model's own
chat template with the tool list as bare function objects, thinking off, GUI coordinates on a 0-1000 grid, tool images
in a user turn after the tool results, and an 8/3 screenshot window (past 8 screenshots, all but the newest 3 become
`[earlier screenshot omitted]`; the target image never drops). A model trained here is evaluated on the layout it
learned.

```
sft/          SFT with TRL (LoRA) on the released SFT data, FineEnvs/designworld-sft
rl/
  common/     shared rollout core for Miles and SkyRL: OpenEnv client, Qwen token segments, episode loop, HF Jobs launcher
  miles/      Miles GRPO (SGLang + Megatron-Bridge, LoRA)
  skyrl/      SkyRL GRPO (vLLM + Megatron, LoRA)
  tinker/     Tinker RL and evaluation (openenv_rl adapter, DesignGym glue, HF Jobs launchers)
```

## SFT (`sft/`)

TRL `SFTTrainer` with a PEFT LoRA (rank 32) on the released SFT data (`FineEnvs/designworld-sft`: 16 subsets, GUI
recordings in the editors plus tool-call and HTML trajectories, in the `cua-v1` schema of `designgym.cua_format`).
The data is released as is; its generation pipeline is not part of this repo. Each trajectory is cut into windows
with `cua_format.windows()`, laid out by `sft_format.py`, and `collator.py` puts loss on the trainable assistant
turns only; `check_format.py` checks offline that every trained turn is exactly what evaluation would prompt and
expect.

```bash
cd sft
python check_format.py /path/to/a/local/copy/of/the/data          # offline format checks (processor files only)
python launch.py sft --flavor h200 --name smoke -- --max-steps 20 --save-steps 10 --grad-accum 4
python launch.py sft --model Qwen/Qwen3.8-27B --flavor h200x4 --timeout 8h --name big-27b -- --max-steps 700 --hours 7.3
python launch.py merge --adapter <namespace>/designgym-sft-qwen3.6-35b-a3b-smoke --checkpoint checkpoint-10
python launch.py logs <job_id> --follow
```

`launch.py` bundles this folder (with `designgym/cua_format.py`) to a runs dataset, starts `sft_job.sh` on HF Jobs
(`torchrun train.py`), and pushes checkpoints to `<namespace>/designgym-sft-<model>-<name>`. `merge` folds a
checkpoint's LoRA into the base weights (`merge_lora.py`, a CPU job) so vLLM can serve it for evaluation.

## RL

- **Tinker** (`rl/tinker/`): sampling and LoRA training run on Tinker; the rollout loop and the DesignGym server run
  on a CPU machine (`train_designgym.py`) or a CPU HF Job (`hf_jobs/launch.py rl`). `eval_designgym.py`
  scores base models and `tinker://` checkpoints; `hf_jobs/launch.py export` / `eval` turn a checkpoint into a Hub
  model and evaluate it with vLLM + `rollout.py`. See `rl/tinker/README.md`.
- **Miles** and **SkyRL** (`rl/miles/`, `rl/skyrl/`): one GPU node per run, DesignGym running inside the job, both
  driven by the shared core in `rl/common/` (token-in/token-out, image window -> trainable chunks, reward from the
  env at submit). Launch from this folder:

  ```bash
  python -m rl.common.hf_jobs launch miles --flavor h200x8 --timeout 2h --name smoke-35b \
      --model Qwen/Qwen3.6-35B-A3B --tasks 4 -- --gpus 8 --rollout-batch-size 4 --group-size 4 --max-turns 20
  python -m rl.common.hf_jobs launch skyrl --flavor h200x8 --timeout 2h --name smoke-27b --model Qwen/Qwen3.8-27B ...
  ```

  See `rl/README.md` for the layout, the per-framework settings and smoke results.

## Credentials and repos

Launchers read `HF_TOKEN` (and `TINKER_API_KEY` for Tinker) from the environment, else from the nearest `.env` up the
tree, and pass them to jobs as secrets; they are never printed. Jobs run under `--namespace` (default
`$HF_JOBS_NAMESPACE`, else the token's user). Code bundles and results go to datasets you own: `--runs-repo` /
`$DESIGNWORLD_SFT_RUNS_REPO` (SFT, default `<namespace>/designworld-sft-runs`), `--runs-repo` / `$RL_RUNS_REPO`
(Miles / SkyRL, default `<namespace>/designworld-rl-runs`) and `--results-repo` / `$DESIGNWORLD_RESULTS_REPO`
(Tinker RL and evals, default `<namespace>/designworld-evals`). Episodes can run against the public DesignGym Space
(`FineEnvs/DesignGym`, `https://fineenvs-designgym.hf.space`) or a server started in the job.
