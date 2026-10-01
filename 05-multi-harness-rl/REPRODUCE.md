# Run the tutorial

Start with one mode and a two-step smoke. Check that the checkpoint reloads and the evaluation grades every requested pair before spending time on a longer run.

## 1. Install and prepare the tasks

From the repository root:

```bash
cd 05-multi-harness-rl
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
bash jobs/install.sh
hf auth login
export HF_TOKEN="$(hf auth token)"
export DAYTONA_API_KEY="your-key"
python prepare.py
```

The installer uses **TRL main**, **OpenEnv main** and Transformers main. OpenEnv's environment examples are loaded from the same checkout's `envs/` directory because they are outside its core wheel. No source files are modified. To update an existing checkout, run `git -C .deps/OpenEnv pull --ff-only` and rerun the installer.

As of the rewrite, TRL #6947 is still open. Installation can finish while `check_setup.py` correctly stops on the missing typed consumer. Wait for that integration to merge before a main-only blackbox run. Do not treat the older pinned-runtime results as qualification of this installation.

Task preparation pins the corrected train/test dataset revisions and verifies instructions, graders and split separation. It downloads all 1,250 tasks into ignored `prepared/`. Credentials belong in the environment, never in source files. Create a [Daytona account](https://www.daytona.io/) and an HF bucket you can write to before using HF Jobs.

## 2. Submit from a laptop with HF Jobs

The submitter needs only `huggingface_hub`. It uploads the tutorial source, then installs dependencies and prepares tasks inside the job. It mounts an existing bucket at `/outputs` so checkpoints and logs survive the job.

```bash
pip install 'huggingface_hub>=1.29.0'
hf auth login
export DAYTONA_API_KEY="your-key"
export RUN_BUCKET="your-org/smoldataenv-runs"

python jobs/hf_job.py train --mode multi_harness \
  --name lfm-harbor-smoke --bucket "$RUN_BUCKET" \
  --steps 2 --save-steps 1 --timeout 1h
```

This prints the plan. Add `--submit` to launch. Use `h200x2` (default). Current upstream AsyncGRPO selects FlashAttention 3, so this tutorial does not patch in A100 support.

After the smoke completes, load its checkpoint in a separate evaluation job:

```bash
python jobs/hf_job.py eval --mode multi_harness \
  --name lfm-harbor-smoke-eval --bucket "$RUN_BUCKET" \
  --checkpoint /outputs/lfm-harbor-smoke/checkpoint-2 --step 2 \
  --tasks 2 --concurrency 4 --timeout 1h --submit
```

Repeat with `--mode opencode` and `--mode whitebox`, giving each job a fresh name. For Qwen, add `--model Qwen/Qwen3.5-2B` to **both** train and eval. A completed two-step smoke establishes execution, not learning. Check `reward`, `grad_norm`, checkpoint files and evaluation coverage. All-equal rewards can legitimately yield zero gradient.

## 3. Run the same commands locally or with Slurm

For a local two-GPU machine, activate the environment and run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python jobs/run.py train --mode opencode \
  --steps 2 --save-steps 1 --output runs/opencode-smoke

CUDA_VISIBLE_DEVICES=0,1 python jobs/run.py eval --mode opencode \
  --checkpoint runs/opencode-smoke/checkpoint-2 --step 2 \
  --tasks 2 --concurrency 4 --output runs/opencode-smoke-eval
```

For Slurm, choose your cluster's H100/H200 partition. Submit from this folder after installing dependencies and preparing tasks:

```bash
sbatch --partition=YOUR_PARTITION jobs/train.slurm train \
  --mode multi_harness --steps 100 --output runs/harbor-pilot

# After checkpoint 100 has been saved:
sbatch --partition=YOUR_PARTITION jobs/train.slurm eval \
  --mode multi_harness --checkpoint runs/harbor-pilot/checkpoint-100 \
  --step 100 --tasks 25 --concurrency 35 --output runs/harbor-pilot-eval
```

One allocation owns one trainer and one inference engine. Evaluation gets a **different allocation**, with two inference replicas and no trainer. Shared sandbox quotas and bucket bandwidth still need headroom. The launcher chooses unused local service ports and records them in `services.json`. Keep each allocation on its own assigned GPUs and use a different output directory.

## 4. Compare baseline with checkpoint 100

Use the same 25 fixed test tasks and interfaces for both measurements. Start an evaluation job without `--checkpoint` for the base model, then train for 100 updates, then evaluate `checkpoint-100`.

```bash
python jobs/hf_job.py eval --mode multi_harness --tasks 25 \
  --name lfm-baseline --bucket "$RUN_BUCKET" --submit

python jobs/hf_job.py train --mode multi_harness --steps 100 \
  --name lfm-harbor-100 --bucket "$RUN_BUCKET" --submit

# Submit this only after checkpoint-100 has finished saving.
python jobs/hf_job.py eval --mode multi_harness --tasks 25 \
  --checkpoint /outputs/lfm-harbor-100/checkpoint-100 --step 100 \
  --name lfm-harbor-100-eval --bucket "$RUN_BUCKET" --submit
```

Both blackbox policies use four evaluation harnesses: 25 tasks means 100 pairs. Whitebox uses its native tools: 25 tasks means 25 episodes. The subset is deterministic, taken from the sorted fixed test list. It is not the older pilot's stratified subset, so generate a new baseline. Omit `--tasks` for all 250 tasks.

For a longer run, omit `--steps 100`. Checkpoints are saved every 50 updates. Submit independent evaluations at steps 100, 200, … using the corresponding path and `--step`. The trainer does not wait for them. This tutorial keeps submission explicit rather than claiming an untested automatic checkpoint watcher.

## 5. Read the outputs

| Output | Contents |
|---|---|
| `train.log` / `eval.log` | Trainer or evaluator output |
| `vllm.log`, `harbor.log` | Inference and environment service logs |
| `training_config.json` | Resolved TRL hyperparameters |
| `task_names.json` | Selected training task identities |
| `checkpoint-50/`, `checkpoint-100/`, … | Standard TRL/Transformers training checkpoints |
| `final/` | Final model weights and tokenizer |
| `trackio/` | Local Trackio database |
| Eval `pairs/*.json` | One result per task/harness, including ungraded errors |
| Eval `summary.json` | Coverage, observed pass@1, reward, tool/token usage, harness and difficulty breakdowns |

Add `--space-id your-org/your-trackio-space` for online Trackio too. Evaluation uses a separate named run and records the checkpoint identity. `pass_at_1_observed` excludes ungraded pairs; always report it together with coverage. A partial evaluation exits with status 2. Rerun the same eval command and output directory to retry missing/ungraded pairs. Graded successes **and graded failures** are retained.

For direct experimentation, start the services once with the settings in [jobs/serve_model.py](jobs/serve_model.py) and [jobs/run.py](jobs/run.py), then run a training file directly. Native OpenCode additionally needs `SANDBOX_VLLM_URL` and `SANDBOX_VLLM_KEY`: the launcher creates an authenticated inference-only tunnel. Harbor uses OpenEnv's capture tunnel. Remote sandboxes cannot reach your machine's localhost directly.

## What differs from the archive?

The three training scripts use public trainers. They do not import custom trainer classes, download archived modules, replace installed source text, or assume that a worker seed is a task index. Training uses the same task lists, learning rate, reward formula, sampling and principal batch settings, but upstream async scheduling/row packing differs from the old custom curriculum. Exact resume, atomic rollout weighting and long-run automatic evaluation need separate qualification before claiming an identical experiment.
