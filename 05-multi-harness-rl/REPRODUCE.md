# Run the tutorial

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Tutorial](README.md)

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

The installer uses **TRL main**, **OpenEnv main** and Transformers main. It refreshes the OpenEnv checkout on each installation and loads the environment examples from that checkout's `envs/` directory. No installed source files are patched. Each run keeps the resolved versions and commits in `dependencies.json`.

TRL's typed OpenEnv integration is merged. Both local and HF Jobs installations use upstream main without a PR checkout. The installer checks the required APIs before starting; [VALIDATION.md](VALIDATION.md) records the versions tested here.

Task preparation pins the corrected train/test dataset revisions and verifies instructions, graders and split separation. It downloads all 1,250 tasks into ignored `prepared/`. Credentials belong in the environment, never in source files. Create a [Daytona account](https://www.daytona.io/) and an HF bucket you can write to before using HF Jobs.

## 2. Submit from a laptop with HF Jobs

The submitter needs only `huggingface_hub`. It uploads the tutorial source, then installs dependencies and prepares tasks inside the job. It mounts an existing bucket at `/outputs` so checkpoints and logs survive the job.

```bash
pip install 'huggingface_hub>=1.29.0'
hf auth login
export DAYTONA_API_KEY="your-key"
export RUN_BUCKET="your-org/smoldataenv-runs"

python jobs/hf_job.py smoke --mode multi_harness \
  --name lfm-harbor-smoke --bucket "$RUN_BUCKET" \
  --timeout 1h
```

This prints the plan. Add `--submit` to launch. Use `h200x2` (default). Current upstream AsyncGRPO selects FlashAttention 3, so this tutorial does not patch in A100 support.

The smoke trains for two updates, saves both checkpoints, stops the trainer, then reloads checkpoint 2 into vLLM for evaluation. Its report is `/outputs/lfm-harbor-smoke/smoke.json`; training and evaluation have separate subfolders.

Repeat with `--mode opencode` and `--mode whitebox`, giving each job a fresh name. For Qwen, add `--model Qwen/Qwen3.5-2B`. A completed two-step smoke establishes execution, not learning. Check `reward`, `grad_norm`, checkpoint files and evaluation coverage. All-equal rewards can legitimately yield zero gradient.

## 3. Run the same commands locally or with Slurm

For a local two-GPU machine, activate the environment and run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python jobs/smoke.py --mode opencode \
  --output runs/opencode-smoke
```

For Slurm, choose your cluster's H100/H200 partition. Submit from this folder after installing dependencies and preparing tasks:

```bash
sbatch --partition=YOUR_PARTITION jobs/train.slurm smoke \
  --mode multi_harness --output runs/harbor-smoke

sbatch --partition=YOUR_PARTITION jobs/train.slurm train \
  --mode multi_harness --steps 100 --output runs/harbor-pilot

# After checkpoint 100 has been saved:
sbatch --partition=YOUR_PARTITION jobs/train.slurm eval \
  --mode multi_harness --checkpoint runs/harbor-pilot/checkpoint-100 \
  --step 100 --tasks 25 --concurrency 35 --output runs/harbor-pilot-eval
```

One allocation owns one trainer and one inference engine. Evaluation gets a **different allocation**, with two inference replicas and no trainer. Shared sandbox quotas and bucket bandwidth still need headroom. The launcher chooses unused local service ports and records them in `services.json`. Keep each allocation on its own assigned GPUs and use a different output directory.

The bounded smoke uses two optimizer updates, two rollouts per group and checkpointing each update. It checks checkpoint contents and reloads checkpoint 2 for two held-out tasks, across four harnesses for blackbox or SETA for whitebox. Normal runs retain eight rollouts per group and the settings in the training scripts.

The launcher starts the actual [environment servers](envs/README.md) inside the allocation. Each folder under `envs/` is a standalone Space with its own Dockerfile, dependencies, data preparation and implementation. That page links the independent local and Hub instructions.

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

Both blackbox policies use four evaluation harnesses: 25 tasks means 100 pairs. Whitebox uses its native tools: 25 tasks means 25 episodes. The [evaluation guide](eval/README.md) maps each training mode to its evaluation interface and shows how to use a deployed environment. The subset is deterministic, taken from the sorted fixed test list. It is not the older pilot's stratified subset, so generate a new baseline. Omit `--tasks` for all 250 tasks.

For a longer run, omit `--steps 100`. Checkpoints are saved every 50 updates. Submit independent evaluations at steps 100, 200, … using the corresponding path and `--step`. The trainer does not wait for them. Checkpoint serving uses vLLM's eager safetensors loader to avoid random reads from bucket and network filesystems. This tutorial keeps submission explicit rather than claiming an untested automatic checkpoint watcher.

## 5. Read the outputs

| Output | Contents |
|---|---|
| `train.log` / `eval.log` | Trainer or evaluator output |
| `vllm.log`, `whitebox.log` / `opencode.log` / `harbor.log` | Inference and environment service logs |
| `training_config.json` | Resolved TRL hyperparameters |
| `dependencies.json`, `packages.txt` | Installed commits and package versions |
| `task_names.json` | Selected training task identities |
| `checkpoint-50/`, `checkpoint-100/`, … | Standard TRL/Transformers training checkpoints |
| `final/` | Final model weights and tokenizer |
| `trackio/` | Local Trackio database or append-only logs on network storage |
| Eval `pairs/*.json` | One result per task/harness, including ungraded errors |
| Eval `summary.json` | Coverage, observed pass@1, reward, tool/token usage, harness and difficulty breakdowns |

Add `--space-id your-org/your-trackio-space` for online Trackio too. Evaluation uses a separate named run and records the checkpoint identity. `pass_at_1_observed` excludes ungraded pairs; always report it together with coverage. A partial evaluation exits with status 2. Rerun the same eval command and output directory to retry missing/ungraded pairs. Graded successes **and graded failures** are retained.

For direct experimentation, start the services once with the settings in [jobs/serve_model.py](jobs/serve_model.py) and [jobs/run.py](jobs/run.py), then run a training file directly. Native OpenCode additionally needs `SANDBOX_VLLM_URL` and `SANDBOX_VLLM_KEY`: the launcher creates an authenticated inference-only tunnel. Harbor uses OpenEnv's capture tunnel. Remote sandboxes cannot reach your machine's localhost directly.

## What differs from the archive?

The three training scripts use public trainers. They do not import custom trainer classes, download archived modules, replace installed source text, or assume that a worker seed is a task index. Training uses the same task lists, learning rate, reward formula, sampling and principal batch settings, but upstream async scheduling/row packing differs from the old custom curriculum. Exact resume, atomic rollout weighting and long-run automatic evaluation need separate qualification before claiming an identical experiment.
