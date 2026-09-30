# Run the recipe

All commands below start in `05-multi-harness-rl/`. Python 3.12 is required. Choose `--mode whitebox`, `--mode opencode` or `--mode multi-harness`, and `--model lfm` or `--model qwen`.

## 1. Prepare and inspect

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
python runtime/bootstrap.py
python prepare.py
python -m pytest -q tests
python run.py plan --model lfm --mode multi-harness
```

`prepare.py` downloads immutable revisions of the corrected Harbor train/test datasets. It selects the exact IDs in `data/`, checks instruction and grader hashes, sets each task to one CPU/4 GB, and records file hashes. It never selects a fresh random test set. Data and downloaded runtime code are ignored by Git.

Log in with `hf auth login` and set `DAYTONA_API_KEY` in your shell. Do not put credentials into a config or commit an `.env` file. The job-local service stages private task data; model-serving access is scoped to capture sessions.

## 2. HF Jobs

Create a durable bucket you own. Supply its `owner/name` below. Launch commands print a plan unless you add `--submit`; neither Git push nor an environment Space is required. The launcher uploads only the allowlisted recipe source.

```bash
# Disposable two-update training smoke.
python runtime/launch.py hf smoke --model lfm --mode opencode \
  --run-name smoke-lfm-native --bucket FineEnvs/YOUR_BUCKET --timeout 2h

# Inspect, then add --submit to launch.
python runtime/launch.py hf train --model lfm --mode multi-harness \
  --run-name lfm-multi-nonthinking-v1 --bucket FineEnvs/YOUR_BUCKET

# Separate CPU watcher submits evaluations on separate GPU Jobs.
python runtime/launch.py hf watch --run-name eval-watcher \
  --bucket FineEnvs/YOUR_BUCKET --concurrency 35 --max-active-evals 1
```

Repeat the smoke for each model/mode before launching that combination. The default GPU flavor is `h200x2`: one trainer GPU and one serving GPU; evaluation uses both GPUs as DP2 replicas. HF Jobs currently has no two-A100 flavor. Override `--flavor` only with hardware that has enough GPUs and memory. The base image is pinned by digest; Python packages and source revisions are pinned separately.

The A100 alternative is `--flavor a100x4`; this recipe uses two of those four GPUs, but the full allocation is billed. Async training selects FlashAttention 2 on Ampere and FlashAttention 3 on Hopper, preserving packed-sequence boundaries. A checked adapter enables this choice in the pinned TRL runtime. See `VALIDATION.md` for hardware qualification status.

The bucket is mounted at `/outputs`. A run writes to `/outputs/RUN_NAME`; the watcher hashes completed checkpoints before launching evals. Keep the watcher alive until all final evaluations finish. Failures remain in `eval-watcher.json` for investigation rather than resubmitting indefinitely. Relaunch a failed evaluation explicitly with the same output name to retain already graded pairs.

```bash
python runtime/launch.py hf eval --model lfm --mode multi-harness \
  --run-name lfm-multi-nonthinking-v1-eval-checkpoint-100 \
  --checkpoint /outputs/lfm-multi-nonthinking-v1/checkpoint-100 \
  --bucket FineEnvs/YOUR_BUCKET --concurrency 35
```

For online Trackio, set `trackio_space_id` in `configs/default.json` before submission. Logs and offline Trackio data remain in the bucket regardless. Use `hf jobs ps` and `hf jobs logs JOB_ID` to monitor. The new HF container path still needs an actual GPU Job smoke; a successful local smoke alone does not qualify it.

## 3. Slurm or a local two-GPU machine

The same Python environment and prepared files must be visible on compute nodes. Choose a free GPU partition; no cluster name is hardcoded into the recipe.

```bash
python runtime/launch.py slurm smoke --partition YOUR_GPU_PARTITION \
  --model lfm --mode opencode --run-name smoke-lfm-native

python runtime/launch.py slurm train --partition YOUR_GPU_PARTITION \
  --model qwen --mode multi-harness --run-name qwen-multi-v1

# On a persistent CPU/login process, submit separate GPU eval allocations.
python eval/watch.py --root runs --backend slurm --partition YOUR_GPU_PARTITION \
  --concurrency 50 --max-active 1 --submit
```

Add `--submit` to the launch commands after inspecting them. They request one node, two GPUs, 16 CPUs and 192 GB RAM. Local execution without Slurm is `CUDA_VISIBLE_DEVICES=0,1 python run.py smoke --model lfm --mode opencode`.

Each job starts its own vLLM and OpenEnv services on unused local ports, and stops its own process groups on exit. Their URLs are recorded in `services.json`. The capture proxy uses OpenEnv's Gradio tunnel so remote Daytona containers can reach it. Sandbox egress and tunnel reliability must be checked in the rollout smoke.

## 4. Checkpoint and evaluation checks

The smoke saves at updates 1 and 2. Run the watcher with `--once` without `--submit` to verify checkpoint files and inspect eval commands. Then reload a smoke checkpoint for a two-task evaluation:

```bash
python eval/watch.py --root runs --backend slurm --partition YOUR_GPU_PARTITION --once
python run.py eval --model lfm --mode opencode \
  --checkpoint runs/smoke-lfm-native/checkpoint-2 \
  --output runs/smoke-lfm-native-eval --limit 2 --concurrency 2
```

That is eight pass@1 pairs for a blackbox policy, or two SETA episodes for whitebox. Require a complete summary, valid captured tokens/logprobs/masks, verified tool counts, finite training metrics and a loadable checkpoint. Inspect `train.log`, `environment.log` and `vllm.log` when a check fails.

Production saves every 50 updates; the watcher evaluates every 100 and the final checkpoint. Full blackbox eval is 1,000 pairs. An incorrect answer is retained as zero; only ungraded failures are retried. Incomplete evaluations are explicitly provisional. Do not retry completed incorrect answers and call the result pass@1.

Resume with `--resume /path/to/checkpoint-N`. Blackbox checkpoints preserve committed group IDs, so groups already used by the optimizer are not replayed. Unconsumed tails of partially committed groups are recorded and abandoned; groups with no committed work can be regenerated. Keep the task schedule, model and reward configuration fixed across a resume.

References: [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), [volume configuration](https://huggingface.co/docs/hub/jobs-configuration), [OpenCode CLI](https://dev.opencode.ai/docs/cli/).
