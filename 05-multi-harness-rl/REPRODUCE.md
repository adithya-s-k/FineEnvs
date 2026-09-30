# Run SmolDataEnv RL

Start from a checkout containing this example:

```bash
cd 05-multi-harness-rl
```

Pick **HF Jobs** if you want the container to install and prepare everything. Pick **local/Slurm** if you already have a compatible GPU machine. Both use the same code, fixed tasks and model revisions. The current branch is local until published; the commands assume you already have its files.

## 1. What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An HF account with Jobs billing enabled, or access to a local GPU machine.
- A Daytona account and API key for task sandboxes. GPU and sandbox usage are billed separately.
- Two GPUs with enough memory: the qualified HF blackbox setup uses A100 80 GB; local smokes used H100s. The default HF request is H200×2. Other combinations need their own smoke.

One GPU trains while the other serves rollouts. Evaluation uses both as independent inference replicas. HF offers A100×4 rather than A100×2; this recipe uses two GPUs but the entire allocation is billed.

## 2. Submit from a CPU laptop with HF Jobs

You do **not** need CUDA, PyTorch, the datasets or a local environment server for submission.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install huggingface_hub==1.24.0
hf auth login

# Replace this with your own HF username or organization.
export HF_NAMESPACE="your-hf-username"
export HF_BUCKET="$HF_NAMESPACE/smoldataenv-runs"
hf buckets create "$HF_BUCKET" --private --exist-ok

# Enter your Daytona key without putting it in shell history.
read -rs -p "Daytona API key: " DAYTONA_API_KEY
export DAYTONA_API_KEY
```

Your HF login needs access to submit Jobs and write the bucket. The launcher uses the bucket owner as the job namespace; pass `--namespace` to choose another account you can use. Tokens are sent as HF Job secrets, not embedded in the uploaded source. Keep credentials out of config files.

### First: verify one small run

Commands print a submission plan by default. Read it, then add `--submit` to launch.

```bash
python runtime/launch.py hf smoke --model lfm --mode opencode \
  --run-name smoke-lfm-opencode-v1 --bucket "$HF_BUCKET" \
  --flavor a100x4 --timeout 45m --smoke-eval --concurrency 4
```

The job creates a Python environment, installs locked dependencies, fetches the pinned runtime and prepares tasks. It then runs two small training updates, saves checkpoints 1 and 2, and reloads checkpoint 2 for evaluation. Blackbox modes evaluate two tasks through each of four Harbor harnesses; whitebox evaluates two native SETA episodes. Require complete grading and valid capture, not a particular score from this tiny sample.

### Next: a 100-step pilot

```bash
python runtime/launch.py hf pilot --model lfm --mode multi-harness \
  --run-name pilot-lfm-multi-v1 --bucket "$HF_BUCKET" \
  --flavor a100x4 --timeout 12h --limit 25 --concurrency 35
```

Add `--submit` after checking the plan. The phases are:

1. Baseline pass@1 on 25 fixed tasks: 3 easy, 12 medium and 10 hard.
2. Train with the normal eight-rollout configuration for 100 updates, saving at 50 and 100.
3. Hash and reload checkpoint 100; repeat the same evaluation.
4. Write `comparison.json` with quality, tool/token usage and gradient diagnostics.

Phases run sequentially in one allocation. Evaluation does not compete with training. An incomplete baseline stops the job before training. The 12-hour timeout is a spending/time cap, not a completion estimate.

Change `--mode` to `opencode` or `whitebox` and choose a **new run name** for each comparison. Add `--preflight-smoke` if that mode/model/hardware combination has not passed the small training and reload test; the pilot proceeds only after it passes. Change `--model lfm` to `--model qwen` to use Qwen3.5-2B. Use `--limit 250` for the full held-out set.

## 3. Monitor and read the results

Submission prints a job URL and records it in `runs/submissions/RUN_NAME.json`.

```bash
hf jobs inspect JOB_ID --namespace "$HF_NAMESPACE"
hf jobs logs JOB_ID --namespace "$HF_NAMESPACE" --tail 50
hf jobs logs JOB_ID --namespace "$HF_NAMESPACE" --follow
```

Logs show phase transitions, training step/reward/gradient norm and evaluation coverage. A job marked `RUNNING` may still be installing packages, loading a model or running its baseline; it does not necessarily mean training has begun. Inspect the final job status as well as the logs.

Outputs are under `/outputs/RUN_NAME` inside the job, backed by your bucket. Afterward, download a run:

```bash
hf buckets sync "hf://buckets/$HF_BUCKET/RUN_NAME" runs/RUN_NAME
```

```text
RUN_NAME/
├── pilot.json                 configuration, exact test IDs and phase status
├── baseline/eval/             summary.json and one result per task/harness
├── train/
│   ├── metrics.jsonl          raw training scalars
│   ├── checkpoint-50/
│   ├── checkpoint-100/        weights, optimizer, RNG and recipe
│   ├── rollouts/              blackbox correctness and tool-count evidence
│   ├── audit/                 async group admission and consumed-rollout records
│   └── trackio/               local Trackio data
├── checkpoint-100/eval/       final held-out results
└── comparison.json           baseline-to-checkpoint comparison
```

Each phase also keeps its service logs and resolved `config.json`. Whitebox keeps `token-audit.jsonl` and `whitebox-records/`; Harbor keeps native trajectories in `trials/`. OpenEnv/vLLM startup URLs are in `services.json`. A preflight, when enabled, has its own directory.

Check these before calling a pilot successful:

| Evidence | Interpretation |
|---|---|
| Complete evaluation coverage | All intended pairs graded; missing infrastructure results are not incorrect answers |
| Nonzero gradient updates | At least some groups supplied learning signal; zero contrast yields zero policy gradient |
| Baseline vs checkpoint pass@1 | Held-out correctness, not shaped training reward |
| Matched-success tool/token savings | Usage changes on pairs solved by both models; report the matched-pair count |
| First/last 20-update reward means | A training diagnostic; async row weighting differs from rollout-level correctness |

A 25-task pilot is noisy. A higher reward or lower tool count alone does not establish better answer quality. Incorrect answers are retained as zero; only ungraded failures are retried. TiTO checks validate token/logprob/mask consistency, not benchmark performance.

### Optional online Trackio

Offline Trackio and scalar logs are always kept. To mirror charts, create a small config file under `configs/`:

```json
{
  "project": "my-smoldataenv-pilot",
  "trackio_space_id": "your-hf-username/smoldataenv-trackio"
}
```

Pass `--config configs/my-run.json`. Your token needs permission to create/update that Space. Config files override [defaults](configs/default.json), so you only need the settings you want to change. Explicit CLI model, mode and concurrency options take precedence. Inspect the resolved settings with `python run.py plan --config configs/my-run.json`.

## 4. Run locally or on Slurm

On the GPU machine, install the full runtime. Keep the source, Python environment, task files and output directory accessible from the allocated node.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
python runtime/bootstrap.py
uv pip install --no-deps --no-build-isolation -e .runtime/trl
hf auth login
# Export DAYTONA_API_KEY as in the HF setup above.
python prepare.py
python -m pytest -q tests
```

The editable TRL installation supplies metadata needed for checkpoint model cards; it does not change the locked dependencies. `prepare.py` checks task instructions, corrected graders, train/test separation and dataset revisions.

On a local two-GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py pilot --model lfm --mode multi-harness \
  --run-name pilot-lfm-multi-local --limit 25 --concurrency 35
```

On Slurm:

```bash
python runtime/launch.py slurm pilot --partition YOUR_GPU_PARTITION \
  --model lfm --mode multi-harness --run-name pilot-lfm-multi-slurm \
  --timeout 12h --limit 25 --concurrency 35
```

Add `--submit` to submit the printed `sbatch` command. It requests one node, two GPUs, 16 CPUs, 192 GB RAM and the time limit you specified. Configure your cluster account/QoS as usual. Use `squeue -u "$USER"` and the printed log path to follow the job. No Slurm job is needed when you use HF Jobs.

## 5. Resume or scale up

To resume a pilot, keep its name, configuration and evaluation limit. Point to the last complete training checkpoint:

```bash
python runtime/launch.py hf pilot --model lfm --mode multi-harness \
  --run-name pilot-lfm-multi-v1 --bucket "$HF_BUCKET" \
  --flavor a100x4 --timeout 12h --limit 25 --concurrency 35 \
  --resume /outputs/pilot-lfm-multi-v1/train/checkpoint-50
```

Keep `--preflight-smoke` if the original pilot used it. Completed phases and graded eval pairs are retained. For Slurm, use the checkpoint's absolute shared-filesystem path. Async resume restores committed group IDs; unconsumed tails of partially committed groups are recorded and abandoned. A pilot that fails before saving a checkpoint needs a fresh training attempt.

For a longer run, use `train` instead of `pilot`. The default ceiling is 1,000 updates, with saves every 50 and evaluations every 100 plus the final checkpoint. A separate CPU watcher launches evaluation GPU jobs:

```bash
python runtime/launch.py hf train --model lfm --mode multi-harness \
  --run-name lfm-multi-full-v1 --bucket "$HF_BUCKET" --timeout 24h
python runtime/launch.py hf watch --run-name eval-watcher \
  --bucket "$HF_BUCKET" --concurrency 35 --max-active-evals 1 --timeout 24h
```

Both are dry runs until `--submit` is added. Full evaluation uses all 250 tasks. Keep the watcher alive until final evals finish. The HF pilot qualifies same-job reload; separate-job live bucket handoff is still an additional qualification item in [VALIDATION.md](VALIDATION.md). Do not run the watcher for pilots.

For Slurm, run `python eval/watch.py --root runs --backend slurm --partition YOUR_GPU_PARTITION --concurrency 35 --max-active 1 --submit` on a persistent process. Failed evaluations remain in its ledger for investigation; it does not resubmit forever.

## Troubleshooting

| Symptom | First check |
|---|---|
| `SCHEDULING` | Hardware allocation or image pull; training has not started |
| Job exits during startup | HF logs, then phase `vllm.log` and `environment.log` |
| Rollouts are ungraded | Sandbox/provider limits, capture tunnel and verifier logs; keep errors separate from wrong answers |
| Reward and gradient remain zero | Per-rollout grades and group contrast; do not assume the optimizer is broken |
| Existing-output error | Use a new run name, or resume a complete checkpoint from the same run |
| Whitebox model-card save fails | Confirm the pinned TRL checkout was installed as above |
| New GPU/kernel combination fails | Run a small smoke first; avoid switching to attention that loses packed-sequence boundaries |

The code fixes the data protocol and supports two pinned models and Daytona sandboxes. Adding a different dataset, model or provider requires adapting the task/template/backend boundary and repeating qualification. It is not a drop-in recipe for arbitrary models.

References: [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), [bucket volumes](https://huggingface.co/docs/hub/jobs-configuration), [Daytona](https://www.daytona.io/docs/).
