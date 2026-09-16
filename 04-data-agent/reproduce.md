# Reproduce training and evaluation

Use one recipe at a time initially. A training smoke performs **two optimizer steps → save/upload → verified remote restore → two more steps**. It must show exact-token capture, finite updates, changed weights and native optimizer state before a long run is admitted.

## 1. Requirements and credentials

Use Linux, Python 3.12, Git, [`uv`](https://docs.astral.sh/uv/), and an HF account. Local runs need Slurm and two suitable CUDA GPUs on one node; HF Jobs need organization Jobs permissions and GPU quota. The default namespace is `HuggingEnvs`; use `--namespace YOUR_ORG` during preparation to create resources in your own organization.

The frozen task/runtime bundle is public at [HuggingEnvs/data-agent-daytona-repro](https://huggingface.co/datasets/HuggingEnvs/data-agent-daytona-repro). Use the exact archive named in `hf/configs/sources.json`; `hf/build.py --seed-archive PATH` verifies its SHA-256. It never substitutes today's dataset for the measured train/test split. [The public artifact index](https://huggingface.co/datasets/HuggingEnvs/data-agent-experiment-results) links environments, dashboards, results and published checkpoints.

```bash
cd 04-data-agent
uv venv --python 3.12 .venv-launcher
uv pip install --python .venv-launcher/bin/python 'huggingface-hub==1.26.0' python-dotenv
source .venv-launcher/bin/activate
```

Create `.env` (ignored by Git):

```dotenv
HF_API_KEY=your_hf_token
DAYTONA_API_KEY=your_daytona_key
# Optional for a non-default Daytona deployment:
# DAYTONA_API_URL=https://app.daytona.io/api
# DAYTONA_TARGET=eu
```

The HF token needs permission to launch Jobs and manage the selected Spaces and run storage. Public bundles and published results can be read without organization membership. The launcher sends credentials as secrets; they are excluded from the bundle and launch metadata. Live raw artifacts remain private: captured tool output can contain credentials. Public releases use audited copies, with redacted files identified in a manifest. The native adapter also supports HF/E2B sandboxes, but these training recipes select Daytona.

## 2. Select and freeze a recipe

Choose `harbor-multi`, `harbor-opencode`, `native-opencode`, or `seta`. The first two use Harbor; the third uses `envs/blackbox-opencode` directly.

```bash
python reproduce.py prepare --recipe harbor-opencode --env-file .env \
  --run-id my-harbor-opencode-01
```

The default output is `temp/reproduction/harbor-opencode/`. Use the same `--recipe` and, if supplied, `--out` in subsequent commands. Preparing an existing run directory is rejected to preserve its identity. A new run should use a new output directory and run ID.

The preparation pins OpenEnv and TRL commits, Qwen3.5-2B revision, the 1,000 training tasks, the 250 test tasks and schedule hashes. Python and shell paths are relocated inside the bundle. Package versions are frozen in separate environment/training lockfiles.

| Setting | Async Harbor / native OpenCode | Sync SETA |
| --- | --- | --- |
| Model | Qwen3.5-2B, pinned revision | Same |
| Training tasks | 150 easy / 600 medium / 250 hard | Same fixed order |
| Initial curriculum | First 32 tasks easy; then shuffled | Same |
| Learning rate / generations | `3e-6` / 8 | Same |
| Sampling | Temperature 0.8, top-p 1, top-k disabled; thinking off | Same |
| Max context / output per call | 131,072 / 16,384 | Same model limits; tool-loop budget differs |
| Staleness | 4 | Synchronous |
| Backpressure | 32 workers, 16 outstanding rollouts; whole-group admission | Native sync batches |
| Save / eval | Every 50 / 100 optimizer steps | Same |
| Step / wall-clock ceiling | 1,000 / about 23 hours plus checkpoint grace | Same |
| Checkpoint suite | Four harnesses × 250 tests | Native SETA × 250 tests |

The 32-worker ceiling is not a claim that eight generations always run simultaneously. Backpressure, group readiness and provider capacity determine active work. Multi-harness training assigns one harness per task per pass and rotates assignments on later passes. Four thousand task/harness pairs are not four thousand optimizer steps.

## 3A. HF Jobs and Spaces

First upload the prepared bundle. Deploy only into an idle environment or a new owned Space; deployment restarts that Space. Existing ongoing runs must finish before changing their environment.

```bash
python reproduce.py upload --recipe harbor-opencode --env-file .env
python reproduce.py spaces --recipe harbor-opencode --env-file .env
python reproduce.py eval --recipe harbor-opencode --env-file .env --flavor a100-large
python reproduce.py smoke --recipe harbor-opencode --env-file .env --flavor a100x4
python reproduce.py status --recipe harbor-opencode --env-file .env
```

Hub eval uses **TP1/DP1 on one A100 80GB**, concurrency **35**, and the fixed pass@1 cohort. To change concurrency, set `--concurrency` during preparation so the chosen value is frozen. Increasing it does not create more sandbox quota. Training uses separate inference and optimizer GPUs: `h200x2` or `a100x4` (the recipe uses two of the four A100s). SETA's validated HF training allocation is `h200x2`.

For a smoke against an already-deployed Space, skip the deployment command and pass its exact `/deployment` `bundle_sha256`:

```bash
python reproduce.py smoke --recipe seta --env-file .env --flavor h200x2 \
  --space-bundle-sha EXACT_DEPLOYED_SHA256
```

This explicitly records two source identities: the trainer bundle and the existing environment bundle. It does not upgrade the live Space or certify untested server changes. Async training checks the advertised rollout API before allocating a Job; an older native OpenCode server without explicit sampling support must be upgraded while idle. The current qualification preserves the active Harbor/SETA deployments and updates the idle native OpenCode Space.

Once both jobs complete and their evidence passes:

```bash
python reproduce.py train --recipe harbor-opencode --env-file .env \
  --flavor a100x4 --baseline-job BASELINE_JOB_ID --smoke-job SMOKE_JOB_ID
```

The launcher submits a separate CPU coordinator. At steps 100, 200, … it waits for the completed checkpoint manifest, submits a separate A100 eval Job, and verifies the model hash before serving it. Step-50 checkpoints remain available for later evaluations. The final checkpoint is also eligible. A failed or ambiguous eval submission is recorded for reconciliation; it is not blindly duplicated.

Native OpenCode's baseline command measures its **native** 250-task protocol on the configured sandboxes. Its **checkpoint comparisons** use the four-harness Harbor Space. Deploy the Harbor environment as well when reproducing native OpenCode in a new namespace. Keep that Space's task/harness pins fixed. Native and Harbor baseline percentages are different cohorts and must be labelled accordingly.

For native OpenCode, run the shared four-harness baseline with the `harbor-opencode` recipe too (or reuse a completed matching baseline). Long-run admission requires both: the native diagnostic verifies that environment's grading, and the Harbor baseline supplies step 0 of the checkpoint curve.

```bash
python reproduce.py train --recipe native-opencode --env-file .env \
  --baseline-job NATIVE_DIAGNOSTIC_JOB --comparison-baseline-job HARBOR_BASELINE_JOB \
  --smoke-job NATIVE_SMOKE_JOB --flavor a100x4
```

The launcher checks fixed task/model/sampling identity and 250 results for each of the four harnesses. It never places the native diagnostic percentage on the four-harness curve.

## 3B. Local / Slurm

Use the same preparation command. The local launcher runs both the environment service and vLLM inside the allocation. Training uses one inference GPU and one optimizer GPU; eval uses TP1/DP2 on two GPUs. Local eval defaults to concurrency 50.

```bash
python reproduce.py eval --platform local --recipe harbor-opencode --env-file .env \
  --partition YOUR_GPU_PARTITION --submit
python reproduce.py smoke --platform local --recipe harbor-opencode --env-file .env \
  --partition YOUR_GPU_PARTITION --submit
```

Omit `--submit` to inspect generated Slurm scripts first. The allocation creates its own hash-locked venvs. To reuse validated local environments without modifying them, supply both `--train-venv /path/to/train-venv` and `--env-venv /path/to/env-venv`. Use a shared filesystem with room for the model, optimizer states and captures.

Before the long run, qualify the checkpoint controller against the smoke's checkpoint 4. The advanced helper prints the resulting plan:

```bash
python hf/local_long.py --arm blackbox \
  --smoke-run /absolute/path/to/repro/outputs/local-train-blackbox-JOB_ID \
  --baseline-score /absolute/path/to/baseline/canonical_scores.json \
  --env-file .env --out temp/checkpoint-qualification \
  --coordination-dir temp/eval-admission --qualify-checkpoint-eval
python hf/local_followup.py watch --plan temp/checkpoint-qualification/plan.json --submit
```

After the independent checkpoint eval produces `checkpoint_eval_verified.json`:

```bash
python reproduce.py train --platform local --recipe harbor-opencode --env-file .env \
  --partition YOUR_GPU_PARTITION --cpu-partition YOUR_CPU_PARTITION \
  --smoke-run /absolute/path/to/repro/outputs/local-train-blackbox-JOB_ID \
  --baseline-score /absolute/path/to/baseline/canonical_scores.json \
  --checkpoint-eval-proof /absolute/path/to/checkpoint_eval_verified.json --submit
```

Use internal arm `opencode` for native OpenCode and `whitebox` for SETA when invoking advanced helpers. Native long-run admission also checks the frozen task grading/tolerance audit; preserve the `verification.json` next to its canonical baseline score. No recipe uses `hopper-extra` or `hopper-atl` by default. Partition names are explicit deployment settings, not source edits.

For local native training, add `--comparison-baseline-score /absolute/path/to/harbor-baseline/repro/outputs/local-eval-blackbox-JOB_ID/canonical_scores.json` to the qualification and long-run commands. Keep that baseline inside its prepared runtime: admission reads the adjacent frozen configuration to verify the comparison protocol. `--baseline-score` remains the separate native diagnostic. The controller records the matching baseline at step 0 and rejects a modified score file.

## Evaluation protocol and logs

Pass@1 keeps the **first graded** attempt for each task/harness cell, including a zero. Only an ungraded infrastructure failure can be retried. Publication requires complete fixed coverage, exact-token audit, task hashes, harness versions and checkpoint provenance. The test set is 33 easy, 118 medium and 99 hard; overall scores are computed from counts, not an unweighted average of difficulty percentages.

Job outputs contain `status.json`, `training_recipe.json`, `space_identity.json`, `training_smoke_verified.json`, `canonical_scores.json`, capture audits and native checkpoints. Remote artifacts are stored under the run ID and unique job owner. Checkpoints publish their ready marker only after all files and hashes are verified. Bucket transfers retry transient transport, rate-limit and server errors up to three attempts; permission or validation errors fail immediately. Full optimizer checkpoints are larger than inference-only exports, so the smoke includes their upload and restore time. [HF Bucket sync](https://huggingface.co/docs/huggingface_hub/guides/buckets) compares existing content when retrying. Local source transformations have their own manifest; the original portable bundle remains intact.

Training logs locally and uploads Trackio events/databases asynchronously. Environment Spaces do not contain a dashboard. The [shared comparison dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio) replays audited training and eval artifacts; see `hf/consolidate_async_runs.py` for the historical collector and `hf/runtime/logging_sync.py` for per-run logging. A new reproduction keeps its own run identity rather than overwriting these measured runs.

`--dry-run` on the beginner CLI prints the exact command without allocating resources. Never reuse a run directory to silently overwrite a baseline. Superseded files go into the ignored `temp/` archive; credentials, raw traces, checkpoints and SQLite databases are not committed.
