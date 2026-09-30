# HF hard-task qualification — 2026-09-17

**Status: production training running at step 805, checkpoint 800 published. Checkpoint-600 evaluation is impaired by failures/timeouts; 700 and 800 await evaluation.**

- [Previous smoke: 6aac2255b1dc2b62dc58ef92](https://huggingface.co/jobs/HuggingEnvs/6aac2255b1dc2b62dc58ef92), `h200x2`, two-hour limit.
- Completed smoke: [6aac2ec9b1dc2b62dc58f0ba](https://huggingface.co/jobs/HuggingEnvs/6aac2ec9b1dc2b62dc58f0ba), bundle `hf-jobs-v11`, SHA-256 `cc4e3801f9db9958fb52a24ef1c460391a896b2c6cb1737721769f03b23f021a`.
- Local validation: **28 distinct tests passed** (26-test combined suite plus two coordinator tests). Full pytest output is retained.
- [H200 kernel diagnostic](https://huggingface.co/jobs/HuggingEnvs/6aac2120b1dc2b62dc58ef67): **passed**. TileLang output/gradient RMS differences were 0.26–0.42% against the BF16 Torch reference at lengths 63, 256 and 2048.

## Run being qualified

Continue checkpoint 500 on 500 screened hard tasks for two task passes. Preserve parent model/optimizer state and start the new task cursor at zero. Four harnesses, eight generations per group, LR 3e-6, staleness 4, 16 outstanding rollouts and a 32-worker ceiling. Save every 50 updates; evaluate every 100 and at the end in separate GPU jobs.

Each GPU job runs its own OpenEnv/Harbor server and capture proxy. Sandboxes run on E2B. Environment Spaces and other experiments are unchanged. The full parent checkpoint is 11,307,363,317 bytes across 14 files; uploads and restores verify file hashes.

## Fixes and checks

- Fixed the frozen trainer CLI signature and finite-worker completion signaling.
- Fixed final partial accumulation: empty work slots reach the optimizer boundary without adding samples, tokens or model calls. Real CPU optimizer tests verify all five test rollouts reach two updates, with and without backpressure.
- Added pinned FLA packages and aligned CUDA compiler/CRT/NVVM/headers to toolkit 13.0.2. GPU startup now checks outputs and gradients before launching rollouts.
- Republishing a changed final-checkpoint marker is tested. Eval-controller tests cover periodic/final selection, restart deduplication and cancellation.
- Cleanup uses the environment interpreter, where the E2B SDK is installed. Trainer exit codes and fault traces are recorded; artifact publication runs even if cleanup fails.
- Trackio initializes its offline path before imports. Smoke eval uses the actual checkpoint step. Qualification requires fresh online metric readback, not just a successful upload call.

## Attempt history

| Attempt | Outcome |
| --- | --- |
| `6aac143d5c02253cfb143bad` | Canceled before training; fixed CLI mismatch. |
| `6aac15cd5c02253cfb143bf3` | Canceled after missing FLA dependencies were identified. |
| `6aac1a80b1dc2b62dc58ee7e` | Confirmed FLA selection; superseded by final-accumulation fix. |
| `6aac1cf1b1dc2b62dc58eeca` | Failed at kernel compilation; exposed compiler/header mismatch and cleanup-interpreter bug. |
| `6aac201c5c02253cfb143e00` | Isolated diagnostic: Triton guard rejected unsafe Hopper combination; TileLang exposed CUDA mismatch. |
| `6aac2120b1dc2b62dc58ef67` | Corrected CUDA toolchain passed H200 output/gradient checks. |
| `6aac2255b1dc2b62dc58ef92` | Training/resume/TiTO passed through step 513; eval failed before rollout because a helper module was absent. Cleanup passed. |

The rejected Triton path is not enabled. See [upstream Hopper precision issue](https://github.com/fla-org/flash-linear-attention/issues/640). Owned-sandbox cleanup checks after the stopped/failed rollout attempts found zero matching live sandboxes.

## Evidence and remaining gates

Experiment root: `/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-hard500-2epochs-from500-20260917/`.

- `hf-jobs-v8/pytest.log`: complete combined local test output.
- `hf-jobs-v8/kernel-diagnostic*.log`: numerical/compiler diagnostics.
- `hf-jobs-v*/`: immutable bundles, upload revisions, submission receipts and cleanup records.
- `smoke-evidence/<job-id>/`: downloaded logs, status and timestamped monitoring records.
- `qualification_events.jsonl`: operation history. [Earlier working notes](qualification-history-20260917.md) are preserved.

The v9 smoke reached step **513** and drained all eight groups: **64 admitted rollouts**, no abandoned partial groups. Full checkpoint 502 was published, downloaded and resumed; checkpoint 513 was published. Both trainer processes exited with code 0.

Capture audit: **72/72 results passed**, **299,485/299,485 eligible tokens retained**, no oversized rows, all **64 optimizer receipts** reconciled. Extra captured results include uncommitted work regenerated around the intentional restart. Cleanup verified zero remaining owned sandboxes.

Of 13 optimizer updates, **9 had zero fresh gradient**. The four nonzero-gradient updates include two after resume. This is a learning-efficiency concern for hard-only groups, not a resume failure; the small smoke does not establish curriculum quality. Timings varied from 6 to 222 seconds after the first cold update. Forked histories remain a material compute cost.

Evaluation stopped at import before any eval rollout: `eval_pass_at_k.py` was missing from the bundle. Bundle v10 adds that helper and `harnesses_supported.txt`, and runs the exact evaluator CLI in dry-run mode before GPU work. The local dry run passed (eight cells, four harnesses, concurrency 4, step limit 17); three coordinator/publication tests passed again.

Bundle v11 also separates Trackio smoke run names by bundle attempt. The v10 job `6aac2e63b1dc2b62dc58f0ab` was canceled during setup for this logging fix. The read-only watcher records evidence every 60 seconds, stops on terminal status, and is bounded to 2.5 hours; it does not launch or restart jobs.

### Final v11 result

Job `6aac2ec9b1dc2b62dc58f0ba` completed successfully. The launcher qualification receipt is saved beside the immutable v11 bundle.

- 13 optimizer updates, finishing at checkpoint 513; 64 rollouts admitted; finite schedule exhausted; no abandoned partial groups.
- Full remote optimizer checkpoint restore passed; model weights changed; fresh gradients occurred before and after resume.
- 72/72 captured results passed TiTO; 293,996/293,996 eligible tokens retained; all 64 optimizer receipts reconciled; zero oversized rows.
- All eight pass@1 eval cells completed across four harnesses, with TiTO and pinned harness-version checks. This two-task test validates plumbing, not benchmark performance.
- Trackio readback verified 16 records in the distinct smoke run; final checkpoint publication and owned-sandbox cleanup passed.
- Maximum observed staleness was 4. Nine of 13 updates had zero fresh gradient: hard-only curriculum efficiency remains a concern.

The smoke validates the in-job environment, training, checkpoint restore, eval and logging paths. It does not establish 72-hour reliability or live concurrency-50 performance of the separate A100 eval jobs. The periodic/final eval coordinator has local tests; its first production dispatch still needs monitoring. Production starts from the untouched checkpoint 500, not the smoke checkpoint.

## Production launch

- Trainer: [6aac5eedb1dc2b62dc58f6b0](https://huggingface.co/jobs/HuggingEnvs/6aac5eedb1dc2b62dc58f6b0), `h200x2`, 72-hour limit.
- Eval coordinator: [6aac5eedb1dc2b62dc58f6b2](https://huggingface.co/jobs/HuggingEnvs/6aac5eedb1dc2b62dc58f6b2), CPU, 96-hour limit.
- Same qualified v11 bundle and immutable HF revision. Save every 50 updates; separate A100 evaluations every 100 updates and at completion, 250 fixed tasks × four harnesses, pass@1, concurrency 50. First periodic checkpoint is 600.
- Local read-only collector checks trainer/coordinator/eval state every ten minutes for the first hour from 2026-09-17 21:48 UTC, then every thirty minutes from 22:48 UTC. It records failures and stops when jobs finish, with a 96-hour bound. It does not automatically repair/restart jobs or send notifications. Evidence: `production-monitor/` and `hf-jobs-v11/production-watcher.log`.

### Status at 2026-09-18 06:03 UTC

305 new updates; 185/305 (60.7%) zero fresh gradient. Last 50 updates: reward mean 0.100 versus 0.155 in the preceding 50; 37/50 zero-gradient updates. Different task groups make this an observation, not a controlled improvement comparison. Last-50 mean step time: 90.6 seconds; maximum staleness 4. Trackio readback verified 304 events, one update behind the downloaded training log.

Checkpoint 600 eval remains running: latest downloaded traces contain 190/1,000 unique graded cells (OpenCode 56, Claude Code 13, Codex 25, Mini-SWE-Agent 96). Numerous retries include agent command failures and 600-second timeouts. The older scoring snapshot covers 166 cells and is not a final score. Checkpoints 700/800 are pending behind this job; the eval pipeline is not keeping up. No completed checkpoint eval is available yet for this continuation.

Last updated: 2026-09-18T06:04:57.211790+00:00
