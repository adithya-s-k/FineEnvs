# Qualification notes

This is a new non-thinking recipe. Historical training results are not its baseline.

## Data and runtime

- Public Hub preparation passed for all 1,000 training tasks and 250 test tasks. Instruction hashes match the fixed selection; the corrected numeric-only grader is checked for every task.
- Train revision: `4719635555de1666f374d847baebb500d368e493`. Test revision: `b130595b579fc6026ad5762925a9a3b40e72f5bb`.
- Compared with the old staged files, 302 training tasks now use the JSON reward wrapper. Task metadata also changed. The test instructions and grader did not change.
- OpenEnv is pinned to public commit `4f4c85fb9038f43efc2f51858a27638277f16355`, including partial-mask validation. No unpublished OpenEnv checkout is required.
- TRL, native environment sources and file hashes are in `configs/runtime-lock.json`. The Python dependency set resolves successfully; `requirements.lock` records the resolved versions.

## Checks on 30 September 2026

| Check | Result |
|---|---|
| CPU contracts and fetched-runtime integration | 22 passed, including Ampere attention selection |
| Clean, minimal CI environment | 17 passed; four runtime checks skipped without fetched dependencies |
| Native OpenEnv to TRL partial mask | `[1, 0]` preserved |
| Native LFM checkpoint handoff | Both initial smoke checkpoints saved; file hashes verified |
| Corrected four-harness reload eval | 8/8 pairs graded; all token checks passed; four correct answers |
| Qwen whitebox training | Two updates and both checkpoints saved; second update had reward contrast and nonzero gradient |
| Qwen whitebox token audit | Both batches passed; 68 captured calls in total |
| Corrected non-thinking HTTP and streaming replies | Passed for LFM and Qwen |
| Both models, all three training modes | All six completed two updates and saved both checkpoints |
| Whitebox checkpoint reload eval | 2/2 tasks graded; one correct answer |
| Async checkpoint resume | Resumed step 1 to 2; committed group 0 was not replayed |
| HF Jobs setup | Fresh Python 3.12 environment, locked installation and all 1,250 tasks prepared successfully |
| HF Jobs native OpenCode GPU smoke | A100 completed two steps; both saved checkpoint inventories match the bucket |
| HF Jobs Harbor smoke and reload eval | Completed two training steps; checkpoint reload graded 8/8 pairs with TiTO and verified tool counts |

| Model | Mode | Local smoke wall time | Nonzero gradient observed? |
|---|---|---|---|
| Qwen3.5-2B | Whitebox | 10m 26s | Yes |
| Qwen3.5-2B | Native OpenCode | 8m 35s | No reward contrast in this sample |
| Qwen3.5-2B | Harbor multi-harness | 9m 07s | Yes |
| LFM2.5-2.6B | Whitebox | 6m 37s | No reward contrast in this sample |
| LFM2.5-2.6B | Native OpenCode | 9m 30s | No reward contrast in this sample |
| LFM2.5-2.6B | Harbor multi-harness | 13m 01s | Yes |

Wall time includes model loading, sandbox setup, two updates, saving and cleanup. Smoke batches use two rollouts and low concurrency, not the full training batch or eval concurrency 35. Local smokes use the existing cluster runtime; the separate HF check qualifies a fresh installation from the lockfile.

The initial LFM smoke exposed a serving error: the reasoning parser classified tool-call text as reasoning, leaving Claude Code with empty assistant replies. Non-thinking serving now omits that parser. Claude Code subsequently emitted native tool calls, and a Harbor OpenCode rollout produced correctness `1` with nine calls and shaped reward `1.0625`. Initial LFM smoke scores must not be reused as a baseline.

Other fixes found by the smokes: job-specific ports on shared Slurm nodes, explicit HF login forwarding to Harbor task staging, the SDK's typed bucket mount, explicit tool-response templates, the whitebox cache-reset response, and the text context limit on Qwen's outer model config. The pinned base image has Python 3.11; the HF entrypoint now creates its own Python 3.12 environment before installing the lockfile.

## Evidence

Local smoke artifacts are ignored under `runs/qualification/`: configuration, service logs, rollout reward records, `token-audit.jsonl`, local Trackio data and checkpoints. CPU logs are `cpu-tests-final.log` and `ci-tests.log`; public data preparation is `prepare-hub.log` and `prepared-hub/`.

- Qwen whitebox: `qwen-whitebox-v5`, Slurm job `92750`.
- Corrected LFM Harbor: `lfm-multi-harness-v4`, job `92751`.
- Corrected native OpenCode: `lfm-opencode-v4` / `qwen-opencode-v1`, jobs `92752` / `92753`.
- Corrected LFM reload evaluation: `lfm-nonthinking-eval-v2`, job `92757`.
- LFM whitebox / Qwen Harbor: jobs `92759` / `92760`.
- Whitebox checkpoint eval: `qwen-whitebox-v5-eval`, job `92821`.
- Async resume: `lfm-native-resume-v1`, job `92822`. Only optimizer step 2 was executed; the new admission record contains group 2, while the restored state retains group 0.
- [HF setup check](https://huggingface.co/jobs/FineEnvs/6abcb82f031314b696343d7d): completed with the corrected Python 3.12 bootstrap.
- [H200 GPU smoke](https://huggingface.co/jobs/FineEnvs/6abcb8c8031314b696343daf): canceled after remaining queued for hardware.
- [A100 native OpenCode smoke](https://huggingface.co/jobs/FineEnvs/6abcc3d9031314b6963440a0): completed. Runtime from container start to finish was 10m 05s; the trainer reported 335.3s. Both optimizer steps had zero reward contrast and zero gradient, so this confirms execution rather than learning improvement. Six completed rollout records had verified native tool counts. Both checkpoints include model, optimizer and RNG state; all 13 recorded file sizes match the persistent bucket. Byte hashes and cross-job reload remain unverified for this checkpoint. Evidence downloaded to `runs/qualification/hf-a100-native-v1/`.
- [A100 Harbor smoke](https://huggingface.co/jobs/FineEnvs/6abcc72d031314b696344167): two-step multi-harness training followed by checkpoint-2 reload evaluation on two tasks × four harnesses, concurrency four, 45-minute cap. Completed with `--smoke-eval` in 21m 17s overall; training took 627.9s. Both updates had zero reward contrast and zero gradient. Checkpoint 2 was hashed and reloaded with restarted services. All eight evaluation pairs graded on their first attempt, with TiTO and verified tool counts: OpenCode 1/2, Claude Code 1/2, Codex 1/2, Mini-SWE-Agent 2/2. These two medium tasks are a smoke, not a benchmark. Evidence is under `runs/qualification/hf-a100-harbor-v1/`. This tests all four evaluation adapters, while two training steps alone need not consume all four harnesses. The new orchestration has 19 passing CPU contract checks, including aborting reload after training failure.

The native and Harbor HF execution smokes passed. A same-job reload does not qualify a separate evaluation job reading the bucket; that handoff remains a separate check. Neither HF training sample had reward contrast, so a positive-gradient HF check also remains before claiming learning validation. Then collect a fresh non-thinking baseline on the fixed 250-task test set. A tiny smoke establishes execution, not benchmark performance or throughput at concurrency 35.

## 100-step pilot launch

Selected scope: LFM only, all three modes, 25 fixed held-out tasks (3 easy, 12 medium, 10 hard). Baseline and checkpoint-100 evaluations use the same task IDs. Each blackbox evaluation has 100 pairs; whitebox has 25 episodes. Save at 50/100, eval concurrency 35, eight rollouts per group, unchanged full training batches and LR. Phases run sequentially inside one HF allocation, bounded to 12 hours. Slurm plans use the same command with two GPUs and the same time cap.

- [Native OpenCode pilot](https://huggingface.co/jobs/FineEnvs/6abcd384031314b69634440b), `pilot-lfm-opencode-v2`: submitted.
- [Harbor multi-harness pilot](https://huggingface.co/jobs/FineEnvs/6abcd3874c46ef19870359af), `pilot-lfm-multi-harness-v2`: submitted.
- [Whitebox HF qualification](https://huggingface.co/jobs/FineEnvs/6abcd28f4c46ef1987035955): failed saving step 1 because the fresh environment lacked TRL package metadata. Training and token auditing had reached the update. The entrypoint now installs the pinned checkout with `--no-deps --no-build-isolation`; package metadata was verified in a clean Python environment.
- [Whitebox pilot](https://huggingface.co/jobs/FineEnvs/6abcd472031314b696344457), `pilot-lfm-whitebox-v1`: submitted with `--preflight-smoke`. It must finish two training updates and reload eval before starting the baseline/100-step phases; any failed qualification stops the job.

The first two pilot submissions were canceled while still scheduling, then replaced with copies that print phase and scalar progress in HF logs. No training progress was discarded. Configurations, scalar logs, local Trackio and phase reports live under each run prefix in `FineEnvs/data-agent-daytona-artifacts`. No Slurm pilot has been submitted.

Pilot checks: 23 CPU contracts passed, plus the four fetched-runtime integration checks. Coverage includes fixed stratification, preserving original task catalog indices, stopping on incomplete baseline, retaining normal training batches, phase resume and Slurm time limits. A comparison fixture verifies repeated-step deduplication and matched-success tool/token savings. Actual baseline/100-step results are pending.
