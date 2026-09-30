# Validation

This page records what we tested for the pinned, non-thinking recipe. [RESULTS.md](RESULTS.md) has the pilot scores and progress; [REPRODUCE.md](REPRODUCE.md) has the commands.

## Merged contract update (30 September)

The recipe now pins merged [OpenEnv #1280](https://github.com/huggingface/OpenEnv/pull/1280) and the updated [TRL #6947](https://github.com/huggingface/trl/pull/6947). The previous GPU results below used the earlier pins and are retained as historical evidence.

- 43 CPU checks cover the updated recipe, including native partial/zero masks, invalid captures, correctness plus efficiency through the typed TRL worker, and the whitebox update budget against TRL's sampler.
- The TRL integration passed 51 contract/reconciliation checks against the merged producer.
- Fresh-clone bootstrap and repeat-bootstrap handling are tested. User edits are preserved; only the recipe's known attention patch is removed before updating TRL.
- Fresh HF A100 training and checkpoint-reload evaluation completed in all three modes, as recorded below.

Each smoke used non-thinking LFM2.5-2.6B, two optimizer updates and two rollouts per group. Both checkpoints were saved; evaluation restarted inference from checkpoint 2. Blackbox evaluation used one fixed task across four harnesses; whitebox used one native task.

| Mode | Updates / checkpoints | Reload eval graded | Correct | Nonzero gradient? | Evidence |
|---|---|---|---|---|---|
| Native OpenCode | 2 / 2 | 4/4 | 2/4 | No reward contrast | [HF smoke](https://huggingface.co/jobs/FineEnvs/6abd47c0fbc85ba68235c27c) |
| Harbor multi-harness | 2 / 2 | 4/4 | 1/4 | Yes, norm 5.47 at step 1 | [HF smoke](https://huggingface.co/jobs/FineEnvs/6abd47bdfbc85ba68235c27a) |
| Whitebox SETA | 2 / 2 | 1/1 | 1/1 | No reward contrast | [HF smoke](https://huggingface.co/jobs/FineEnvs/6abd47c3fbc85ba68235c27e) |

These tiny samples validate execution, capture, saving and reload, not learning quality. Blackbox evaluations passed token/mask checks. A correct Harbor training rollout with six verified tool calls received 1.0714; failed rollouts received zero. Trainer reward averages are over training rows and must not be read as rollout pass@1.

The jobs ran OpenEnv `86a180ed` and TRL `a39476c6`. The current TRL pin has identical experimental trainer code; later changes cover example warmup, optional tests, documentation and an upstream CI fix. The first fresh-container attempt exposed a bootstrap restore bug before training; it was fixed and regression-tested before these replacement jobs.

Artifacts are in bucket `FineEnvs/data-agent-daytona-artifacts`, under `merged-contract-{opencode,multi-harness,whitebox}-v2`. Each has `metrics.jsonl`, checkpoints, local Trackio logs and `reload-eval/eval/summary.json`. Bucket access requires permission.

Execution status, typed execution limits and structured tool outcomes in OpenEnv remain follow-up API work from the TRL review. The recipe uses explicit per-session turn/time budgets and verified native tool counts; it does not claim the broader session API is complete.

## Earlier automated checks

- 29 tests passed with the downloaded runtime and prepared tasks.
- A clean CPU-only checkout passed 25 tests. Four integration tests skip until the runtime and tasks are downloaded.
- HF submission plans and `run.py plan` work without PyTorch installed.
- Shell syntax, Python compilation, documentation links and the repository index passed.

Tests cover task identity, train/test separation, reward calculation, token/logprob alignment, partial masks, checkpoint integrity, resume, config overrides and pilot failure handling. They also check that the uploaded source excludes credentials and local run artifacts.

## Local GPU tests

Both models completed two optimizer updates and saved both checkpoints in all three modes.

| Model | Mode | Total time | Nonzero gradient observed? | Slurm job |
|---|---|---|---|---|
| Qwen3.5-2B | Whitebox | 10m 26s | Yes | `92750` |
| Qwen3.5-2B | Native OpenCode | 8m 35s | No reward contrast in this sample | `92753` |
| Qwen3.5-2B | Harbor | 9m 07s | Yes | `92760` |
| LFM2.5-2.6B | Whitebox | 6m 37s | No reward contrast in this sample | `92759` |
| LFM2.5-2.6B | Native OpenCode | 9m 30s | No reward contrast in this sample | `92752` |
| LFM2.5-2.6B | Harbor | 13m 01s | Yes | `92751` |

These times include startup, sandbox work, training, saving and cleanup. Smokes use two rollouts per group and low concurrency; they do not measure full-run throughput.

Additional checks passed:

- Whitebox token audits matched 68 engine calls per model.
- LFM checkpoint evaluation graded 8/8 task/harness pairs with valid token capture (job `92757`).
- Qwen whitebox checkpoint evaluation graded 2/2 tasks (job `92821`).
- Async resume continued from step 1 to step 2 without replaying committed group 0 (job `92822`).
- The OpenEnv-to-TRL adapter preserved the partial completion mask `[1, 0]`.

Local evidence is kept under ignored `runs/qualification/`, including service logs, reward records, token audits and checkpoints.

## HF Jobs tests

| Test | Result | Evidence |
|---|---|---|
| Fresh installation | Python 3.12, locked dependencies and all 1,250 tasks prepared | [Setup job](https://huggingface.co/jobs/FineEnvs/6abcb82f031314b696343d7d) |
| Native OpenCode on A100 | Two steps; both checkpoint file inventories match the bucket | [Native smoke](https://huggingface.co/jobs/FineEnvs/6abcc3d9031314b6963440a0) |
| Harbor on A100 | Two steps; checkpoint 2 reloaded; 8/8 eval pairs passed capture checks and graded on the first attempt | [Harbor smoke](https://huggingface.co/jobs/FineEnvs/6abcc72d031314b696344167) |
| Whitebox on A100 | Two-step preflight and reload eval passed before its pilot started | [Whitebox pilot](https://huggingface.co/jobs/FineEnvs/6abcd472031314b696344457) |

Native OpenCode took 10m 05s overall, including 335.3s in the trainer. Harbor took 21m 17s overall, including 627.9s in the trainer. Both standalone blackbox smokes had zero reward contrast. The later pilots produced nonzero gradients; their current progress is in [RESULTS.md](RESULTS.md).

## Fixes found during testing

- **Hidden tool calls:** the LFM reasoning parser swallowed tool-call text. Non-thinking serving now omits that parser and checks visible streaming and ordinary replies. Earlier smoke scores from that setup are superseded.
- **A100 attention:** the pinned async trainer hardcoded FlashAttention 3. A checked adapter selects FlashAttention 2 on Ampere while preserving packed-sequence boundaries.
- **Fresh HF installation:** the image contains Python 3.11. The entrypoint creates Python 3.12 and installs the pinned TRL checkout, including metadata needed for whitebox checkpoint saving.
- **Whitebox compatibility:** explicit response templates, Qwen's text context limit and an empty successful cache-reset response needed handling.
- **Shared-node startup:** job-specific ports, HF login forwarding and the typed bucket mount were tested in the live smokes.

## Data and source versions

The task lists contain 1,000 training tasks and 250 test tasks, with no task, notebook or instruction-hash overlap. Preparation verifies every instruction and the corrected numeric-only grader.

- Train revision: `4719635555de1666f374d847baebb500d368e493`.
- Test revision: `b130595b579fc6026ad5762925a9a3b40e72f5bb`.
- Current OpenEnv: `86a180ede21e044f7929b9a7783ad83aa67d83a3`. Earlier GPU qualification used `4f4c85fb9038f43efc2f51858a27638277f16355`.
- Other source revisions and archive hashes: [runtime-lock.json](configs/runtime-lock.json).
- Python dependencies: [requirements.lock](requirements.lock).

Compared with the old staged task files, 302 selected training tasks now use the JSON reward wrapper. Instructions and the corrected grader match; metadata changed. Historical results need to be checked against each run's actual data and serving configuration.

## Still to check

The earlier blackbox pilots completed: Harbor 17% to 32% and native OpenCode 24% to 20%, each over 100 held-out pairs. Whitebox stopped at step 31 after an HTTP read timeout. See [RESULTS.md](RESULTS.md); these results do not establish sustained learning or qualify the new pins. HF same-job reload has passed, but a separate evaluation job loading newly saved bucket checkpoints still needs testing before relying on the long-run watcher. Qwen HF execution and other hardware combinations need their own smoke.

This PR contains the recipe, tests and evidence links. Large artifacts remain in the HF bucket or ignored local run directories.
