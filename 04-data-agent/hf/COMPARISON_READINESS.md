# Comparison training readiness

Updated 2026-09-15 21:33 UTC. **Both main comparison runs are launched:** native OpenCode80626 on two local H100s, with checkpoint controller80627; HF SETA6aa9b6c9f76d6a098a70e786 on h200x2, with controller80605. All optimizer/checkpoint/logging qualification gates passed. Persistent10-minute monitor80636 follows all three runs; see [RUN_PROGRESS.md](RUN_PROGRESS.md). Original multi-harness training continues independently as80608 from checkpoint684.

| Setting | Native OpenCode | Whitebox SETA |
| --- | --- | --- |
| Main training | hopper-prod, 2 H100s | HF Jobs, `h200x2` |
| GPU roles | One trainer, one inference replica | Same |
| Algorithm | Atomic AsyncGRPO | Synchronous GRPO, DAPO loss |
| Model | Qwen/Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | Same |
| Learning rate / generations | 3e-6 / 8 per task group | Same |
| Sampling | Temperature 0.8, top-p 1, top-k disabled, thinking disabled | Same |
| Training data | Frozen 1,000 tasks: 150 easy, 600 medium, 250 hard | Same manifest and task order |
| Test data | Fixed 250 tasks: 33 easy, 118 medium, 99 hard | Same |
| Rollout limits | 16 outstanding, 32 worker ceiling, staleness 4 | 8 synchronous rollout slots |
| Checkpoints | **Save every 50 optimizer steps; retain all** | Same |
| Evaluation | **Every 100 steps**, plus final checkpoint | Same |
| Evaluation scope | 250 tests × 4 harnesses, pass@1 | 250 tests × native SETA, pass@1 |
| Independent eval GPUs | 2 H100s, TP1/DP2 | A100 80GB, TP1/DP1 |
| Eval concurrency | 50 | 50 |
| Run limit | 1,000 updates or 82,200 seconds, within a 24-hour allocation | Same |
| Logging | Native Trackio locally; metric events and database backups in the artifact bucket | Same |

The earlier 100-task wording is recorded in the historical plan. Matching the existing run uses its established **1,000-task pool**. Eight generations make one task group; they do not mean eight optimizer steps. The two algorithms consume different token counts per update. The 1,000-update limit is a ceiling, not a promise to finish 1,000 updates within 24 hours. Both main runs initialize independently from the pinned base; smoke weights are disposable.

**Native tolerance correction:** the audit found that the native host verifier replaced explicit zero tolerances with 0.001. This is corrected in task parsing and grading. Deterministic rescoring of all 250 original first responses changed zero grades (still 21 correct); all 1,250 frozen grading parameter sets match. Corrected optimizer smoke80593 passed all four updates, remote restore, changed weights and exact TiTO; live checkpoint evaluation80603 passed8/8 cells before automatic main launch. The old main plan is superseded. Whitebox’s verifier and its qualification are unchanged.

The requirement-by-requirement [bring-up audit](BRINGUP_AUDIT.md) rechecks the frozen evidence and records the later-authorized launch scope.

## Verified short tests

| Test | Result |
| --- | --- |
| Corrected native OpenCode optimizer80593 | Four nonzero-gradient updates; save2, remote restore, updates3–4; native optimizer state, changed weights and exact TiTO passed; 12m15s; cleanup0 |
| Local SETA optimizer 80555 | Four updates, two with nonzero gradients; same save/restore/TiTO/state checks passed; 18m19s; cleanup 0 |
| Corrected native checkpoint eval80603 / controller80594 | Checkpoint4 on separate GPUs;8/8 cells across two tasks × four pinned harnesses; TiTO passed; cleanup0 |
| Local SETA checkpoint eval 80576 / controller 80572 | Checkpoint 4 on separate GPUs; 2/2 tasks graded, TiTO passed; cleanup 0 |
| HF SETA smoke `6aa9a487f76d6a098a70e3d2` | Passed: four updates, three nonzero-gradient; full native state, remote restore, TiTO and changed weights verified |
| Save/eval scheduling | Actual main commands save at 50; controller fixtures select 100/200 from saved 50/100/150/200. Short GPU tests use save 2/eval 4. |

The short checkpoint tests used concurrency 8 and qualify the checkpoint path. They do not establish 50-way scalability. Fifteen focused checkpoint, dispatch and launch tests pass, including wrong-weight rejection and a proof-checked dry run that allocates no jobs. Seven local controller/score-forwarding checks passed earlier.

The corrected native OpenCode capture audit retained all **25,232/25,232 eligible supervised tokens**, with44 completed TiTO-valid captures and16 separate optimizer-consumption receipts. The independent incremental CPU monitor's first audit matched these counts. Captured and optimized rollout counts are distinct.

The HF optimizer's archived TRL package now installs with its distribution metadata, preserving locked dependencies. Its optimizer proof passed. Independent CPU controller 80567 launched HF eval `6aa9af55f76d6a098a70e52d` on A100: checkpoint 4, all 250 SETA tests, pass@1, concurrency 50. This completed250/250 with52correct (20.8% pass@1), all TiTO. Late Trackio replay/idempotence passed; main HF trainer6aa9b6c9f76d6a098a70e786 is submitted with controller80605. Main CPU controllers have 36-hour allocations to cover training and trailing evals.

## Baselines

| Cohort | Pass@1 | Evidence |
| --- | --- | --- |
| Native OpenCode, local 80514 | **21/250 = 8.4%** | 250 TiTO-valid, no ungraded attempts/retries; fixed concurrency 50; eval phase 14m23s |
| Native SETA, fresh HF baseline | **47/250 = 18.8%** | 250 TiTO-valid, one ungraded infrastructure retry; job `6aa97763f76d6a098a70de94` |
| Earlier Harbor four-harness Daytona baseline | **159/1,000 = 15.9%** | OpenCode 32, Claude 42, Codex 38, Mini 47 correct out of 250 each; all TiTO/version checks passed |
| Earlier SETA cluster cohort | 41/250 = 16.4% | Historical cohort retained independently |

These are separate runtime cohorts and are not pooled. Native OpenCode's 250-task baseline is distinct from the four-harness evaluation protocol used on its training checkpoints. The earlier HF 100-container native attempt was incomplete and is not reported as a baseline. Its inference queues and SDK command deadlines motivated the tested process-wait handling and fixed-50 local run.

The native baseline's 2,849 agent calls had no per-call sampling override; the server supplied the pinned defaults. The qualified optimizer additionally forwards and validates the explicit trainer sampling policy through the native client and session registry. Baseline artifacts were not rewritten to invent newer evidence.

## Evaluation and recovery behavior

Full checkpoints contain model, optimizer, scheduler and RNG state; async checkpoints also preserve the rollout cursor. The publisher uploads checkpoint files before the completion marker. Separate evaluators verify downloaded model hashes and never reload the trainer's inference process. Scores are keyed to checkpoint identity; infrastructure failures may be retried only while ungraded, preserving the first graded zero or one.

Intermediate 50-step checkpoints remain available for later evaluation. The local CPU supervisor forwards completed full evaluation scores into the training metric ledger, including late results. HF scores are persisted by the external coordinator and polled by the training logger while it runs. The main external controller now also replays late scores into the finished trainer’s Trackio ledger and database backup on CPU; qualification replay/idempotence check80586 passed after checkpoint eval completion. No Trackio Space is deployed, and the present proof establishes offline logging plus remote artifact persistence, not a live online dashboard.

Daytona's measured EU resource limits support 125 current 4-GiB sandboxes. Admission reserves 24 for the two trainers and defers evals when capacity is insufficient. Independent GPU jobs avoid taking inference capacity from training; storage and sandbox-provider latency remain shared resources.

## Implementation and evidence

Native training uses `envs/blackbox-opencode`, not Harbor training. Its checkpoint evaluations use four Harbor adapters. SETA uses native bash/read/write/edit/grep/glob/ls/submission tools with the frozen Harbor task verifier. The eight core correctness functions match all 1,250 frozen task graders by AST comparison (`grader-core-validation.json`); answer-file discovery and tool wrappers remain different. Binary correctness excludes native chat/efficiency shaping. The native baseline preserves raw shaped fields, but pass@1 counts only correctness ≥1; its nine partial-chat answers count as failures. The training session applies the same binary threshold. SETA's 16,384-token whole-episode budget includes masked tool results; native OpenCode keeps its per-call limits and atomic packing behavior.

Real provider tests passed three Harbor/Daytona lifecycles and eight SETA tool/grading checks across four tasks. Native Daytona and HF backend process/file smoke tests also passed and deleted their sandboxes. The Harbor tar/UID problem and async-client loop problem were handled separately: [issue 1959 comment](https://github.com/harbor-framework/harbor/issues/1959#issuecomment-5584356565), [related PR 2043](https://github.com/harbor-framework/harbor/pull/2043). Evidence is under `experiments/daytona_harness_comparison/logs/20260915/` and `logs/hf-20260915/`.

Prepared main plans: `logs/hf-20260915/local-opencode-main-ready-v2/plan.json` and `logs/hf-20260915/hf-whitebox-main-ready/plan.json`. Exact source/proof paths are recorded there. [LOCAL.md](LOCAL.md) describes the launchers; [configs/cadence_validation.json](configs/cadence_validation.json) records scheduling evidence.

The original multi-harness checkpoint-100 recovery is complete: 1,000/1,000 graded, 248 correct, all TiTO; recovery 79057 preserved the existing 999 grades. See [checkpoint-100 report](../eval/CHECKPOINT_100_PASS_AT_1.md).

The source worktree remains unpushed. HF jobs use the explicitly deployed, pinned runtime bundles. No git pushes or edits to existing training snapshots/shared virtual environments were made.

The completion audit re-read all full baseline ledgers and found no duplicate or conflicting grades. `baseline-ledger-completion-audit.json` binds those ledgers to the scores; `seta-hf-baseline-token-reaudit.json` verifies exact engine context, token IDs, masks and logprobs for all 250 HF SETA captures (825,964 supervised tokens). Both late-logging rejection checks pass: an active trainer cannot be touched, and scores from another trainer cannot be uploaded. The real late-score replay and its idempotence check passed.
