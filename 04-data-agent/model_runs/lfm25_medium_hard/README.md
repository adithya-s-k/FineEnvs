# LFM medium/hard comparison

Both GPU smokes and production orchestration checks passed on 2026-09-21. Production training has not been launched.

Two fresh pretrained `LiquidAI/LFM2.5-2.6B` arms use the same 1,000 tasks, task order, seed, sampling, optimizer, verifier and eval cohort. OpenCode means the Harbor OpenCode route, so the harness choice is the main changed variable; this does not switch to the separate native OpenCode environment.

| Setting | Both arms |
|---|---|
| Training pool | 600 hard + 400 medium; no easy tasks |
| Exposure | Proposed two passes: 2,000 task groups × 8 rollouts = 16,000 scheduled rollouts per arm |
| Harness routing | OpenCode only versus OpenCode / Claude Code / Codex / Mini-SWE-Agent |
| Multi-harness balance | Per pass: 250 groups per harness, including 150 hard and 100 medium; rotate on pass two |
| Async GRPO | LR 3e-6, G8, inflight32, staleness4, outstanding16, batch4, accumulation4 |
| Model/capture | Same pinned LFM revision and qualified TiTO/packed-convolution setup |
| Sampling | Temperature .8, top-p1, native reasoning, 4,096 tokens/model call |
| Checkpoints | Every 50 optimizer steps and final; two passes do not imply 200 optimizer steps |
| Evaluations | Every 100 optimizer steps and final; fixed250 × four harnesses, pass@1, concurrency50 |
| Resources | Separate trainer/inference GPU pair per arm; eval uses its own allocation |
| Logging | FineEnvs comparison Space, project `data-agent-rl-comparison` |

The 1,000-step field in candidate configs is a safety ceiling, not a promised duration. Dataset exhaustion should stop both arms. Resume must preserve committed-group accounting. Failures or abandoned partial groups reduce realized exposure and must be reported.

## Prepared artifacts

`prepare.py` copies and hash-checks the shared tasks under `experiments/lfm25-medium-hard1000-20260921/dataset`. It records selection provenance and verifies source, notebook and normalized-instruction separation from the fixed test set. Each arm has a candidate config, indices and schedule. Neither contains credentials.

```bash
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/prepare.py
.venv312/bin/python -m pytest HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/test_setup.py -q
```

## Efficiency reward proposal

Reuse the existing `train/harbor_reward.py` approach, with weight reduced from .3 to .1:

`training_reward = correctness * (1 + 0.1 * 15 / (15 + action_count))`

Award the bonus only when a positive action count is verified. Unknown or zero counts receive correctness only; missing verifier results remain unscorable. Incorrect answers always get zero. At 5 / 15 / 30 verified actions, correct answers receive 1.075 / 1.05 / 1.0333. `efficiency.py` installs this policy through the existing rollout reward callback, including spawned workers.

Count executed harness actions, including failed actions, from the existing native trajectory representation. Do not use API calls, turns, re-rendered history, or regular expressions over generated code as a universal proxy. Use native ATIF tool-call identities and require native counts to cover retained captured calls. Retries can leave more native actions than retained training actions; count those actions too. Missing or inconsistent evidence receives no bonus. Offline replay covers all four harnesses, including Mini-SWE-Agent.

Log raw correctness, action count, count coverage, efficiency bonus, shaped reward, tokens, failures and group reward contrast separately. Keep canonical evaluation binary correctness only; report action/token efficiency separately, ideally on the paired tasks solved by both models. A lower tool count is not necessarily lower compute: an agent can combine commands into one call.

GRPO group normalization can amplify a small bonus in all-correct groups: weight .1 does not guarantee a small gradient effect. All-wrong groups still receive no reward contrast. Monitor correctness-only versus bonus-driven groups and their gradient contributions; pause shaping if held-out correctness falls while shaped reward rises.

## Smoke validation

The runner uses the existing qualified LFM inference, Harbor service and TiTO runtime. Each arm trains to step 2, saves and resumes to step 4, checks finite/nonzero gradients and committed-group replay, then reloads checkpoint 4 for two fixed test tasks across all four harnesses. This is functional validation, not a benchmark or evidence of reward improvement.

To rerun a smoke on an available two-GPU allocation (from the workspace root):

```bash
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/run.py opencode train-smoke
# Use multi-harness in place of opencode for the other arm.
```

This uses the existing local model/runtime cache and credentials loaded from `experiments/.env`. Production evaluation needs its own GPU allocation.

Local checks cover the shared 1,000-task pool, explicit two-pass schedules and reward invariants. GPU evidence and job IDs are recorded in `experiments/lfm25-medium-hard1000-20260921/STATUS.md`. Qualification is bound to configuration, schedule and reward hashes.

Resume retries groups with no committed optimizer work, including cancelled or zero-scorable groups. Already committed groups are not replayed; their unfinished tails remain abandoned and reported. This is not lossless queue recovery. Actual task exposure must be audited rather than inferred from the planned two passes.

## Production validation and launch

Both GPU smokes passed: four optimizer steps (two nonzero), checkpoint save/resume and eight graded TiTO-valid evaluations per arm. Additional tests cover production arguments, early dataset exhaustion, cancelled-group recovery, fresh-worker imports, checkpoint finalization, eval cadence, restart deduplication and incomplete-eval reporting. Trackio upload/readback was verified against the live Space in the separate `lfm-production-validation-20260921` project.

The eval controller converts saved checkpoints to verified ready checkpoints, evaluates every 100 steps and the final checkpoint, and requests separate GPUs outside the training node. It reports incomplete jobs as failures; it does not automatically relaunch failed allocations. Model quality or reward improvement is not established by these smoke checks.

Preview the jobs (no submission):

```bash
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/launch.py opencode
.venv312/bin/python HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/launch.py multi-harness
```

Add `--submit` when launching. Each arm requests two H100s and starts an independent zero-GPU Slurm controller after training starts. The empty `hopper-cpu` partition is avoided. Both submission configurations passed Slurm `--test-only`; capacity is not reserved. The launch receipt records both job IDs. If controller submission fails, the newly submitted trainer is cancelled.

Checkpoints save every 50 steps. Evaluations use the fixed 250-task test set × four harnesses, pass@1, concurrency 50, TP1/DP2 on their own two-GPU allocations. Full training is still unlaunched. See `experiments/lfm25-medium-hard1000-20260921/PRODUCTION_VALIDATION.md` for evidence and remaining limitations.
