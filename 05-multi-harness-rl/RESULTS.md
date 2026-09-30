# Current results

These are qualification results for this pinned, non-thinking recipe. They do not establish that a 100-step run improves either model. Earlier experiment scores are kept on the [archived branch](https://github.com/adithya-s-k/FineEnvs/tree/archive/data-agent-experiments-20260930/04-data-agent).

## Completed checks

| Check | Result | Evidence |
|---|---|---|
| Local training, LFM and Qwen × all three modes | All six completed two updates and saved checkpoints | [Validation record](VALIDATION.md) |
| HF native OpenCode, LFM on A100 | Two updates; both checkpoints persisted | [HF job](https://huggingface.co/jobs/FineEnvs/6abcc3d9031314b6963440a0) |
| HF Harbor, LFM on A100 | Two updates; checkpoint reload; 8/8 pairs graded with TiTO and verified tool counts | [HF job](https://huggingface.co/jobs/FineEnvs/6abcc72d031314b696344167) |
| HF whitebox, LFM on A100 | Corrected preflight completed training and reload evaluation before starting its pilot | [HF job](https://huggingface.co/jobs/FineEnvs/6abcd472031314b696344457) |

The Harbor reload smoke solved 5/8 pairs on two medium tasks. This is a functional check, not a benchmark. Both standalone HF blackbox training smokes had zero reward contrast; they validated execution and checkpointing, not learning.

## LFM 100-step pilots

Snapshot: **2026-09-30 10:20 UTC**. Each pilot uses the same 25 held-out tasks (3 easy, 12 medium, 10 hard), baseline evaluation, 100 training updates, saves at 50/100 and final evaluation. Evaluation concurrency is 35. Blackbox evaluations have 100 task/harness pairs; whitebox has 25 native episodes.

| Mode | Baseline pass@1 | Optimizer step (metrics) | Learning signal observed? | Job |
|---|---|---:|---|---|
| Native OpenCode | 24/100 = 24% | 10 | Yes: step 9 reward 0.126 and gradient norm 3.50 | [OpenCode pilot](https://huggingface.co/jobs/FineEnvs/6abcd384031314b69634440b) |
| Harbor multi-harness | 17/100 = 17% | 8 | Yes: step 8 reward 0.598 and gradient norm 2.70 | [Harbor pilot](https://huggingface.co/jobs/FineEnvs/6abcd3874c46ef19870359af) |
| Whitebox SETA | 7/25 = 28% | 5 | No reward contrast in its first five updates | [Whitebox pilot](https://huggingface.co/jobs/FineEnvs/6abcd472031314b696344457) |

All three jobs were running at this check. The whitebox save failure in its earlier smoke was traced to missing TRL package metadata and corrected before this pilot. The running jobs use immutable uploaded source snapshots; later tutorial edits do not modify them.

Baseline scores above come from complete `baseline/eval/summary.json` files downloaded from the bucket, not minute-by-minute progress counts. The two blackbox pilots independently sampled the same untrained model at temperature 0.8; these are separate stochastic samples, so their difference cannot be attributed to training. Whitebox uses a different native evaluation interface. No checkpoint-100 result exists yet, and these early updates do not establish a rising reward curve.

| Baseline evaluation harness | Native OpenCode pilot | Harbor pilot |
|---|---:|---:|
| OpenCode | 2/25 (8%) | 3/25 (12%) |
| Claude Code | 5/25 (20%) | 4/25 (16%) |
| Codex | 4/25 (16%) | 2/25 (8%) |
| Mini-SWE-Agent | 13/25 (52%) | 8/25 (32%) |

The 10:17 UTC artifact sample contains 2 correct answers among 51 native OpenCode rollouts, 4 among 32 Harbor rollouts, and 0 among 80 whitebox episodes. These include collected work and are not counts of optimizer-consumed samples. They explain why useful gradient updates are sparse; they are not held-out scores.

## Locate an artifact

Our qualification outputs use bucket `FineEnvs/data-agent-daytona-artifacts`, under these run prefixes:

- `pilot-lfm-opencode-v2`
- `pilot-lfm-multi-harness-v2`
- `pilot-lfm-whitebox-v1`

Each prefix contains `pilot.json`, `baseline/`, `train/`, `checkpoint-100/` and, after completion, `comparison.json`. Checkpoints contain model weights, optimizer and RNG state. Logs include local Trackio data, scalar metrics and rollout/audit evidence. Bucket access requires permission; job links may also require an HF login. No public trained-model artifact is claimed for these unfinished pilots.

The [historical Trackio Space](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) does not automatically contain these new pilots. Their current config keeps Trackio offline in the bucket. Follow the HF job logs for live progress. The tutorial explains how to enable your own online Trackio Space.

## Remaining qualification

- Finish the pilots and compare complete baseline/checkpoint-100 results.
- Confirm sustained learning signal, especially in native OpenCode and whitebox.
- Qualify separate evaluation jobs reading newly saved bucket checkpoints before enabling a long-run watcher. The completed smoke tested reload within one job.
- Treat Qwen HF execution and hardware combinations not tested here as needing their own smoke.

Detailed pins, failure diagnoses, test counts and original smoke evidence are in [VALIDATION.md](VALIDATION.md).
