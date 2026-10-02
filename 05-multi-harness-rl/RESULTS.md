# Historical results

[Read the article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Tutorial](README.md)

These are checks and pilot results from the earlier pinned recipe at `4d9c040`, before the tutorial rewrite. They do not qualify the current training scripts. Two blackbox pilots completed 100 updates and evaluation; the whitebox pilot stopped at step 31 after an HTTP read timeout. Those pilots used the previous runtime pins. The merged-contract smoke qualification is recorded separately in [VALIDATION.md](VALIDATION.md). Earlier experiment scores are kept on the [archived branch](https://github.com/adithya-s-k/FineEnvs/tree/archive/data-agent-experiments-20260930/04-data-agent).

## Completed checks

The merged-contract HF smokes completed two updates and checkpoint reload in all three modes. Reload evaluation graded native OpenCode 4/4 pairs, Harbor 4/4 pairs and whitebox 1/1 task. [Earlier validation](https://github.com/adithya-s-k/FineEnvs/blob/4d9c040a28695c484de75bc23d6a70a99beda258/05-multi-harness-rl/VALIDATION.md) records pins and evidence. The qualification table below predates that migration.

| Check | Result | Evidence |
|---|---|---|
| Local training, LFM and Qwen × all three modes | All six completed two updates and saved checkpoints | [Validation record](VALIDATION.md) |
| HF native OpenCode, LFM on A100 | Two updates; both checkpoints persisted | Internal job record `6abcc3d9` |
| HF Harbor, LFM on A100 | Two updates; checkpoint reload; 8/8 pairs graded with TiTO and verified tool counts | Internal job record `6abcc72d` |
| HF whitebox, LFM on A100 | Corrected preflight completed training and reload evaluation before starting its pilot | Internal job record `6abcd472`; the subsequent pilot failed at step 31 |

The Harbor reload smoke solved 5/8 pairs on two medium tasks. This is a functional check, not a benchmark. Both standalone HF blackbox training smokes had zero reward contrast; they validated execution and checkpointing, not learning.

## LFM 100-step pilots

Snapshot: **2026-09-30 17:45 UTC**, read from the saved bucket summaries. Each pilot uses the same 25 held-out tasks (3 easy, 12 medium, 10 hard), baseline evaluation, 100 training updates, saves at 50/100 and final evaluation. Evaluation concurrency is 35. Blackbox evaluations have 100 task/harness pairs; whitebox has 25 native episodes.

| Mode | Baseline pass@1 | Checkpoint-100 pass@1 | Training status |
|---|---|---|---|
| Native OpenCode | 24/100 = 24% | 20/100 = 20% | Completed 100 updates |
| Harbor multi-harness | 17/100 = 17% | 32/100 = 32% | Completed 100 updates |
| Whitebox SETA | 7/25 = 28% | Not available | Stopped at step 31; HTTP read timeout |

Scores come from complete `baseline/eval/summary.json` and `checkpoint-100/eval/summary.json` files. These are separate stochastic baseline samples at temperature 0.8, so the initial 24% versus 17% difference is not a training effect. Whitebox uses a different native evaluation interface. With only 25 tasks per harness, treat changes as pilot observations, not evidence of a reliable model ranking.

The jobs used immutable uploaded source snapshots; the new contract migration does not modify these results. The whitebox timeout remains a limitation of that earlier pilot, not a completed 100-step comparison.

| Baseline evaluation harness | Native OpenCode pilot | Harbor pilot |
|---|---:|---:|
| OpenCode | 2/25 (8%) | 3/25 (12%) |
| Claude Code | 5/25 (20%) | 4/25 (16%) |
| Codex | 4/25 (16%) | 2/25 (8%) |
| Mini-SWE-Agent | 13/25 (52%) | 8/25 (32%) |

The 10:17 UTC artifact sample contains 2 correct answers among 51 native OpenCode rollouts, 4 among 32 Harbor rollouts, and 0 among 80 whitebox episodes. These include collected work and are not counts of optimizer-consumed samples. They explain why useful gradient updates are sparse; they are not held-out scores.

## Locate an artifact

Internal test and pilot outputs use the **private** bucket `FineEnvs/data-agent-daytona-artifacts`, under these run prefixes:

- `pilot-lfm-opencode-v2`
- `pilot-lfm-multi-harness-v2`
- `pilot-lfm-whitebox-v1`

Each prefix contains `pilot.json`, `baseline/`, `train/`, `checkpoint-100/` and, after completion, `comparison.json`. Checkpoints contain model weights, optimizer and RNG state. Logs include local Trackio data, scalar metrics and rollout/audit evidence. These paths are maintainer records, not publicly accessible artifacts. No public trained-model artifact is claimed for these pilots.

The [historical Trackio Space](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) does not automatically contain these new pilots. Their current config keeps Trackio offline in the bucket. Maintainers can inspect the saved HF job logs. The tutorial explains how to enable your own online Trackio Space.

## Remaining checks

- Complete a whitebox 100-step pilot with the updated runtime. The earlier pilot failed before checkpoint 50.
- Confirm sustained learning signal, especially in native OpenCode and whitebox.
- Test separate evaluation jobs reading newly saved bucket checkpoints before enabling a long-run watcher. The completed smoke tested reload within one job.
- Treat Qwen HF execution and hardware combinations not tested here as needing their own smoke.

Detailed pins, failure diagnoses, test counts and original smoke evidence are in [VALIDATION.md](VALIDATION.md).

## Historical article curves

The [multi-harness RL article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) covers the earlier LFM and Qwen experiments on [SmolDataEnvs](../04-smoldataenvs/). The README figure shows its two LFM runs, both trained with correctness plus tool efficiency and evaluated on 250 tasks across four harnesses.

| Historical LFM run | Step-1,000 pass@1 | Graded pairs | Tool-call savings on matched successes |
|---|---:|---:|---:|
| Harbor, OpenCode only | 52.3% | 1,000/1,000 | 11.39% |
| Harbor, multi-harness | 54.2% | 1,000/1,000 | 31.07% |

The recorded baseline is 421/998 graded pairs (42.18%); two pairs were ungraded. Some early checkpoints also have incomplete coverage, marked with hollow points in the figure. These are observed pass@1 scores, not confidence intervals. The two final scores alone do not establish a statistically significant difference between runs.

Both historical runs used Harbor. The current tutorial instead uses native OpenCode for its `opencode` mode and explicitly non-thinking serving for both models. These curves do not validate the new pilot configuration. The article also includes Qwen results, whose historical reward was correctness only.

The figure uses a [pinned copy of the article data](https://huggingface.co/spaces/FineEnvs/multi-harness-rl/blob/a1e03a6c7d8fd0d51d0cf7361ba51a469cc6defc/app/src/content/assets/data/training-results.json). The [local excerpt](assets/historical-lfm-curves.json) keeps the plotted values, evaluation coverage, source URL and source SHA-256. The [Trackio dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) provides the historical run charts.

To redraw the figure from this folder, without downloading data or starting a job:

```bash
uv run --no-project --with matplotlib==3.11.1 python assets/plot_history.py
```

Training reward is a trailing 50-update mean, with raw values shown faintly. Evaluation is unsmoothed. Tool savings use task/harness pairs solved by both baseline and checkpoint; positive values mean fewer calls.
