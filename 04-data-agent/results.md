# Data-agent results

Snapshot: **2026-09-16 UTC**. Metric: **pass@1** on the fixed 250-task test set (33 easy, 118 medium, 99 hard). Each accepted async checkpoint has 250 tasks × four harnesses = **1,000 first-graded cells**. SETA uses its native bash/SETA evaluator, 250 cells.

![Training and evaluation comparison](results/2026-09-16/comparison.png)

The [complete report](results/2026-09-16/REPORT.md) includes training history and **harness × difficulty at every accepted checkpoint**. [CSV](results/2026-09-16/checkpoint_scores.csv) provides the underlying correct/graded counts; [snapshot](results/2026-09-16/snapshot.json.gz) retains audited provenance and training metrics. The [live Trackio dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio) may contain newer observations.

| Checkpoint | Harbor multi-harness | Native OpenCode training, four-harness eval | SETA native eval |
| --- | ---: | ---: | ---: |
| Base | 14.6% | 15.9% | 18.8% |
| 100 | 24.8% | 19.7% | 34.8% |
| 200 | 26.3% | 22.1% | — |
| 300 | 28.6% | 21.6% | — |
| 400 | 33.3% | 26.4% | — |
| 500 | **37.0%** | 23.1% | — |
| 600 | 31.8% | 25.1% | — |
| 684 (recovery) | 32.1% | — | — |
| 700 | 28.8% | 23.2% | — |
| 800 | 27.0% | 25.6% | — |
| 900 | Incomplete | 25.3% | — |
| 1000 | Incomplete | **29.8%** | — |

Harbor multi-harness and native OpenCode reached 1,000 training steps. Native OpenCode's final four-harness evaluation is complete at 29.8%; Harbor multi-harness still lacks accepted step-900/1000 scores. SETA was intentionally stopped after a verified checkpoint 150; its final evaluation is pending in this snapshot. Harbor OpenCode-only is a new run in progress; no post-training checkpoint score is claimed here.

## Baselines and difficulty

| Measured cohort | Easy | Medium | Hard | Overall |
| --- | ---: | ---: | ---: | ---: |
| Harbor multi-harness base (E2B) | 53/132 = 40.2% | 68/472 = 14.4% | 25/396 = 6.3% | 146/1000 = 14.6% |
| SETA base (HF Job / Daytona) | 14/33 = 42.4% | 27/118 = 22.9% | 6/99 = 6.1% | 47/250 = 18.8% |
| SETA checkpoint 100 | 23/33 = 69.7% | 45/118 = 38.1% | 19/99 = 19.2% | 87/250 = 34.8% |

Native OpenCode's **standalone** base evaluation scored **21/250 = 8.4%**. That is a different protocol from the **15.9%** four-harness Harbor/Daytona baseline used for its checkpoint comparison. Do not mix these denominators or relabel one cohort as the other. The shared Harbor OpenCode-only run reuses the recorded E2B base cohort and has no new measured gain yet.

## What the runs established

- Exact captured prompt/completion IDs, real aligned log probabilities and authoritative loss masks are usable across the selected harnesses. Lossless forks preserve supervision when prompts change; more rows still affect token cost and weighting.
- The async recipe admits complete rollout groups and checks retained supervision against capture records. The optimizer, checkpoint, upload and remote-resume paths have real GPU evidence.
- The native OpenCode baseline completed 250 tasks at local concurrency 50. HF SETA exercised 8, 32 and 53 concurrent slots. Earlier scaling failures are preserved; the reproduction defaults to **35** on Hub infrastructure.
- Checkpoint evaluations run on separate GPUs, with fixed test identities and first-graded results. A graded zero is never replaced by a retry.
- Task parsing now preserves explicit zero numerical tolerances. Native baseline qualification deterministically rechecks the unchanged submitted answers and frozen grading parameters.

Historical qualification receipts: native optimizer smoke **80593**, local SETA **80555**, HF SETA **6aa9a487f76d6a098a70e3d2**; independent checkpoint smokes **80603**, **80576**, and **6aa9af55f76d6a098a70e52d** respectively. These are evidence for their recorded source snapshots, not substitutes for qualifying a changed bundle. Fresh PR qualification is recorded separately in [validation.md](results/validation.md).

## Limits of the comparison

Infrastructure, harness protocols, batching and recipe versions changed during bring-up. Async atomic batching and synchronous GRPO/DAPO have different scheduling and token accounting. These curves are observational; they do not isolate a causal effect of sync versus async or multi-harness versus one harness. Training reward is a sampled training signal, not held-out pass@1.

Scores only enter the accepted table after complete coverage and their recorded TiTO/version/provenance checks. The Harbor decline after step 500 is observed; this report does not assign a cause without a controlled ablation. Failed or incomplete cohorts remain visible in the snapshot's pending section and are not estimated.
