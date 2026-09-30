# CPU Basic Spaces and planned SETA stop

**Verified 2026-09-16 09:48 UTC:** all three Spaces are RUNNING on CPU Basic. Synchronous SETA training stopped at its fully verified checkpoint 150 at 09:45:32 UTC. Its final evaluation is pending sandbox admission behind native OpenCode checkpoint-1000 evaluation 81098. The independent Harbor OpenCode-only training run continues.

| Environment Space | Hardware | Transport sessions | Sandbox capacity | Reserved training slots |
| --- | --- | ---: | ---: | ---: |
| data-agent-blackbox-harbor-env | CPU Basic, verified running | 1,024 | 64 | 16 |
| data-agent-blackbox-opencode-env | CPU Basic, verified running | 1,024 | 100 | 8 |
| data-agent-seta-whitebox-env | CPU Basic, verified running | 1,024 | 61 | 8 |

All three completed hardware changes preserve every concurrency variable and frozen application bundle hash. Root/health/deployment checks passed, and each Space passed a burst of 50 concurrent read-only health/deployment requests. SETA was restarted only after its trainer stopped. No concurrency limits or model/harness pins are changed. These are configured limits; equivalent peak throughput on CPU Basic has not been benchmarked. The canonical deployment configuration now selects CPU Basic for future deployments.

## Stop and evaluate

CPU operation **81096** verified **checkpoint 150** of HF trainer `6aa9b6c9f76d6a098a70e786`, then stopped that trainer. All 14 checkpoint files (11.31 GB) were downloaded and checked against their SHA-256 manifest before cancellation. Independent saved metrics contain exactly 150 optimizer updates; all observed numeric scalars are finite. The 1,200 recorded training rows through this boundary passed TiTO, retaining 2,722,792 supervised tokens. Offline Trackio and its artifact ledger remain available.

The operation checked every checkpoint file hash, including optimizer, scheduler, RNG and model state, before canceling the GPU job. HF reports CANCELED for this intentional stop. The frozen trainer has no remote graceful-stop callback, so there may be extra in-flight work while publication finishes; the preserved and evaluated boundary is exactly step 150.

SETA's Space is now healthy on CPU Basic, with its endpoint, frozen bundle and concurrency settings verified. The independent final evaluation controller is running without alerts and has step 150 pending. Evaluation uses the same pinned protocol: **250 fixed test tasks, native bash/SETA, pass@1, concurrency 50, one A100**. Existing sandbox admission coordination remains in force: native OpenCode checkpoint-1000 evaluator 81098 currently holds the evaluation slot. SETA will dispatch automatically after that evaluator releases capacity.

The controller normally suppresses evaluation after cancellation. A narrowly bound final-checkpoint receipt now permits only the explicitly requested, fully verified checkpoint after this planned stop. Ordinary cancellation still suppresses new eval jobs. Submission identity remains tied to checkpoint manifest and evaluation protocol. The existing checkpoint-100 score is preserved.

Validation: **28 tests and 21 subtests passed**, including corrupt optimizer state, mismatched checkpoint/job identity, ordinary cancellation suppression and final evaluation of a verified stopped run. No GPU training source or frozen inference/evaluation bundle was changed.

## Existing verified SETA scores

| Checkpoint | Overall pass@1 | Easy | Medium | Hard |
| --- | ---: | ---: | ---: | ---: |
| Base | 18.8% (47/250) | 42.4% (14/33) | 22.9% (27/118) | 6.1% (6/99) |
| 100 | 34.8% (87/250) | 69.7% (23/33) | 38.1% (45/118) | 19.2% (19/99) |
| 150 | Pending final evaluation | Pending | Pending | Pending |

All final checkpoint-150 results and the difficulty breakdown will be written to the operation report after full coverage and TiTO/provenance checks pass.

- [Live operation report](/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/stop-whitebox-20260916/REPORT.md)
- [Machine-readable status](/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/stop-whitebox-20260916/status.json)
- [Space hardware and concurrency evidence](/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/stop-whitebox-20260916/space-hardware.json)
- [Operation request](/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/stop-whitebox-20260916/request.json)
