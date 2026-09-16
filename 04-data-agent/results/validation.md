# PR preparation validation — 2026-09-16

This records validation of the prepared sources separately from the historical learning curves.

| Check | Evidence |
| --- | --- |
| HuggingEnvs CPU regression suite | 137 passed, 1 skipped; 21 subtests. Includes real CPU optimizer grouping, save/resume boundaries, capture budgets, task dispatch, HTTP controls, eval recovery and artifact provenance. |
| Portable archive | 10,602 packaged files hash-verified; source runtime matches the reviewed files; no configured credential values included. No external local experiments checkout required to build. |
| Frozen native grading | 1,250 task configurations verified; all 250 original first-graded baseline answers replayed with identical scores. |
| Local/Hub commands | CLI help, dry-run commands, Python compilation and fatal-error lint passed; generated project index checked. |
| OpenEnv | 2,393 CPU tests passed with unrelated QED service tests excluded; 107 additional upstream MCP integration tests passed after the current-main merge. Harbor capture/UI checks include concurrent trace isolation, session budgets and browser layout. |
| TRL | 245 CPU tests passed, plus HTTP controls and pre-commit checks. Main is merged and PR CI is tracked separately. |

## Fresh training smokes

The trainer bundle is `ccfe97822cf7c88931acda8a4894bd7515e40a939ebfcc4c20c14db45e607de3`, uploaded to `HuggingEnvs/data-agent-daytona-repro` at revision `3808a6d5c48320b5e7745c877dc9b7ed2819310b`. Source pins are in [sources.json](../hf/configs/sources.json). The OpenEnv runtime pin includes the TiTO/UI changes; later OpenEnv PR commits update documentation, optional tests and merge newer upstream MCP behavior.

These jobs qualify the new trainer against existing separately pinned Spaces. They do not restart or upgrade those Spaces. A smoke proves four optimizer updates with a checkpoint-2 remote restore; it is not a new baseline or evidence of a reward gain.

| Implementation | Job | GPUs | State |
| --- | --- | --- | --- |
| Harbor / OpenCode | [6aaa75bb5527934177ee9b8b](https://huggingface.co/jobs/HuggingEnvs/6aaa75bb5527934177ee9b8b) | A100 ×4 allocation; two used | Failed before optimizer startup: missing endpoint directory |
| Native OpenCode | [6aaa75bb5527934177ee9b8d](https://huggingface.co/jobs/HuggingEnvs/6aaa75bb5527934177ee9b8d) | A100 ×4 allocation; two used | Failed before optimizer startup: missing endpoint directory |
| SETA whitebox | [6aaa75bbf76d6a098a710867](https://huggingface.co/jobs/HuggingEnvs/6aaa75bbf76d6a098a710867) | H200 ×2 | Failed before optimizer startup: missing endpoint directory |

The clean-Job failure is fixed by creating the endpoint/log parent directories in `serve/vllm.sh`. Replacement jobs are being qualified; the failed cohort is preserved.

Completion requires `training_smoke_verified.json`: exact capture, retained supervision, native optimizer state, remote restoration and changed weights. Pending jobs are not counted as passed.

## Preserved material

Preparation uses separate Git worktrees. Original dirty worktrees, active services, source snapshots, raw captures and checkpoint files remain intact. Superseded local guides/build inputs moved into ignored `04-data-agent/temp/historical-notes/` and `temp/legacy-hf/`; replaced bundle outputs are archived in `temp/build-archive/`. Committed results contain compact scores, a static figure and a compressed full metrics/provenance snapshot. No raw task answers, credentials, model checkpoints or Trackio databases are committed.
