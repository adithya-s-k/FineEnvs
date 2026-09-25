# PR preparation validation — 2026-09-16

This records validation of the prepared sources separately from the historical learning curves.

| Check | Evidence |
| --- | --- |
| HuggingEnvs CPU regression suite | 149 passed, 1 skipped; 24 subtests. Includes real CPU optimizer grouping, save/resume boundaries, capture budgets, task dispatch, HTTP controls, eval recovery and artifact provenance. |
| Portable archive | 10,606 packaged files hash-verified; source runtime matches the reviewed files; no configured credential values included. No external local experiments checkout required to build. |
| Frozen native grading | 1,250 task configurations verified; all 250 original first-graded baseline answers replayed with identical scores. |
| Local/Hub commands | CLI help, dry-run commands, Python compilation and fatal-error lint passed; generated project index checked. |
| OpenEnv | 2,393 CPU tests passed with unrelated QED service tests excluded; 107 additional upstream MCP integration tests and 65 Gradio/MCP/TBench tests passed after the current-main merge; 39 client/TiTO and 59 rollout/session regressions passed for the final fixes; 334 passed and 7 skipped for client cancellation/discovery/Harbor regressions after the last upstream merge. GitHub CI is green on Python 3.11/3.12. Harbor capture/UI checks include concurrent trace isolation, session budgets and browser layout. |
| TRL | 245 CPU tests passed, plus HTTP controls and pre-commit checks. Main is merged; all PR CI passed, including the distributed GPU smoke. |

## Current training qualification

| Implementation | HF Job | Trainer bundle | State |
| --- | --- | --- | --- |
| Harbor / OpenCode | [6aaa8a06f76d6a098a710a5e](https://huggingface.co/jobs/HuggingEnvs/6aaa8a06f76d6a098a710a5e) | v4 | **Passed**: four nonzero-gradient updates, exact-token retention, native optimizer state, remote restore, changed weights; [receipt](qualification/harbor-v4.json) |
| Native OpenCode | [6aaa7b875527934177ee9d15](https://huggingface.co/jobs/HuggingEnvs/6aaa7b875527934177ee9d15) | v3 | **Passed**: four nonzero-gradient updates, exact-token retention, native optimizer state, remote restore, changed weights; [receipt](qualification/opencode-v3.json) |
| SETA whitebox | [6aaa7f915527934177ee9da4](https://huggingface.co/jobs/HuggingEnvs/6aaa7f915527934177ee9da4) | v3 | **Passed**: four steps, exact-token audit, native optimizer state, remote restore, changed weights; [receipt](qualification/seta-v3.json) |

- **v2**: SHA256 `8b02b40687414905830799a458bf253d3552f9a40860f9980c983fb4ededa45a`, Hub revision `0e59f18b0ddf0df0f46aa8925b4d8bb66aa95bb5`.
- **v3**: SHA256 `6527c25ae379ab10f055577c5b87374c9018df3c8d28bb983f84fbd1e7f6302e`. Hub revision `599efbda7c93056e9d0a6a2a3324d24ac1ba2f3f`. Uses OpenEnv `b13aeb9f8ecd4817e02d3a37c2a9ae15e41710e3` and TRL `8e87edb45eac7c52d749256379714fa40f0eb746`; 10,604 packaged files. Subsequent OpenEnv PR commits preserve verifier warning diagnostics, clarify timeout scope and merge upstream client cancellation/discovery fixes. The later TRL merge changes only a tiny Gemma2 test-model generator; runtime qualification remains tied to the explicit pins above.

- **v4**: SHA256 `d24c3641bdda424259741e27d451430830df1c54e0a31245ac2c8ae6210a44da`, Hub revision `d7622b44f55c65387778327229543d36446e6597`; 10,606 packaged files. Model, OpenEnv/TRL pins, training code and settings match v3. It adds the shared transfer retry path and host baseline-cohort checks.

Harbor and SETA use their existing separately pinned environments. Native OpenCode uses v3, deployed only after confirming no active Jobs used that Space; CPU Basic and sandbox capacity 100 are retained. The active SETA evaluation and Harbor-only Slurm trainer were not restarted. Every future long run still requires proofs matching its own exact bundle, environment and baseline; these receipts do not waive that gate for another bundle.

The final v3 SETA smoke completed successfully. The earlier successful v2 [Job 6aaa77a65527934177ee9c34](https://huggingface.co/jobs/HuggingEnvs/6aaa77a65527934177ee9c34) and its [receipt](qualification/seta-v2.json) are retained independently. Both have two nonzero-gradient updates and four completed optimizer steps.

The final host-launcher regression suite additionally verifies that native diagnostic and four-harness comparison baselines remain separate, that checkpoint curves receive a matching baseline at step 0, and that changed score files fail validation. These host-only admission/reporting changes do not alter the GPU trainer runtime used by the qualification bundle.

## Earlier qualification attempts

The first trainer bundle was `ccfe97822cf7c88931acda8a4894bd7515e40a939ebfcc4c20c14db45e607de3`, uploaded to `HuggingEnvs/data-agent-daytona-repro` at revision `3808a6d5c48320b5e7745c877dc9b7ed2819310b`. Source pins are in [sources.json](../hf/configs/sources.json). The OpenEnv runtime pin includes the TiTO/UI changes; later OpenEnv PR commits update documentation, optional tests and merge newer upstream MCP behavior.

These jobs qualify the new trainer against existing separately pinned Spaces. The first attempts did not restart or upgrade those Spaces; the later idle native OpenCode update is recorded above. A smoke proves four optimizer updates with a checkpoint-2 remote restore; it is not a new baseline or evidence of a reward gain.

| Implementation | Job | GPUs | State |
| --- | --- | --- | --- |
| Harbor / OpenCode | [6aaa75bb5527934177ee9b8b](https://huggingface.co/jobs/HuggingEnvs/6aaa75bb5527934177ee9b8b) | A100 ×4 allocation; two used | Failed before optimizer startup: missing endpoint directory |
| Native OpenCode | [6aaa75bb5527934177ee9b8d](https://huggingface.co/jobs/HuggingEnvs/6aaa75bb5527934177ee9b8d) | A100 ×4 allocation; two used | Failed before optimizer startup: missing endpoint directory |
| SETA whitebox | [6aaa75bbf76d6a098a710867](https://huggingface.co/jobs/HuggingEnvs/6aaa75bbf76d6a098a710867) | H200 ×2 | Failed before optimizer startup: missing endpoint directory |

The clean-Job failure is fixed by creating the endpoint/log parent directories in `serve/vllm.sh`. The failed cohort is preserved. A second cohort exposed a deployed-server API mismatch in both async arms; those two jobs were stopped before optimizer updates (`6aaa77a65527934177ee9c30`, `6aaa77a65527934177ee9c32`). The Harbor client now omits only default provider/eval arguments. Explicit settings are still sent. The idle native OpenCode Space was upgraded to accept and enforce sampling. Training submission now checks the remote tool schema before allocating a GPU Job.

Completion requires `training_smoke_verified.json`: exact capture, retained supervision, native optimizer state, remote restoration and changed weights. Pending jobs are not counted as passed.

The v3 Harbor Job later ended with an HF Xet upload `TimeoutError` after completing all four updates. Its final cleanup published both checkpoints. A separate audit reconciled 37/37 completed captures, all 19,880 eligible supervised tokens and 25 optimizer rollout receipts; this does **not** waive the failed integrated qualification. The [failure receipt](qualification/harbor-v3-upload-failure.json) is preserved. The shared publisher now retries transient transport/429/5xx errors up to three attempts, keeps ready markers last, and allows an hour for an already-active full-checkpoint upload during shutdown. Permission and validation errors remain fatal. Fault-injection and full CPU regression tests passed. The v4 Harbor rerun completed successfully, including both checkpoint publications.

## Preserved material

Preparation uses separate Git worktrees. Original dirty worktrees, active services, source snapshots, raw captures and checkpoint files remain intact. Superseded local guides/build inputs moved into ignored `04-data-agent/temp/historical-notes/` and `temp/legacy-hf/`; replaced bundle outputs are archived in `temp/build-archive/`. Committed results contain compact scores, a static figure and a compressed full metrics/provenance snapshot. No raw task answers, credentials, model checkpoints or Trackio databases are committed.
