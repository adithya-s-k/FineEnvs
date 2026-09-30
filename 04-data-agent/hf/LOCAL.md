# Local comparison runs

**Latest placement (2026-09-15):** Main Whitebox training is planned on HF Jobs `h200x2`; asynchronous native OpenCode remains on hopper-prod. The user additionally requested local Whitebox qualification: job80555 uses two H100s and the same portable bundlef4a288e as the HF Whitebox smoke6aa9a487f76d6a098a70e3d2. Both are four-update/save/remote-resume tests. Native OpenCode80549 has completed the full optimizer smoke. See `STATUS.md` for live progress; these are separate from long-training launches.

The local launcher reuses the HF evaluator, native trainers, TiTO audits, checkpoint publisher, remote restore, and Trackio collector. It stages isolated source under the experiment output directory; existing running source snapshots are not changed. Environments bind to loopback on the GPU node. Daytona agents reach that node's capture proxy through its existing Gradio tunnel.

| Setting | Local baseline | Local optimizer smoke |
| --- | --- | --- |
| Partition | hopper-prod | hopper-prod |
| GPUs | 2 H100, TP1/DP2 | 1 inference H100 + 1 trainer H100 |
| Model | Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | Same |
| Environment | Actual standalone OpenCode, Daytona | Standalone OpenCode or native bash/SETA, Daytona |
| Tasks | 250 fixed test tasks | Frozen training schedule |
| Concurrency | **50 from the start, no ramp** | Native atomic rollout recipe / SETA synchronous G8 |
| Metric | pass@1, first graded result per task | Real optimizer, TiTO, checkpoint and restore checks |
| Steps | All 250 graded cells required | 1–2, save/upload 2, remote restore, 3–4 |
| Learning rate | — | 3e-6 |
| Logging | Captures, errors, throughput, GPU telemetry | Native Trackio locally plus artifact persistence |

The dataset contains 33 easy, 118 medium and 99 hard test tasks. Training uses the existing 1,000-task manifest (150 easy, 600 medium, 250 hard). The evaluator does not import partial results from canceled HF cohorts. A graded zero is retained; infrastructure failures remain ungraded and can be retried. Every returned capture, including ungraded attempts, is saved.

`cluster.py` accepts `--arm opencode|whitebox`, `--phase baseline|smoke`, `--bundle`, `--out`, `--env-file`, `--submit`, and optional `--dependency afterok:JOB`. Use a new output directory for each staged runtime. The corrected Whitebox portable bundle is under `experiments/daytona_harness_comparison/logs/hf-20260915/whitebox-h200-package-fix/bundle`; its local qualification copy is under `local-whitebox-smoke-v2/repro`. Reuse each passing smoke's exact source root for a subsequent long run.

The frozen source and transformed manifest are recorded in `launch.json` and `repro/local_manifest.json`. Live artifacts are under `repro/outputs/local-ROLE-ARM-JOB/`. `status.json` is a runtime status; only `canonical_scores.json` with complete coverage certifies a baseline, and only `training_smoke_verified.json` certifies optimizer/save/remote-resume validation. These are separate gates.

Use `LOCAL_INFERENCE_PORT` for the API address. `VLLM_PORT` controls vLLM internals and must not be repurposed as the API port; doing so caused the initial DP2 startup collision. The wrapper now passes the HTTP port with `--port` and lets vLLM allocate its internal worker ports. [vLLM environment-variable documentation](https://docs.vllm.ai/en/latest/configuration/env_vars/).

Native OpenCode's long agent command uses the backend's existing background-process API. A real agent deadline terminates the process and preserves the captured trajectory for grading; transport errors remain ungraded. The real Daytona process test and six native regressions passed. Whitebox's reset control endpoint accepts vLLM's empty successful response. Its trajectory audit matches exact context, tokens and logprobs to distinct engine call occurrences, including duplicated GRPO samples and truncated final spans. GPU smoke completion is tracked in `STATUS.md`; code readiness alone is not a training qualification.

The long-run recipe remains save every 50 updates and independent evaluation every 100. Both local optimizer and independent checkpoint-evaluation qualifications have passed. `cluster.py` exposes baseline/smoke staging; `local_long.py` prepares the main run from the exact qualified runtime. Use a new output directory, both proof paths, and an explicit `--submit` to allocate the trainer and CPU controller. The prepared OpenCode plan remains unsubmitted.

`local_long.py` prepares long-run scripts from a passing optimizer smoke's exact source root, preserving its source identity. `local_followup.py` adds a separate Slurm checkpoint-evaluation controller: native OpenCode-trained weights use all four Harbor adapters; Whitebox-trained weights use native SETA. It restores immutable, hash-verified inference files and uses independent TP1/DP2 GPUs at concurrency50. Its service uses58total slots and8reserved slots to provide50eval slots through the existing shared admission policy; the separate global admission check reserves24slots for the two main trainers. An attempted zero reservation was rejected before any rollout; that failed job is archived.

Use `local_long.py --qualify-checkpoint-eval` to prepare a short controller plan after an optimizer smoke passes. This selects checkpoint4, retaining checkpoint2 for restore. The actual four-harness qualification must grade and audit both fixed tasks through all four pinned harnesses; it remains distinct from a full1,000-cell pass@1 evaluation. The main cadence is unchanged: save50/eval100. `configs/cadence_validation.json` records checks against both actual training command builders and controller selection logic. Six local admission/qualification tests and13checkpoint/dispatch tests pass; live qualification is recorded separately.

`hf_followup.py prepare --training-job JOB --bundle BUNDLE --env-file ENV --out NEW_DIR --coordination-dir SHARED_DIR [--submit]` freezes a CPU coordinator for HF Whitebox. For a smoke job it waits for a successful four-update optimizer proof, then evaluates only the remotely verified checkpoint4 on an independent HF allocation. The evaluator inherits the actual training Space's separately qualified bundle pin. Main runs retain the50/100cadence. This coordinator shares the local eval admission lock, keeps its own immutable source/configuration hashes and never launches a trainer.

Both local checkpoint evaluations passed (OpenCode80565: eight cells; SETA80576: two cells). The main CPU controllers have a36-hour allocation, covering a24-hour trainer plus eval completion. HF `deploy.py job --dry-run` checks completed baseline, optimizer and checkpoint-eval proof before saving a job preview; it does not allocate GPUs. Independently pinned HF environments use `--external-checkpoint-coordinator` plus `hf_followup.py`.

`finalize_hf_plan.py` verifies the completed HF checkpoint qualification, writes the exact long-job preview and launch command, and submits nothing. The current read-only CPU check80585 is queued after controller80567. Its prepared configuration changes only orchestration (H200 placement and eval admission); the trainer/evaluator archive is the same one used in the successful qualification.
