# Three parallel data-agent runs

Both comparison runs are launched: native OpenCode80626 with controller80627; HF SETA6aa9b6c9f76d6a098a70e786 with controller80605. All live qualification gates passed. The original multi-harness trainer continues independently, including its existing allocation-resume chain. Current state is in [RUN_PROGRESS.md](RUN_PROGRESS.md); full configuration and evidence are in [COMPARISON_READINESS.md](COMPARISON_READINESS.md).

| Run | Trainer allocation | Environment | Checkpoint eval |
| --- | --- | --- | --- |
| Original multi-harness | Existing hopper-prod allocation | Existing E2B / four harnesses | Existing independent 250 × four-harness pass@1 |
| Native standalone OpenCode | New hopper-prod, two H100s | Local native environment, Daytona | Separate two H100s, TP1/DP2, 250 × four harnesses, concurrency50 |
| Native SETA Whitebox | HF Jobs in HuggingEnvs, h200x2 | Existing pinned SETA Space, Daytona | Separate HF A10080GB, TP1/DP1, 250 native SETA tests, concurrency50 |

Each new training allocation has one optimizer GPU and one inference GPU. Both begin from pinned Qwen3.5-2B base weights. They use LR3e-6, eight generations per task, frozen1000 train tasks (150easy/600medium/250hard) in the same task order, and fixed250 test tasks (33easy/118medium/99hard). Native OpenCode uses the qualified async recipe; SETA uses synchronous GRPO/DAPO. Save every50 optimizer steps, retain all checkpoints, evaluate every100 plus the final checkpoint. The 1000-update/24-hour ceilings do not guarantee1000 updates within one allocation.

## Launch chain

Everything below is rooted at `/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/`.

- Native launcher80595 waits for corrected optimizer80593 and separate checkpoint qualification/controller80594. It reads `local-opencode-main-ready-v2/plan.json`, submits the long GPU job and CPU checkpoint controller, and writes their IDs to `long-launches/opencode/state.json`. The prior v3/native80549 main plan is superseded by the explicit-zero-tolerance correction.
- HF launcher80596 waits for checkpoint eval/controller80567, late Trackio replay80586 and final provenance preview80585. It reads `hf-whitebox-main-ready/plan.json`, submits the h200x2 trainer and its36-hour CPU eval controller, and records IDs in `long-launches/whitebox/state.json`.
- `hf/launch_qualified.py` is the canonical launcher. It requires completed live proofs. HF submission intent and owner identity are persisted before the API allocation, allowing an ambiguous response to be adopted without launching another trainer. Repeated successful invocations return the existing IDs.

The launchers have already been submitted; do not run a second launch command. Inspect their `state.json` and `submission.json` first. Existing qualified trainer bundles and Space revisions are unchanged by orchestration.

## Monitoring and isolation

CPU job80636 runs the frozen `monitor_three_runs.py` every600seconds for48hours. Its first live check passed. It follows the original allocation continuation and inherited checkpoint evaluations, new launch receipts, actual local/HF optimizer metrics, TiTO capture/optimizer receipts, Trackio persistence, checkpoint saves and independent eval controllers. It compares the most recent20 optimizer rewards with the preceding20; a decline triggers investigation, not a weight reset. Exact TiTO auditing runs on CPU using the qualified runtime’s capture and sequence code.

The current Markdown summary is [RUN_PROGRESS.md](RUN_PROGRESS.md). Append-only snapshots and action history are under `three-run-monitor/history.jsonl`, `three-run-monitor/HISTORY.md` and `three-run-monitor/events.jsonl`. `three-run-monitor/STOP` stops the observer. This Slurm monitor has no app/chat wakeup channel; alerts remain visible in the Markdown and JSON artifacts.

Known transient dead CPU controllers may restart at most twice using the same frozen controller and state. Ambiguous submissions, failed eval provenance, numerical errors and TiTO failures are recorded for diagnosis; no automatic restart can silently reset training to base weights. The existing original-run continuation restores full native checkpoint state when its allocation ends.

Daytona comparison evals reserve both trainers’ sandbox capacity and run one at a time when required by the measured125-sandbox resource budget. Eval never reloads a training inference endpoint. The original E2B run uses a separate provider. Storage and provider latency are still shared resources, so measured step time is tracked for regressions.

Native/HF CPU controllers forward verified completed evals into the same training metric ledger. Late evals are replayed into the finished trainer’s Trackio database and artifact backup. There is no Trackio Space; comparison logging is an offline native database plus durable uploaded events/backups.
