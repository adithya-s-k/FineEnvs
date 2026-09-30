# Multi-harness Trackio logging

Dashboard: https://huggingface.co/spaces/AdithyaSK/multi4-qwen35-2b-trackio (private; sign in).
Current project: `multi4-qwen35-2b-prod-20260915`. Select **`training-full-history`** for every optimizer step from 1 through the live continuation, and `evaluation-curve` for the audited baseline/checkpoint history. Allocation-specific series `training-80608` and `training-79083` are retained. Older projects retain their original records; the combined view follows the actual checkpoint resume chain rather than including every attempted job. Client and replay adapter: Trackio **0.33.0**,
already installed in `.venv312`; training dependencies were not upgraded.

The native Transformers callback logs every optimizer update locally. The launcher
sets `TRACKIO_STORAGE_MODE=jsonl` and `TRACKIO_DIR=job-<id>/trackio`, clears remote
Trackio environment variables, and disables automatic Space publishing/freezing.
This follows Trackio's [FSx/network filesystem guidance](https://huggingface.co/docs/trackio/environment_variables).
`audit/metrics.jsonl` is the durable scalar source; raw capture results, token IDs,
logprobs, loss masks and completion tables also stay local.

The CPU collector `tools/trackio_multi4.py` runs independently of both GPU jobs.
It reuses Trackio's fragment importer and deterministic `log_id` deduplication,
with a version check because that adapter uses the installed storage API. Replaying
the same sources or restarting the collector does not duplicate metric rows.
SQLite writes stay on node-local disk. A consistent backup is atomically saved as
`<run>/trackio/dashboard/<project>.db` before online sync.
Do not point a writable dashboard directly at that published snapshot during training;
copy it into a node-local directory first.

The [native Trackio sync support](https://huggingface.co/docs/trackio/quickstart) publishes
the scalar project to a private Space and private persistent HF Bucket. The pinned
adapter uses `trackio.deploy.sync_incremental` (bulk logging endpoints) for live
updates. Uploading the whole database initially loaded the baseline but left active
dashboard readers stale on subsequent updates; switching to the bulk API resolved
this without touching the trainer. A sync
subprocess has a 180-second timeout and retries on later checks. Only the collector
can be delayed by HF outages; training and evaluation do not wait for it. Polling
is every 60 seconds; initial Space build, uploads, or outages can increase dashboard lag.
`trackio/status.json`, `history.jsonl`, and `sync.log` record local/sync health.
Tokens come from the existing environment or `experiments/.env` only inside the sync
process. They are never placed in run configuration or uploaded logs.

### Late evaluation repair — 2026-09-16

The parent allocation's logger exited at 00:04 UTC, before checkpoint-500/600 version-provenance audits were repaired. The continuation kept publishing live training metrics but only scanned its own checkpoint directory. Remote inspection confirmed that the production project's curve contained steps 0/200/300/400/700; checkpoint 100 was still isolated in the older bounded project.

The active collector now scans explicit `logging.evaluation_sources` for the bounded and parent production allocations as well as the continuation. It retains the exact protocol, full-coverage, harness-version and TiTO gates, deduplicates matching checkpoint scores, and rejects conflicting scores. Pending evaluations are listed in status. Logger completion also checks recovery-job submission receipts from these source directories, including recoveries launched after an older watcher exited.

Eight focused tests passed, including late parent completion, protocol mismatch, deduplication and recovery-job lifetime checks. The operational copy is `multi4-long-prod-cont-20260915/operations/trackio-eval-history-v4/`; only the CPU logger was replaced. Live trainer dependencies, model state and source snapshots were not modified. `trackio/status.json` now includes the latest collected training step, published evaluation steps and evaluation source directories; `sync-receipt.json` verifies remote event contents.

Runs in the project:

- `training-<Slurm id>`: reward, reward dispersion, loss, gradients, learning rate,
  entropy/KL/importance ratio, staleness and row drops, packed-row utilization,
  samples per rollout, token throughput, rollout latency, and weight sync timing.
  Metrics use the actual optimizer step.
- `audit-<Slurm id>`: monitor snapshots, per-harness observed reward, coverage,
  completed/incomplete results, retained/eligible tokens, rows, alert count, and
  eval backlog. Audit freshness follows the watchdog (2 minutes then 10 minutes).
  TiTO measures capture/sequence construction, not complete optimizer consumption.
- `evaluation-curve`: completed baseline at optimizer step **0**, then full fixed
  **250 tasks × 4 harnesses × pass@1** evaluations every **100** optimizer steps and at the final complete checkpoint.
  Scores include harness, difficulty, harness × difficulty and baseline deltas.
  Incomplete evaluations, mismatched protocols/versions, and failed TiTO audits
  are withheld and listed in logger status. Regular saves every **50** are supplemented by hourly recovery saves.

Offline viewing, without network access (copy a fresh snapshot to see later updates):

```bash
RUN=/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-gas64-reliable-20260915
TRACKIO_VIEW=$(mktemp -d /tmp/multi4-trackio-view.XXXXXX)
cp "$RUN"/trackio/dashboard/*.db "$TRACKIO_VIEW/"
TRACKIO_DIR="$TRACKIO_VIEW" .venv312/bin/trackio show --project multi4-qwen35-2b-gas64-reliable-20260915
```

Manual recovery after the collector stops: run the following with the actual training
job ID, adding `--watch` for continuous polling. Omit `--online` for offline-only replay.
The exclusive collector lock prevents accidental duplicate workers.

For superseded training job **78618**, the logging job used the separately frozen
operational fix `operations/trackio-bulk-v2/trackio_multi4.py`. Its hashes and restart
record are beside it. Use that path for recovery of this run. Logging job 78622 was
cancelled after its first live sync stalled; original training sources remain intact.
Current job IDs are in `submission.json`: job 78631 and its companion monitor/watcher
were subsequently cancelled by Slurm root when their CPU node disappeared. Replacements
were submitted as watcher 78644, monitor 78645, logger 78646.

A lightweight login-node supervisor checks support-job liveness every 60 seconds and
resubmits dead CPU watcher/monitor/logger allocations while work remains. It stores
exclusive locks, durable submission intents, restart history and status under `supervisor/`.
Create `supervisor/STOP` to stop recovery before intentionally stopping the run.
It does not diagnose/restart a failed GPU trainer; that requires checkpoint-aware
recovery by the active coding task. It is also dependent on the login host staying up.

```bash
.venv312/bin/python "$RUN/source-snapshot/tools/trackio_multi4.py" \
  --run "$RUN" --train-job 78767 --online
```

The baseline has been uploaded and verified. No synthetic training or checkpoint
scores are mixed into the real project. Online payloads contain scalar metrics and
an allowlisted configuration; no raw task text, completion tables, credentials,
model weights, or capture files are uploaded.

Current job 78767 resumes checkpoint 25 with GAS64 while retaining G8/I8/S4. Its source snapshot includes both logging repairs described below; the separately versioned operational files belong to the parent phase.
Its frozen collector includes the bulk-sync repair. The separate operational collector
`operations/trackio-config-v3/trackio_multi4.py` also restores online configuration:
Trackio 0.33.0's bulk export emits `config=None`, so the collector replays one existing
log ID per run through the native bulk API with the allowlisted config attached.
This preserves metric counts. Model/LR/G8/I8/S4/GAS16 metadata was verified remotely
for training, audit, and baseline runs; `operations/trackio-config-v3/VERIFIED.json`
records the result. `run_config.json` identifies the current collector, and
`submission.json` identifies the current logging allocation.

The current monitor keeps healthy-progress evidence during normal rollout waits and
switches from 120 to 600 seconds after two progressing checks at the configured minimum step or later (step 20 for this continuation).
Any alert resets the closer interval. Its operational launcher is recorded under
`monitoring.launcher_override`; the supervisor preserves it after CPU node loss.

`job-78956/STOP_AFTER_STEP`
requests a save and graceful stop at the next completed optimizer boundary.

### Resolved startup transport incidents

Job 78767 encountered two zero-call failures: a transient Gradio relay 404 and an E2B HTTP/2 error during template lookup. Replaying the original results through the frozen TRL worker confirmed no training rows and reward `None`, including the relay attempt whose native verifier reported zero. Original captures and failed TiTO checks remain unchanged. The monitor can resolve only individually named, investigated zero-call incidents in `monitor/resolved_incidents.json`; new failures, generated tokens, and other fatal findings still alert. Six monitor tests cover those restrictions. The active override is `operations/monitor-incidents-v2/`; support job IDs remain authoritative in `submission.json`.

## Atomic rollout continuation 78831

The loss is a supervised-token mean across the complete optimizer update. GAS4
counts outer packing units; physical forwards can exceed four when a rollout forks.
`audit/optimizer_rollouts.jsonl` records rollout IDs, absolute group IDs, row counts
and supervised-token counts only after the optimizer boundary. This is separate
from the capture TiTO audit. Native forwarded/supervised-token and timing metrics
measure the actual work. Stale admission rejects whole rollouts; those drops remain
visible. Startup cadence is 120 seconds; stable cadence is 600 seconds from step 33
plus two progressing healthy checks. Current support job IDs are in submission.json.

## Outstanding-work bound, continuation 78956

Maximum 16 outstanding rollouts, G8/GAS4/S4. Credits reserve whole groups and
return after consumption or complete-rollout rejection. The new metric
`admission/stale_rollouts_dropped_total` reports exact rollout counts;
`sample/dropped_stale_total` reports exact row counts. In job 78831 the older
`admission/stale_rollouts_dropped` was reduced as a gauge and is not a count;
its row counter remains valid. `admission/outstanding_rollouts_max` verifies
actual pressure against the configured 16. Rejections now have per-rollout
receipts in `audit/rejected_rollouts.jsonl`, alongside optimizer receipts.
Stable-cadence threshold is 63 plus two progressing healthy checks.

## Trackio landing-view repair — 2026-09-16T06:13:26.689484+00:00

Reproduced the user's screenshot in Chromium: the bare Space URL selected `multi4-qwen35-2b-20260914` and plotted `training-78618`, ending at step11. The earlier upload repair had not fixed this UI default. Trackio0.33.0 selects the first returned project; its `show(project=...)` argument only changes the printed launch URL.

Updated only the Space's `app.py` to return the current production project first, preserving all seven historical projects. Reproducible source: `hf/trackio_app.py`; deployed commit `4c72c4ad523075fa6f448027358794bd1f908681`. The dashboard rebuild did not restart trainers.

Fresh Chromium verification of the plain `.hf.space/` URL selected the current project, rendered the reward and evaluation charts, fetched training through step967, and showed all nine baseline/checkpoint points through800 (27.0% pass@1). Browser-fetched values for500/600 were checked as37.0%/31.8%; no JavaScript errors. Evidence and before/after screenshots: `multi4-long-prod-cont-20260915/operations/trackio-ui-20260916/`. Already-open tabs need one reload to apply the new default; their current selection otherwise remains intact.

## Complete training history in Trackio — 2026-09-16T06:22:31.913390+00:00

The current project previously showed only jobs79083 (197–684) and80608 (685 onward); steps1–196 were in older projects. The CPU collector now publishes `training-full-history`, a separate continuous view of the actual resumed training lineage:78647 (1–17),78681 (18–25),78767 (26–30),78831 (31–53),78956 (54–196),79083 (197–684),80608 (685 onward). Abandoned base attempt78618 is excluded because its weights were not resumed.

History follows exact checkpoint boundaries, excludes aggregate final summaries and any parent steps beyond the selected checkpoint, checks for gaps/conflicting records, preserves allocation provenance as `train/source_job_id`, and records each segment's actual training configuration. Earlier recipe changes are not relabelled as one unchanged configuration. Existing allocation traces remain available.

Nine focused tests passed. CPU logger80959 was replaced by80966 through the existing supervisor; trainers were untouched. Chromium fetched and rendered all 973 distinct steps 1–973, plus the full eval curve through800, with no JavaScript errors. Evidence: `operations/trackio-full-history-v5/lineage.json`, `FULL_HISTORY_UI_VERIFIED.json` and `full-history.png`. The combined series updates on the existing60-second logging cadence.
