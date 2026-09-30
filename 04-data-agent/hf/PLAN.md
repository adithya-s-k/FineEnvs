# Daytona whitebox and blackbox training on HuggingEnvs

Prepared 2026-09-15 UTC. Deployment target: **HuggingEnvs**. All new resources belong to that organization. Updated by the two-Space instruction: Trackio stays in training Jobs; environment Spaces, reproduction data and raw artifacts stay private. Daytona supplies the task sandboxes; Hugging Face supplies training, inference, coordination and artifact hosting. The original multi-harness E2B job 79083 continues on hopper-prod.

This is the deployment plan, not a claim that the HF training jobs have started. `configs/deployment.json` contains the machine-readable configuration. Frozen inputs are packaged from `experiments/daytona_harness_comparison/logs/20260915/` in the workspace. Runnable HF launchers live in this directory.

## Existing evidence and remaining gates

Both Daytona baselines completed on the cluster and retain their original provenance:

| Evaluation | Correct / evaluated | Pass@1 |
| --- | ---: | ---: |
| Blackbox evaluation: OpenCode | 32 / 250 | 12.8% |
| Blackbox evaluation: Claude Code | 42 / 250 | 16.8% |
| Blackbox evaluation: Codex | 38 / 250 | 15.2% |
| Blackbox evaluation: Mini-SWE-Agent | 47 / 250 | 18.8% |
| Blackbox evaluation: overall | 159 / 1,000 | 15.9% |
| Whitebox native bash/SETA | 41 / 250 | 16.4% |

All selected captures passed TiTO; blackbox harness versions matched the fixed pins. Provider lifecycle, verifier, whitebox tools and model-only smokes passed. The whitebox recovery ramp completed 168 cases at concurrency 100 in 693.4 seconds. These results establish the existing implementation's baseline, not HF runtime readiness or simultaneous two-arm throughput.

The ATL optimizer smokes 79195/79197 were canceled after severe filesystem startup stalls; owner-scoped cleanup jobs 79196/79198 completed. Neither comparison arm has passed its optimizer/checkpoint/resume gate. No new ATL or hopper-extra allocation will be used.

HF organization Jobs access was verified using `HF_API_KEY` from `experiments/.env`. The desktop's ambient OAuth token lacks the required org Jobs scope, so deployment explicitly selects the configured write token. Credentials must be injected as Job/Space secrets and never included in the reproduction bundle.

## Organization resources

| Resource | Target | Purpose |
| --- | --- | --- |
| Reproduction dataset repo | `HuggingEnvs/data-agent-daytona-repro` | Immutable source bundle, task manifests and data, lockfiles, launch configuration, baseline evidence and instructions |
| Artifact Bucket | `HuggingEnvs/data-agent-daytona-artifacts` | Full checkpoints, capture archives, attempt ledgers, logs, coordinator state and ready manifests |
| Blackbox environment | `HuggingEnvs/data-agent-opencode-blackbox-env` | One Task API and interactive UI for training and evaluation |
| Whitebox environment | `HuggingEnvs/data-agent-seta-whitebox-env` | One native bash/SETA Task API and interactive UI for training and evaluation |
| Blackbox model repo | `HuggingEnvs/qwen35-2b-daytona-opencode` | Verified export checkpoints and model card |
| Whitebox model repo | `HuggingEnvs/qwen35-2b-daytona-whitebox` | Verified export checkpoints and model card |
| GPU Jobs | `namespace="HuggingEnvs"` | Two independent trainers, disposable smokes, base evals and checkpoint evals |
| CPU coordinator Job | `namespace="HuggingEnvs"` | Checkpoint discovery, eval submission, metric replay, monitoring and bounded recovery |

Artifact/logging persistence is independent of the environment UI. An environment restart interrupts its in-flight RPCs, which must be treated as ungraded infrastructure failures; use recovery and avoid redeploying active services. The coordinator runs as a CPU Job and restores its durable ledger on restart; a Space presents its state. Do not use ZeroGPU for persistent training or serving.

## Matched experiment configuration

| Setting | Blackbox OpenCode | Whitebox bash/SETA |
| --- | --- | --- |
| Initial weights | Pinned Qwen/Qwen3.5-2B base | Same base |
| Model revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Same revision |
| Trainer | Frozen AsyncGRPO, atomic rollouts | Frozen native synchronous GRPO, DAPO, beta=0 |
| Training task pool | Fixed 1,000: 150 easy / 600 medium / 250 hard | Same task identities and reference order |
| Training harness | OpenCode only | Native bash/SETA tools |
| LR / generations / seed | 3e-6 / 8 per task / 0 | Same |
| Precision / optimizer | BF16 / paged_adamw_8bit | Same |
| Gradient checkpointing | Enabled, nonreentrant | Same |
| Sampling | Temperature 0.8, top_p 1, top_k off, thinking off | Same |
| Training context / completion | 131,072 / 16,384 | Same configured sizes; whitebox completion includes masked tool results |
| Admission | Worker ceiling 32; max outstanding 16; staleness 4 | Synchronous group of 8, no async staleness parameter |
| Batch / accumulation | 4 / 4; native atomic admission can vary actual microbatches | 1 / 8 |
| Token assembly | Packing target 40,960; hard row ceiling 131,072; fork threshold 0 | Native exact token tool loop and tool-result masks |
| Save / eval cadence | Save every 50 optimizer steps, eval every 100 plus final | Same |
| Recovery save | Also at most one hour between recovery checkpoints | Same |
| Run budget | Up to 1,000 optimizer steps; 24-hour Job, graceful 82,200-second training budget | Same |
| Checkpoint evaluation | 250 tasks × four harnesses = 1,000 pass@1 cells | 250 tasks × native bash/SETA = 250 pass@1 cells |

Both consume the same 1,000-task pool, rather than silently substituting the earlier 100-task mention. Task count is not optimizer-step count. Log actual task coverage, successful rollouts and supervised tokens alongside steps and elapsed time. Sync and async batch/loss semantics differ; a harness-quality-only causal claim would overstate this comparison. Also record E2B versus Daytona and H100 versus H200 when comparing against the original multi-harness arm.

## Execution sequence

1. **Package the exact experiment.** Copy the frozen OpenEnv, TRL, whitebox package, serving helper and required experiment tools into a portable bundle. Include the actual uncommitted patches and hashes, not merely Git commit IDs. Copy the frozen 36 MB task tree, selection/order manifests and canonical baseline evidence. Replace FSx/Slurm assumptions only in the HF copy. Use stable HF run/job IDs for sandbox labels and cleanup. Pin container image digest, Python, model revision, harness versions and all dependency hashes. Train and environment services retain separate virtual environments.

2. **Validate a fresh runtime on HF.** A CPU preflight verifies artifact download/upload, source hashes, task identities, imports and CLI construction before GPU allocation. Dependency resolution already found two inherited incompatibilities: the trainer had NumPy 2.5.1 and websockets 16.1.1, while vLLM/mistral-common and Harbor/Supabase require older versions. The clean training lock resolves NumPy 2.3.5 and websockets 15.0.1. Both hashed lockfiles resolve successfully, but installation/import/GPU behavior still need validation. Do not mutate either live cluster environment.

3. **Run HF task and TiTO smokes.** Start a local vLLM server through the existing `serve_vllm_tunnel.sh`, adapting paths only. Verify actual engine prompt IDs, completion token IDs, one finite sampled logprob per token, assistant-only loss masks, exact fork assembly and retention of eligible supervision. Exercise Daytona create, task data transfer, model/tool calls, known-correct/wrong verifier outcomes and deletion. Exercise all four blackbox evaluation harnesses plus native whitebox tools. Confirm the public capture endpoint reaches the intended local job, and pin all harness versions.

4. **Establish HF step-zero results and scale.** Retain the completed cluster baselines as historical reference. Because HF changes hardware and resolves runtime dependencies, run one HF base evaluation per arm on the same fixed 250-task set: 1,000 blackbox cells and 250 whitebox cells. Use the same sampling and first-graded selection rule. Ramp concurrent requests 8 → 32 → 100, measuring completions/minute, request latency, proxy errors, sandbox lifecycle latency, GPU queueing and cleanup. The ramp can contribute to that same ledger; it must not select a better result. Establish joint sandbox capacity before two trainers and two 100-concurrency eval jobs run together. The earlier E2B limit of 600 does not establish Daytona CPU/RAM/disk or rate limits.

5. **Pass optimizer/save/resume for each arm.** Use a disposable h200x2 Job: perform two optimizer updates, save checkpoint 2, start a fresh trainer from it, reach step 4 and save again. Require finite metrics, nonzero learning signal, actual parameter changes, saved optimizer/scheduler/RNG, native rollout cursor where applicable, checkpoint hashes, exact resume parameter continuity and passing TiTO. Exercise download-and-resume of a remotely persisted checkpoint before treating recovery as ready. Smoke weights do not initialize the long run.

6. **Start both long runs from the pinned base.** Candidate independent h200x2 allocations use GPU 0 for rollout vLLM (TP1/DP1) and GPU 1 for training. Also benchmark sharing one a100x4 allocation as two GPU pairs; it costs less for both arms together but couples their allocation lifecycle. Choose using measured cost per useful update. Model weights synchronize only within that Job. Environment and capture services run in two private Docker Spaces, one per arm, sharing train and test catalogs with explicit Daytona ownership labels and reserved training capacity. The original E2B multi-harness trainer continues independently. A failed smoke blocks only that arm's long launch.

7. **Launch checkpoint evaluation asynchronously.** Save the full checkpoint to fast local disk. An independent uploader transfers its immutable files and hashes, then publishes a ready marker last. The coordinator discovers complete markers, records an eval identity `(run, arm, step, checkpoint hash, protocol hash)`, reconciles existing HF Job labels to avoid duplicate submissions, and launches a separate a100-large Job for each due evaluation. First benchmark one A100 80GB at TP1/DP1, concurrency 100. HF has no a100x2 flavor; a larger allocation needs measured throughput justification. Each evaluation uses exactly the same environment Space as its arm’s trainer, with its own inference endpoint and per-rollout generation budget. One active eval per arm; queue later checkpoints without skipping them. If Daytona capacity requires it, serialize the two eval jobs while reserving training capacity. Training does not wait for eval completion.

8. **Monitor, recover and finish.** Check startup every two minutes, then every ten minutes after at least ten optimizer steps and two progressing checks. Monitor loss/gradients, reward, group variance, staleness, row expansion, token retention, throughput, inference queues, sandbox failures, checkpoint publication and eval coverage. Treat model tool mistakes separately from infrastructure failure. Infrastructure eval retries preserve the first graded answer, including zero. Bound training recovery to two automatic resumes per arm; TiTO corruption, nonfinite updates or invalid checkpoint provenance stop the affected arm for diagnosis. A remote resume restores the full native state; in-flight rollouts may be regenerated. At completion, persist final state, run the final eval if needed, export verified model weights/cards and terminate only owner-labelled resources.

## Serving, logging and durability

Reuse vLLM 0.25.1 with server dev mode, `processed_logprobs`, raw token-ID return, no prefix caching, `qwen3_xml` tool parser, `qwen3` reasoning parser with thinking disabled, Triton GDN prefill and FlashInfer sampler disabled. The model stays local to its GPU Job. Daytona blackbox harnesses reach only that Job's capture proxy through the validated tunnel. HF authenticated exposed URLs are not automatically usable by sandbox agents, so verify the complete route rather than assuming endpoint reachability.

Every evaluation keeps the same 33 easy / 118 medium / 99 hard tasks, 4,096-token per-call output cap, 17 model-call limit and 600-second episode deadline. Publish pass@1 only when all expected fixed cells are graded, TiTO and version checks pass, and task/checkpoint hashes match. Report infrastructure attempts separately from task failures.

The optimizer writes local append-only metrics. A collector inside the training Job replays them into local Trackio storage with stable event IDs preventing duplicate charts after resume. The user requested no Trackio deployment on Spaces; its database and events persist with the training artifacts. Persist JSONL, SQLite backups, raw captures and score ledgers in the organization Bucket. Trackio network calls never occur in the optimizer path. Log per-harness/per-difficulty pass@1, reward, gradients, throughput, stale rollout rejections, rows per rollout, supervised tokens, task coverage and artifact-upload lag. Dashboard alerts are persistent; this does not claim autonomous chat wakeups.

Full checkpoints include optimizer, scheduler, RNG and native cursor state. Export-only model repositories are not the recovery source. Keep every 50-step checkpoint remotely; delete a local copy only after remote publication is verified. Checkpoint serialization itself takes time, and uploads share host/network resources, so measure their overhead and throttle the background uploader. Separate eval compute removes GPU contention but is not a literal zero-overhead guarantee.

## Cost and completion criteria

HF's live hardware API on 2026-09-15 lists h200x2 at approximately $10/hour per Job. Two trainers therefore cost approximately $240 for 12 hours or $480 for 24 hours, plus $2.50 per single-A100 eval-Job-hour. A shared a100x4 allocation for both training pairs costs $10/hour total, pending optimizer benchmarks. Smokes, HF baseline evaluations, coordinator/Spaces, Daytona, storage and transfer are additional. H200 differs from the original H100 hardware; preserve that distinction in throughput reports.

Deployment is complete when both long runs have verified nonzero updates; checkpoints survive a Job restart; online/offline metrics agree; a checkpoint-triggered eval starts on separate compute; fixed coverage/TiTO gates work; and the published recipe can recreate the run without FSx or Slurm. A configured Space or successful model call alone is not completion.

Sources: [HF Jobs API and hardware](https://huggingface.co/docs/huggingface_hub/guides/jobs), [HF storage Buckets](https://huggingface.co/docs/hub/storage-buckets), [Space storage](https://huggingface.co/docs/hub/spaces-storage), [Daytona resource and rate limits](https://www.daytona.io/docs/limits).

Consolidation capacity update (2026-09-15): the authenticated Daytona usage API
reports 500 GiB RAM in EU, so 4-GiB task sandboxes have a joint ceiling of 125.
Current static reservations are blackbox64 (train16/eval48) and whitebox61
(train8/eval53), with 1,024 native transport sessions per Space. The 100-concurrency
per-arm target requires a larger quota or serialized evaluation. Do not interpret
E2B's 600-sandbox limit as Daytona capacity. The user approved deletion of the
separate Daytona Trackio Space and four former train/eval Spaces on 2026-09-15;
all five are now deleted, with repository snapshots backed up locally. Historical
evaluation artifacts remain in their separate Buckets. Only the blackbox and
whitebox shared environment Spaces are deployment targets for this experiment.

Hardware update (2026-09-15): both shared environment Spaces use CPU Basic by
explicit user request. Their slugs and titles omit the sandbox provider name.
Health and UI readiness passed after the switch; repeat the concurrency ramp on
this hardware before relying on the configured admission ceilings. GPU training
and inference remain in separate HF Jobs.
