# Daytona comparison bring-up — 2026-09-15

The goal is two isolated comparison arms alongside the existing four-harness training run:

| Arm | Training | Evaluation at baseline and every 100 steps |
| --- | --- | --- |
| Blackbox OpenCode | Current OpenEnv/Harbor exact-TiTO AsyncGRPO pipeline, OpenCode only | Fixed 250 test tasks × OpenCode, Claude Code, Codex, mini-swe-agent; 1,000 pass@1 cells |
| Whitebox bash/SETA | Native synchronous GRPO tool loop | The same 250 test tasks through bash/SETA; 250 pass@1 cells |

Both use Daytona, pinned Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, the frozen task instructions and binary Harbor verifier, LR 3e-6, G=8, temperature 0.8, top_p=1, top_k disabled and thinking disabled. Planned checkpoints are every 50 steps, with evaluation on separate GPUs every 100. The current training pool has 1,000 tasks (150 easy, 600 medium, 250 hard); the later request to proceed with the matched comparison is implemented using this same 1,000-task pool. The earlier mention of 100 tasks is retained in the configuration history; no reduced subset is substituted. The test set is unchanged: 33 easy, 118 medium, 99 hard.

## Completed gates

- Native Harbor/Daytona provider: three repeated sandbox lifecycles passed file upload/download, shell output and nonzero exit code, known-correct and known-wrong verifier outcomes, and deletion.
- Whitebox HTTP/MCP tools: eight cases across four tasks passed bash, read, write, edit, grep, glob, ls, task identity and binary grading.
- Blackbox model smoke: two test tasks × all four harnesses, eight graded rollouts. All eight captures passed the original exact-token/logprob/mask and lossless sequence-assembly audit. The two shuffled tasks were both unsolved by all four harnesses; this small smoke is not a baseline estimate.
- Whitebox model smoke: two graded rollouts through the frozen TRL tool loop. Sampled token IDs and logprobs were preserved; tool-result tokens were masked. One task was solved. This is not a baseline estimate.
- Serving preflight: two H100s, TP=1/DP=2, correct frozen task identities, engine token IDs/logprobs and public/private capture-proxy identity verified.
- Daytona admitted 40 simultaneous sandboxes during the first ramp: 32 blackbox plus eight whitebox. Completion/throughput at 100 slots is still being measured.

## Work still running

The full 1,000-cell blackbox and 250-cell whitebox baselines are in progress. **No completed Daytona baseline score is claimed yet.** The controller preserves the first graded result, including zero, and retries only ungraded infrastructure failures. It audits captured supervision before increasing concurrency.

Optimizer tests are prepared but gated on completed baselines. Each arm must complete two updates, save checkpoint 2, resume and reach checkpoint 4, with finite metrics, token/mask provenance, optimizer state and parameter-change evidence. Passing model-only smokes does not establish that optimizer updates are correct.

## Changes and comparison limits

- Current Harbor already contains the `tar --no-same-owner` fix discussed in [issue 1959](https://github.com/harbor-framework/harbor/issues/1959#issuecomment-5584356565); related [PR 2043](https://github.com/harbor-framework/harbor/pull/2043) remained open when checked. The installed fix passed real verifier transfers as host UID 150234.
- Blackbox enables Harbor's native cached Daytona snapshots. The current persistent OpenEnv event loop is retained. Whitebox uses one persistent I/O loop so Harbor's shared async Daytona client is not reused after its loop closes.
- Whitebox reuses the existing tool surface and frozen Harbor tasks/verifier. Its previous shaped reward is bypassed. Ungraded episodes return NaN to the existing native GRPO exclusion path, rather than becoming incorrect-answer rewards.
- Whitebox evaluation calls the actual frozen `GRPOTrainer._tool_call_loop` and tool-suffix builder. It sends raw token IDs to vLLM, checks the returned engine prompt IDs and sampled-token/logprob pairing, and checks every supervised output token and masked tool token.
- The frozen native vLLM generation helper assumes blocks of G identical prompts. After tool calls, histories can differ. The local whitebox training wrapper disables that grouping for distinct histories and asserts that each returned engine prompt equals the requested prompt. Three regression cases pass; actual optimizer validation remains pending.
- Sync GRPO uses one rollout per physical training row, DAPO token normalization and batch size 1 × accumulation 8. The async arm retains its existing atomic-rollout admission, packing and staleness rules. These are recorded differences, not identical update semantics.
- Whitebox's 16,384-token whole-episode completion budget includes masked tool results. Evaluation caps each generation at 4,096 tokens and allows at most 17 model calls. Blackbox retains its existing per-call and harness limits. The bash/SETA output clipping and submission tool are inherent harness differences.

## Evidence and isolation

Durable evidence root:

`experiments/daytona_harness_comparison/logs/20260915/`

- `comparison.json`, `reference_config.json`, `reference_schedule.json`, `opencode_schedule.json`: configuration and matched task order.
- `train_manifest.json`, `test_manifest.json`, `datasets/`: frozen task identities and hashes.
- `source_input_hashes.json`, adapter change records, `bringup_sources_sha256.json`: source provenance.
- `provider_smoke.json`, `whitebox_tools_smoke.json`, `blackbox/smoke_tito.json`: completed gate reports.
- `whitebox/baseline/captures/`: native loop IDs, sampled logprobs and loss masks.
- `blackbox/traces/`, `blackbox/captures/`, `whitebox/baseline/attempts.jsonl`: durable attempts, including exclusions.

The user subsequently authorized migration of the reference run and starting all three long runs in parallel after the comparison gates. **No `hopper-extra` allocations are permitted on 2026-09-15 UTC.**

Reference job 78956 stopped cleanly at checkpoint 196. Replacement **79083** on `hopper-prod` loaded the saved model, optimizer, scheduler, RNG and native rollout cursor, and passed checkpoint 200. Its checkpoint-200 eval **79092** is running separately on `hopper-prod`. Reference CPU support jobs are 79084–79087, with a login-node supervisor recovering failed support allocations. The source recipe and hyperparameters are unchanged; the replacement allocation uses 16 CPUs and 256 GiB with two H100s. The native resume cursor may regenerate pending groups; this is not an exactly-once replay claim.

Daytona service **79079** and controllers 79088/79096 were stopped after connection failures under load. The incident included WebSocket keepalive timeouts, pooled failed connections and an async close override that discarded its awaitable. Whitebox also stopped its I/O loop without closing MCP sessions and used a shorter inner HTTP timeout. Local isolated fixes passed nine targeted transport, generation-routing and deadline tests. Replacement service **79116** runs on `hopper-prod` with two H100s, TP1/DP2 and eight CPUs (previously four); model, sampling and task identities remain pinned. Preflight verified both replicas and the capture endpoint identity. CPU controller **79119** is repeating measured ramps before returning to concurrency100. Old-owner cleanup79091 verified zero remaining sandboxes; cleanup79118 covers the replacement. All previously graded cells are retained; infrastructure failures remain separate attempts. At10:25UTC the baselines had218/1000blackbox and49/250whitebox graded cells; these are progress counts, not completed baseline estimates. The whitebox recovery ramp subsequently graded8/8 at concurrency8. Sustained concurrency100 and optimizer/save/resume validation remain outstanding.

Both long configurations and independent eval/monitor/Trackio launchers are prepared under each arm's `training-long/`. They retain LR 3e-6, G=8, saves50, evals100, a 1,000-step ceiling and a 24-hour allocation with a graceful 82,200-second training budget. Each requests two H100s, eight CPUs and 256 GiB. Long training starts from the pinned base, separately from the disposable optimizer smoke. Online Trackio reuses the existing private Space with a distinct project per arm, and network sync runs on CPU allocations. The long runs are authorized, but have **not yet passed the full baseline and optimizer smoke gates**. All changes remain local; no pushes.

## Whitebox baseline complete — 2026-09-15 10:53 UTC

Pinned Qwen3.5-2B with Daytona bash/SETA scored **41/250 = 16.4% pass@1** on the fixed test set. All250 graded trajectories passed native engine-token, sampled-logprob and tool-mask checks. Easy13/33 (39.4%), medium22/118 (18.6%), hard6/99 (6.1%). Preserve the first graded attempt, including incorrect answers;11 earlier infrastructure attempts were excluded and retried. The recovery sequence graded8/8 at8slots,32/32 at32slots and168/168 at100slots; the last stage took693.4seconds. This completes evaluation scalability for this arm, not its pending optimizer/save/resume smoke. Canonical evidence: `whitebox/baseline/canonical_scores.json`, `ramp.json`, `attempts.jsonl` and the corresponding captures under the comparison root.

At10:53UTC blackbox had532/1000graded cells and had passed64/64new recovery cases at32slots before returning to100. Reference training79083 reached242; independent checkpoint200 eval79092 had919/1000graded cells. One earlier training event at218 rejected four entire over-stale rollouts (67rows); no later stale rejection was recorded through242. The two Daytona optimizer smokes and long training jobs remain unsubmitted.

## Both baselines complete; optimizer smokes allocated — 2026-09-15 12:06 UTC

The blackbox Daytona baseline is complete: **159/1000 = 15.9% pass@1**, with all 1,000 captures passing TiTO and all four harness versions matching the reference. Per-harness scores: opencode: 32/250 (12.8%), claude-code: 42/250 (16.8%), codex: 38/250 (15.2%), mini-swe-agent: 47/250 (18.8%). The 864 ungraded attempts remain in the history, including the earlier transport incident; only the first graded outcome per fixed cell is selected. Whitebox is complete at 41/250 (16.4%), with all 250 TiTO checks passing. CPU controller 79119 completed successfully.

Both optimizer/save/resume smoke allocations are running on `hopper-atl`: blackbox **79195** on ip-10-53-196-100 and whitebox **79197** on ip-10-53-196-194, with two verified H100 80GB GPUs, eight CPUs and 256 GiB each. Their inference services are starting; no optimizer success or long-run launch is claimed yet. Cleanup jobs 79196/79198 are owner-scoped. Completed baseline server 79116 was cancelled to release two `hopper-prod` GPUs; cleanup 79118 covers that owner. Existing trainer 79083 is unchanged. Its checkpoint-200 eval 79092 ended with 999 graded cells, so recovery **79199**, cleanup 79200, preserves those results and retries only Claude Code test index 170.

## HF deployment plan under HuggingEnvs — 2026-09-15

The user moved the two new comparison arms to HF Jobs and Spaces and selected **HuggingEnvs** as the organization for all new resources. ATL smokes 79195/79197 were canceled after filesystem startup stalls; cleanup 79196/79198 completed. Neither optimizer smoke passed. The original multi-harness training job 79083 continues on hopper-prod.

The plan and machine-readable configuration are in `experiments/daytona_harness_comparison/hf_repro/PLAN.md` and `deployment.json`. They specify independent h200x2 training Jobs, separate TP1/DP2 checkpoint eval Jobs, a CPU coordinator, private organization Spaces/Buckets, durable checkpoints and native Trackio replay. HF organization Jobs access was verified with the existing configured write token. Hash-locked trainer/environment dependencies resolve, with required trainer NumPy/websockets compatibility adjustments; fresh-runtime validation remains pending. The plan includes an HF step-zero baseline to accompany the existing cluster baseline because the runtime/hardware changes. No HF GPU Job or Space has been created by this planning step.


HF comparison resources now exist in **HuggingEnvs**: four private Docker environment Spaces, private reproduction dataset and artifact Bucket, and **public** `data-agent-daytona-trackio` as requested. Both offline→online→read-back metric checks passed. Real Space tests passed discovery and8 whitebox Daytona tool/verifier cases. Corrected whitebox training catalog ordering to match Harbor’s sorted task indices; all1,000train/250test identities and schedules verified. Clean CPU Job **6aa9459cf76d6a098a70d6fc** passed frozen imports/CLIs and all8,774 bundle-file checks.

A100 smoke Jobs **6aa946a05527934177ee4e67** and **6aa946a1f76d6a098a70d750** loaded Qwen3.5-2B successfully, then the inherited serving launcher exited when writing its URL file to an FSx-only path. The portable copy now uses `/tmp`; redeployment/retest is in progress. Neither HF full baseline nor optimizer smoke is claimed passed. A small explicitly labeled checkpoint fixture separately passed real private-Bucket upload/download, native cursor restore and optimizer-tamper rejection. Canonical scripts are under `04-data-agent/hf`; no git pushes, no new hopper-extra/ATL allocations, and no changes to the running reference trainer.


At13:31UTC both eval Spaces report corrected bundle `86a2f815461bbd1dca7c85fb63409616daec7bf3cff3fcb208ef716d79263def`; A100 smoke reruns were submitted for both arms. See `experiments/daytona_harness_comparison/logs/hf-20260915/jobs/` for exact submission IDs. Full HF baselines and optimizer smokes remain pending.
