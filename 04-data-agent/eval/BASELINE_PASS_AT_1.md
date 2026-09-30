# Qwen3.5-2B · four-harness pass@1 baseline

**Completed 2026-09-14: 1,000/1,000 evaluations, overall pass@1 14.6%.**

Each of the same 250 fixed test tasks was evaluated once with each harness. These are **pass@1** results, not pass@4 or best-of-four results. The overall score averages four equally sized harness evaluations; it does not mean that 14.6% of tasks were solved by at least one harness.

| Harness | Correct / tasks | Pass@1 | Observed harness version |
| --- | ---: | ---: | --- |
| OpenCode | 27 / 250 | 10.8% | 1.18.31 |
| Claude Code | 42 / 250 | 16.8% | 2.1.270 |
| Codex | 41 / 250 | 16.4% | 0.154.0 |
| mini-swe-agent | 36 / 250 | 14.4% | 2.4.6 |
| **Overall** | **146 / 1,000** | **14.6%** | |

## Difficulty breakdown

| Difficulty | Unique test tasks | Task share | Correct / evaluations | Pass@1 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 33 | 13.2% | 53 / 132 | 40.2% |
| Medium | 118 | 47.2% | 68 / 472 | 14.4% |
| Hard | 99 | 39.6% | 25 / 396 | 6.3% |

| Harness | Easy (33 tasks) | Medium (118 tasks) | Hard (99 tasks) |
| --- | ---: | ---: | ---: |
| OpenCode | 11/33 · 33.3% | 10/118 · 8.5% | 6/99 · 6.1% |
| Claude Code | 14/33 · 42.4% | 22/118 · 18.6% | 6/99 · 6.1% |
| Codex | 14/33 · 42.4% | 20/118 · 16.9% | 7/99 · 7.1% |
| mini-swe-agent | 14/33 · 42.4% | 16/118 · 13.6% | 6/99 · 6.1% |

## Fixed protocol

- Model: original **Qwen/Qwen3.5-2B**, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. No trained checkpoint was used.
- Test dataset: **HuggingEnvs/data-agent-harbor-test**, revision `291c8e50bfa7e34135090071ecaa0686bd99d06f`; 250 fixed tasks with recorded file hashes and a fixed shuffled dispatch order.
- Sampling: temperature 0.8, top_p 1.0, top_k disabled, thinking disabled, one graded rollout per task/harness cell.
- Inference: two H100 80GB GPUs, TP=1 / DP=2, one vLLM endpoint; vLLM 0.25.1, BF16, eager, Qwen3 XML tool parser, GDN Triton prefill, prefix caching disabled.
- Budgets: 131,072 context tokens, 4,096 output tokens per call, 17 model calls and 600 seconds per agent.
- Admission: client, server and sandbox concurrency 100; OpenEnv capacity 128.
- Sandbox: shared E2B template, 1 CPU / 4 GiB.
- Grading: native task verifier; explicit `correctness,reward` key preference reads one verifier result. An ungraded infrastructure failure is not scored as zero.
- Retry/selection: retain the first valid graded measurement for each cell. Retry ungraded failures; never replace a graded failure with a later success. A queued duplicate graded attempt was excluded from the canonical matrix.

All 250 observed trials per harness used the versions listed above. They were dynamically installed at baseline time; checkpoint evaluations must pin these versions or explicitly report a protocol change.

## TiTO and scope of validation

**1,000/1,000 selected captures passed** the exact-context, token-ID, behavior-logprob and loss-mask audit through the real TRL trace reader and sequence builder, with `fork_threshold_tokens=0`. All **2,721,986** eligible supervised tokens were retained, including multiplicity. Auxiliary calls and synthetic budget-stop messages do not become targets.

This validates capture and sequence assembly. It does not establish that a subsequent optimizer consumes every row or that weighting is independent of microbatch packing. Claude Code averaged 12.9 sequences per rollout; repeated masked context adds compute, not an automatic 12.9× gradient multiplier. Trainer normalization and rollout admission are tracked in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206).

## Runtime and reliability

GPU job **78215** completed at **2026-09-14 19:59:21 UTC**, elapsed **1:05:31**. It reused 88 validated measurements and completed the remaining 912 cells; this is not a measurement of 1,000 fresh rollouts in 65 minutes. Budget approximately 60–75 minutes provisionally for a fresh evaluation at this configuration.

Cleanup job **78216** completed with **zero owned sandboxes remaining**. The successful GPU job needed retries for ten relay/API failures. One legitimate context-budget stop had its diagnostic/status corrected in a derived capture while preserving the original verifier score and tokens; original artifacts remain intact.

No OpenEnv capacity rejections or KV-cache preemptions were recorded. A separate relay test still exposed a roughly 60-second timeout before initial response bytes; a synthetic SSE heartbeat test passed, but production heartbeat handling is not yet implemented. Concurrency 150 has not been benchmarked.

## Evidence and checkpoint comparisons

Machine-readable protocol and summary: [baseline_protocol.json](baseline_protocol.json).

Durable workspace evidence is under:

`experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914/`

- `manifest.json`, `indices.txt`: immutable task selection, source revisions and hashes.
- `job-78215/canonical_results.json`: the 1,000 selected measurements and excluded duplicate.
- `job-78215/final_tito.json`: complete capture audits.
- `job-78215/BASELINE_REPORT.md`: detailed results, provenance and limitations.
- `job-78215/measurement_report.json`, `inference_measurement_report.json`: measured throughput and serving behavior.
- `source-snapshot-v4/`: launch sources and local repository patches.

For every checkpoint, use the same 250 tasks × four harnesses × pass@1 and the same evaluation budgets and sampling. Serve each completed checkpoint on a separate two-GPU TP=1 / DP=2 job. Record checkpoint identity and report complete coverage, per-harness scores, difficulty scores, exclusions and deltas from this baseline.

The training pool is separate and screened for test source IDs, shared notebooks and instruction overlap. Start the long run from the original base model: earlier diagnostic checkpoint 78170 used test tasks and is unsuitable as a clean initialization for this comparison.
