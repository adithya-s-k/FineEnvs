# Native OpenCode baseline — 2026-09-15

**21/250 = 8.4% pass@1.** This is the standalone `envs/blackbox-opencode` implementation on Daytona, not the Harbor OpenCode adapter.

| Difficulty | Correct / tasks | pass@1 |
| --- | ---: | ---: |
| Easy | 9/33 | 27.27% |
| Medium | 10/118 | 8.47% |
| Hard | 2/99 | 2.02% |
| Total | 21/250 | 8.40% |

All 250 first attempts were graded. All 250 exact-token TiTO audits passed. There were no ungraded infrastructure attempts and no retries. The evaluator preserves each first graded result, including zero. All evaluation-owned Daytona sandboxes were released; the final cleanup query found zero remaining.

- Model: Qwen/Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Harness: native OpenCode 1.18.31.
- Inference: two H100s on hopper-prod, TP1/DP2, vLLM 0.25.1, compiled execution, context 131,072, processed logprobs and engine token IDs, prefix caching disabled.
- Dispatch: fixed concurrency **50**, no ramp; 250 frozen test tasks; dispatch seed 42.
- Limits: 17 model calls, 600-second agent budget, 4,096 output tokens per call; temperature 0.8, top-p 1.0, top-k disabled; thinking disabled.
- Evaluation stage: **862.70 seconds (14.38 minutes), 17.39 graded tasks/minute**. This excludes inference startup, final uploads and cleanup.
- Slurm job: **80514**, completed successfully.
- Local source identity: `9039c3b13b7cbde6bb5c7f804f55800487fedd0f2a100fb820046037bb26cf5a`.
- Portable base bundle: `c18bcaedf792a657237e9797a66854daa683038da821217ead36966f83ad399c`.
- Test manifest SHA256: `38943d89f5bb0fec79db8c7a2680c4a8353cf5c53c6eabffaaef8971cb2eb964`.

Canonical evidence is in `experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-baseline-v2/repro/outputs/local-eval-opencode-80514/`: `canonical_scores.json`, `daytona/results.jsonl`, `daytona/captures/`, `daytona/scalability.json`, `status.json`, and `cleanup.json`. Captures and scores are also persisted under `hf://buckets/HuggingEnvs/data-agent-daytona-artifacts/20260915-hf/jobs/local-eval-opencode-80514`.

The canceled HF partial cohorts remain archived and were not imported. The native SETA Whitebox baseline is separately complete at 47/250 (18.8% pass@1). Harness prompts, tool interfaces and answer submission differ, so these are separate native baselines. The original multi-harness baseline/checkpoint series uses its own four-harness evaluation protocol.

Sampling review: all 2,849 captured tool-enabled agent calls have empty sampling-override records. The inference process explicitly supplied temperature 0.8, top-p 1.0 and top-k -1 as its defaults. The score therefore retains that serving configuration. These baseline captures do not contain the explicit per-session training-policy evidence required by TRL admission. The later native training fix forwards the policy through factory, client, MCP tool and capture registry; it does not backfill or alter baseline captures. See `daytona/sampling_review.json` in the evidence directory.
