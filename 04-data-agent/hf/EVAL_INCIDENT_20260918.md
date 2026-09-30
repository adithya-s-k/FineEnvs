# HF checkpoint evaluation incident — 2026-09-18

Checkpoint 600 evaluation (`6aac8a9d5c02253cfb14571a`) is making poor progress. Training is unaffected by this investigation; no live jobs or scoring rules were changed.

## Confirmed findings

- Failed OpenCode and Mini-SWE-Agent trials received HTTP 404 HTML from the Gradio relay: “No interface is running right now.” These were model-call transport failures, despite the local inference service continuing to answer requests. Claude Code also terminated with API status 404, presented as a model-access error.
- Three external relay health probes at 06:09 UTC returned HTTP 200 with the same capture instance. The relay failure is intermittent; a single startup health check cannot establish sustained reliability.
- The evaluator's circuit breaker then skipped many cells after 15 consecutive connection failures. In the downloaded Codex trace, 421 records had no reward; early examples were breaker skips, not completed model evaluations.
- At 06:03–06:07, vLLM commonly had about 30 running requests and 20–35 waiting, with generation throughput around 280–400 tokens/s. KV usage was about 13–22%. This supports inference queue pressure, not a demonstrated KV-memory exhaustion or engine crash.
- Agent 600-second timeouts and incomplete Claude trajectories also occur. The evidence does not establish which fraction of timeouts comes from inference latency versus relay disruption.

## Coverage and scoring

Latest downloaded trace snapshot: 190/1,000 unique graded cells: OpenCode 56, Claude Code 13, Codex 25, Mini-SWE-Agent 96. Retry and breaker-skip records are not additional completed evaluations. The earlier canonical progress snapshot contains only 166 graded cells. Neither is a valid final benchmark score.

Checkpoints 700 and 800 are waiting behind 600. The coordinator serializes eval jobs, so this failure delays every later checkpoint. It records failures but does not implement automatic recovery of failed jobs.

## Recovery recommendation

1. Test an authenticated HF Jobs exposed capture port to remove the Gradio relay from the sandbox path. Preserve capture-session identity and token/logprob auditing; never expose the raw inference endpoint as a capture bypass. Validate authentication propagation across all four harnesses before adopting it.
2. Start the isolated recovery qualification at concurrency 8 on the A100; measure per-request latency and completed-cell throughput before increasing it. Keep the fixed tasks, checkpoint, harness pins, sampling, scoring and 600-second task budget unchanged.
3. Preserve original traces and the first valid graded result per cell. Retry only unresolved cells, retaining provenance and auditing capture paths after restoration.
4. Make coordinator recovery attempt-aware before resubmitting jobs; the current duplicate-key guard rejects multiple jobs with one checkpoint key. A recovery must not silently create duplicate evaluations or leave 700/800 using the failing setup.

HF exposed ports require an HF token with namespace read access: https://huggingface.co/docs/hub/jobs-configuration#expose-ports.

## Isolated transport qualification

The user authorized the existing HF token for this test. It is supplied through secret environment variables, not harness configuration files.

- Direct E2B → HF Jobs: 24/24 SSE requests passed at concurrency 8. Session/nonce identity and event order matched; unauthenticated access returned 401. Events arrived incrementally, about 150 ms apart.
- E2B → sandbox-local bridge → HF Jobs: another 24/24 streams passed. The bridge adds HF authentication while forwarding the original capture credentials separately. A startup race was found and a bounded readiness check added.
- Capture middleware checks passed for Bearer, Anthropic and Google credential headers. Invalid transport credentials and unknown capture sessions remain rejected.
- CPU probe `6aacd709b1dc2b62dc590ba2` was stopped; all three test sandboxes were removed.
- GPU qualification `6aacd987b1dc2b62dc590be8`: one A100-large, checkpoint 600, fixed test indices 0–19 × four pinned harnesses, concurrency 8, three-hour limit. It stops after eight cells, resumes the same cohort, verifies that graded results are retained, and audits token/logprob captures. This is a qualification subset, not a benchmark result.

The GPU test uses immutable v11 plus a separately hashed operational adapter. Production training and the existing evaluator were not changed. Full harness qualification remains pending; the transport checks alone do not establish sustained evaluation reliability.

Evidence is retained under `experiments/async_grpo_harbor_data_agent/logs/multi4-hard500-2epochs-from500-20260917/eval-investigation/` and `smoke-evidence/6aac8a9d5c02253cfb14571a/`.

## Promotion to full evaluation

The isolated endpoint job completed all 80/80 cells, passed per-rollout TiTO and
harness-version checks, preserved the initial eight graded cells across resume, and
found no raw HF token in its artifacts. This qualifies the transport at concurrency 8;
one genuine 600-second agent timeout was still observed.

At the user's request, the old checkpoint-600 job and old coordinator were canceled.
An old-transport checkpoint-700 job launched during that transition was also canceled.
The replacement uses concurrency **16** on an A100-large, a 12-hour job limit, and the
same fixed 250-task/four-harness protocol. It restores the first valid graded results
and their captures, validates them, and evaluates only missing cells. Original artifacts
remain intact. The new coordinator schedules subsequent checkpoints with the same
transport, serializes eval jobs, and allows at most three total attempts per checkpoint.
A saved submission intent prevents blind duplicate submission after an ambiguous failure.

The transport files are byte-identical to the completed qualification. Full-cohort
recovery and increased-concurrency throughput remain under observation. Training is
unchanged. Submission receipts are under `endpoint-production-20260918-v2/` in the run
log directory.

### Recovery validation defect found at 09:08 UTC

Attempt 2 (`6aace9255c02253cfb146b78`) failed before evaluation because the recovery
validator assumed ascending task indices. The frozen manifest is deliberately shuffled.
Its indices exactly match the original evaluator; the local fix now compares against
`inputs/test_indices.txt`, preserving order. This fix is not deployed to the active job.

Attempt 3 (`6aacea60b1dc2b62dc590d97`) started from zero because its immediate predecessor
had no restored top-level trace files. At 09:09 it has 185/1,000 graded cells, no ungraded
trace records, and about 2.2 cells/min at concurrency 16. Genuine agent timeouts remain.
The original 211+ grades remain archived. Attempt 3 is a fresh cohort, not a successful
resume; its output must be reconciled against original first grades before publication
as a recovery score. Recovery lineage must search earlier attempts when an intermediate
attempt failed before restoring traces. The coordinator has a three-attempt limit.

### Concurrency 100 upgrade

The user requested 100 concurrent evaluations. The replacement uses `a100x4` with
TP1/DP4 behind one vLLM endpoint, keeping the authenticated capture transport.
This increases inference capacity as well as sandbox concurrency; measured throughput
and timeout behavior still need checking. vLLM reference:
https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/.

Recovery now traverses all previous attempts in chronological order, checks the exact
frozen shuffled cohort and evaluation protocol, and retains the first graded cell.
A local recovery test restored **370 unique cells** (107 OpenCode, 51 Claude Code,
70 Codex, 142 Mini-SWE), preserved **211 zeros**, and passed the frozen token/logprob
and harness-version audit for every restored cell. The active job may restore more
cells from the final uploaded snapshot. No original artifacts were deleted.

The previous concurrency-16 evaluator and coordinator were canceled after that test.
The new generation allows at most three attempts, preserves submission intents, and
uses concurrency 100 for subsequent checkpoints. Training remains unchanged.
