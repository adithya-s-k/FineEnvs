# Hard-task HF qualification — 2026-09-17

Target: resume multi-harness checkpoint 500 on 500 screened hard tasks for two passes. Keep the original checkpoint unchanged. Save every 50 optimizer steps and evaluate every 100 in separate jobs.

- Local tests: **26 tests passed in the combined suite (128.42 seconds)** (finite schedule, resume accounting, atomic rollout admission, multiprocessing and CPU optimizer updates).
- Refreshed runtime: `hf-jobs-v9`, SHA-256 `4f2d6777b1959d794b28e929767a1fc51ebf3b2b01d64bd455b2ffad41eacfcd`.
- Credential scan: passed before upload.
- Full parent checkpoint uploaded: 11,307,363,317 bytes, 14 files.
- [GPU smoke](https://huggingface.co/jobs/HuggingEnvs/6aac2255b1dc2b62dc58ef92): running on `h200x2` with the numerically qualified CUDA toolchain. Full qualification pending.

The job runs its own Harbor server, capture proxy and vLLM. Task sandboxes remain on E2B. No environment Space is required. Existing Spaces and other runs are unchanged.

Smoke gates: optimizer updates; full checkpoint upload and restore; finite schedule completion; capture-to-optimizer token audit; changed weights and nonzero gradient; two evaluation tasks across four harnesses. Production launch remains held for approval.

The first smoke (`6aac143d5c02253cfb143bad`) was canceled before training after the frozen CLI check exposed an entry-point mismatch. Fixed that mismatch and finite-worker completion signaling; added regression coverage for both. Archived bundles v2/v3 remain preserved.

## Kernel dependency correction

The second smoke passed model restore, job-local Harbor startup, capture identity, TiTO endpoint probing and all 500/250 catalog checks. It produced rollout captures but no optimizer update before cancellation. The trainer logged a slow Torch fallback; the HF lockfile lacked the FLA package installed locally.

Added hash-pinned `fla-core==0.5.2` and `flash-linear-attention==0.5.2`, matching local versions, and a startup check that refuses training without the FLA Gated DeltaNet kernel. Replacement bundle v5 is being prepared. Each new attempt now gets a bundle-specific artifact prefix so retries cannot mix their logs.

Job `6aac15cd5c02253cfb143bf3` was canceled. Cleanup matched 16 owned trial names; no matching live E2B sandboxes remained. Production remains unqualified and unlaunched.

Logging checks now require fresh remote Trackio readback before qualification. The logger initializes its offline path before importing Trackio, and smoke evaluation is labeled with the actual checkpoint step. Full HF console logs and JSON operation history are preserved in the experiment directory.

Latest attempt: `6aac1a80b1dc2b62dc58ee7e`, bundle v7. Kernel dependency pins and all logging fixes are included. The final-marker publication regression passed. GPU qualification is pending; no production job has been submitted.

## Final partial update

A new real CPU optimizer-loop test exposed a finite-stream edge case: five rollout bundles with GAS 4 produced one optimizer update instead of two. Native Transformers uses the configured accumulation boundary for unsized streams even when prefetch returns a shorter final batch.

The fix fills only the remaining accumulation slots with empty work. These slots contain no samples or tokens and execute no model forward/backward. All real gradients retain the supervised-token normalizer and reach one final optimizer step. Credit return and admission receipts are checked with backpressure enabled and disabled. GPU qualification must use the updated v8 bundle.

The v7 live trainer confirmed `fla.ops.gated_delta_rule.chunk` is active. The separate missing causal-convolution warning remains; the critical Gated DeltaNet path is accelerated. No long run has been launched.

Current smoke: `6aac1cf1b1dc2b62dc58eeca` (v8). The v7 attempt was canceled after confirming FLA selection; no matching live sandboxes remained. Final partial-update tests now pass. The complete local suite is being rerun, with output saved to `hf-jobs-v8/pytest.log`.

Combined validation completed: **26 passed**. The full output is saved in `hf-jobs-v8/pytest.log`; only expected experimental/deprecation and CPU-host CUDA warnings were emitted.

## H200 kernel diagnostic

The v8 smoke exited before its first optimizer update. Its last trainer log is a TileLang compilation. A subsequent cleanup error (`e2b` absent from the trainer venv) masked the primary failure in the console. Cleanup now uses the environment interpreter; trainer return codes, Python fault traces and explicit failure status are recorded. Artifact publication runs even if cleanup fails.

A separate diagnostic job `6aac201c5c02253cfb143e00` compares FLA outputs and gradients against Torch at lengths 63, 256 and 2048, with Triton and TileLang separately. Backend selection remains pending this evidence. See [upstream Hopper precision issue](https://github.com/fla-org/flash-linear-attention/issues/640); avoiding a compiler crash alone is not sufficient numerical qualification.

Two additional coordinator tests passed (0.48 seconds): periodic/final checkpoint selection, restart deduplication and cancellation. Total local checks passed: 28. Production remains blocked on live GPU qualification, not launched.

## CUDA compiler/header mismatch

Kernel diagnostic `6aac201c5c02253cfb143e00` isolated both outcomes:

- Triton is explicitly rejected by FLA on Hopper with Triton >=3.4 and <3.7.1 because of incorrect backward results. No override is used.
- TileLang compilation failed because CUDA 13.2 nvcc was paired with CUDA 13.0 runtime headers.

Aligned the lockfile with CUDA toolkit 13.0.2 metadata: nvcc/CRT/NVVM 13.0.88 and CCCL 13.0.85; runtime remains 13.0.96. Exact wheel hashes are recorded. Diagnostic `6aac2120b1dc2b62dc58ef67` retests TileLang numerical forward/backward behavior using this lockfile. No more full rollout smokes will launch until this narrower check passes.

The aligned CUDA diagnostic **completed successfully** on H200. TileLang output/gradient relative RMS errors were 0.26–0.42% against the BF16 Torch reference at lengths 63, 256 and 2048. The replacement runtime runs this GPU check before model restore or sandbox rollouts, records `kernel_preflight.json`, and requires it in production admission. TileLang remains enabled; the unsafe Triton combination is not used.

Latest smoke: `6aac2255b1dc2b62dc58ef92`, bundle v9. Coordinator/publication checks rechecked: 3 passed in 0.67 seconds. Total distinct local tests: 28, plus the completed H200 numerical diagnostic.
