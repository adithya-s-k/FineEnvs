# Validation of the tutorial rewrite

The tutorial now uses public TRL trainers and OpenEnv producers. It does not download the archived experiment runtime or patch installed libraries.

## Checked locally

- 29 tests passed with the installed runtime: 10 data/launch checks and 19 contract/tokenizer checks.
- All 1,000 training and 250 test tasks were re-prepared from the local copies, checking instruction and grader hashes.
- A fresh dependency resolution succeeded with upstream TRL, OpenEnv and Transformers main. This was resolution, not a clean GPU installation.
- HF train/eval submission plans, Python compilation, shell syntax, Ruff, documentation links and the generated repository index passed.


The CPU checks exercise the fixed task split, reward gating, native tool counts, task identity independent of worker seeds, sandbox cleanup, tool schemas, and both non-thinking tokenizer templates. Contract checks pass partial and zero masks through the real OpenEnv producer and TRL consumer.

```bash
python -m pytest tests -q             # Lightweight checks, no model/runtime installation
python -m pytest tests -q --contract  # Also exercise installed TRL/OpenEnv and tokenizers
python check_setup.py                # API prerequisites
```

Contract validation used merged OpenEnv `86a180ede21e044f7929b9a7783ad83aa67d83a3` and local TRL PR #6947 at `910138e9392dda75d3a9766e7c494d44f1d82723`. That TRL branch includes the public main trainer plus the pending typed consumer. This is not a claim that the consumer is already on main. The setup script fails clearly until main contains it.

## Still required before a long run

- Resolve and install current upstream dependencies in a clean GPU job after TRL #6947 merges.
- For each model/mode, complete the documented two-update smoke and reload checkpoint 2 for evaluation.
- Verify LFM packed logits agree with separate sequences on the GPU. The async scripts perform this check and abort on a mismatch; a kernel that ignores sequence boundaries is not acceptable.
- Check nonzero learning signal over a larger sample, and complete the fresh baseline/checkpoint-100 comparison.
- Validate simultaneous training and evaluation against the available sandbox quota. Evaluation uses separate GPUs, but sandbox and storage capacity are shared.

Native `opencode_env` is available on current OpenEnv main but is marked deprecated upstream. This tutorial retains it because the comparison explicitly calls for native OpenCode. Harbor-only OpenCode would be a different experiment.

The public native API bounds time and tokens per call; it does not expose Harbor's strict turn limit. Upstream async packing can drop rows longer than 40,960 tokens and can give one rollout multiple training rows. The tutorial does not claim the archive's custom scheduling, exact resume or rollout weighting guarantees.

## Earlier evidence

[RESULTS.md](RESULTS.md) retains the historical pilot scores and article curves. The [validation record before this rewrite](https://github.com/adithya-s-k/FineEnvs/blob/4d9c040a28695c484de75bc23d6a70a99beda258/05-multi-harness-rl/VALIDATION.md) contains the old GPU smokes, source pins and artifact paths. Those jobs used custom runtime changes, including A100 support, and do not qualify the rewritten scripts.

The previous files have been preserved locally under ignored `temp/pre-tutorial-rewrite/`; they also remain in git history. Prepared data, old experiments and credentials are not included in the PR.
