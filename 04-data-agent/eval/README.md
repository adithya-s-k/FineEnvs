# Evaluation

Start with [../reproduce.md](../reproduce.md). Accepted checkpoint scores and full harness/difficulty tables are in [../results.md](../results.md).

`eval_concurrent.py` is the asynchronous Harbor client for an already-hosted OpenEnv service and model URL; `hf/runtime/job.py` stages and serves the pinned model before invoking it. `checkpoint_evals.py` retains the historical local controller. `hf/runtime/coordinator.py` handles independent HF checkpoint evaluation Jobs.

Every comparison is pass@1: keep the first graded attempt, including zero; retry only ungraded infrastructure failures. Publish complete cohorts after task, harness-version, checkpoint and TiTO checks. Hub concurrency defaults to 35, local to 50.

The standalone native OpenCode evaluator is `hf/runtime/eval_opencode.py`; the native SETA evaluator is `tools/eval_whitebox_native.py`. They have different agent protocols and their standalone baselines must be labelled separately from Harbor's four-harness suite.
