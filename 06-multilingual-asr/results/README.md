# Verification results

Committed summaries of runs used to verify this environment. Checkpoints and adapters stay
out of source control; these are the records needed to tell what was checked, at which
source revision, and against which data.

| File | Run |
|---|---|
| `a100-smoke-gemma-4-E2B-it.json` | Single-A100 GRPO train + eval smoke, Gemma 4 E2B |
| `a100-smoke-gemma-4-E4B-it.json` | Single-A100 GRPO train + eval smoke, Gemma 4 E4B |

Each summary records the job id and URL, the source revision, the corpus `snapshot_id`, the
`evalset_id` of the frozen split it scored, the resolved LoRA target modules, optimizer
steps, training loss, whether adapter weights changed, whether any reward group lacked
variance, and the before/after evaluation with per-sample predictions.

These verify that the pipeline executes end to end on a GPU: the corpus is read from the
bucket, audio is fetched per task, evaluation runs on a frozen split, GRPO takes optimizer
steps with non-zero advantage, and adapter weights change. They are **four** optimizer
steps over **eight** evaluation tasks. They do not establish a reward improvement, and the
macro reward is unchanged in both runs — as it should be at this size. Reproduce with the
commands in [REPRODUCE.md](../REPRODUCE.md#7-verified-single-gpu-smoke).
