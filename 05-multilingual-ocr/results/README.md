# Verification results

Committed summaries of runs used to verify this environment. Generated checkpoints,
adapters, and per-run archives stay out of source control; these are the records needed to
tell what was actually checked and at which source revision.

| File | Run |
|---|---|
| `a100-smoke-gemma-4-E2B-it.json` | Single-A100 GRPO smoke, Gemma 4 E2B |
| `a100-smoke-gemma-4-E4B-it.json` | Single-A100 GRPO smoke, Gemma 4 E4B |

Each summary records the served corpus manifest, the frozen `evalset_id`, the resolved
LoRA target modules, optimizer steps, training loss, whether adapter weights changed, and
the before/after evaluation with per-sample predictions.

These verify that the pipeline executes end to end on a GPU. They are four optimizer steps
over ten evaluation tasks: they do **not** establish a reward improvement, and the
before/after macro rewards are expected to be near-identical. Reproduce them with the
commands in [REPRODUCE.md](../REPRODUCE.md#5-grpo-and-hf-jobs).
