# Verification results

Committed summaries of runs used to verify this environment. Checkpoints and adapters stay
out of source control; these are the records needed to tell what was checked, at which
source revision, and with which snapshot.

| File | Run |
|---|---|
| `a100-smoke-gemma-4-E2B-it.json` | Single-A100 GRPO train + eval smoke, Gemma 4 E2B |
| `a100-smoke-gemma-4-E4B-it.json` | Single-A100 GRPO train + eval smoke, Gemma 4 E4B |

Each summary records the snapshot id, the resolved LoRA target modules, optimizer steps,
training loss, whether adapter weights changed, and the before/after evaluation with
per-sample predictions.

These verify that the pipeline executes end to end on a GPU. They are **two** optimizer
steps over **one evaluation utterance per language**: they do not establish a reward
improvement, and any before/after difference at this size is noise, not a result.
Reproduce with the commands in [REPRODUCE.md](../REPRODUCE.md#6-verified-single-gpu-smoke).
