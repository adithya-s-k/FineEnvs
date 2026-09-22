# Training implementation pending

The environment, its scoring policy, and its verification exist; no GRPO runner is wired up
yet. Add one runner used by both the notebook and HF Jobs, with the offline environment
smoke and a small optimizer smoke before any longer run.

**Candidate policy.** `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it` are omni
checkpoints: both carry an `audio_config` and a `Gemma4AudioFeatureExtractor` at 16 kHz,
matching FLEURS' sampling rate, so the same family already verified on `05-multilingual-ocr`
can serve here. Two lessons from that GPU smoke transfer directly and are worth not
rediscovering:

- Resolve LoRA target modules from the real module tree rather than naming them. Gemma 4
  wraps projections in `Gemma4ClippableLinear`, which PEFT cannot adapt, and the wrapped
  ones are in the audio tower — which for *this* environment is the part that must be
  adapted, not skipped.
- A GRPO smoke needs reward variance. Identical rewards within a group make the advantage
  zero by construction, so the adapter cannot change and the run proves nothing. Use at
  least four generations and tasks the model does not already solve exactly.

Record normalization, reward, language sampling, and source revision with every run, and
report per-language scores separately: a macro average over 102 languages hides exactly the
low-resource behaviour this corpus exists to measure.
