# Training

One GRPO runner, `grpo_asr.py`, used by a local run and by HF Jobs alike. It talks to the
environment over HTTP: either a server it starts itself from a corpus manifest (`--corpus`),
or one already running, including the deployed Space (`--env-url`).

```bash
uv run --frozen --project envs/multilingual_asr --extra train python train/grpo_asr.py \
  --corpus data/corpus-manifest.json \
  --languages en_us hi_in --families transcription \
  --eval-split eval_21_test --eval-limit 8 \
  --num-generations 8 --max-steps 4 --smoke
```

**Sampling never lists a split.** The train split holds 815,226 tasks, so drawing a few per
language by paging it is not an option. `/group_count` and `/group_tasks` answer from the
index by language and task family, and the runner draws positions and fetches only those:
two languages at four tasks each take 2.9 s. The draw is seeded, so a run reproduces.

**Only immutable identifiers reach the sampler** — task id, language, family. A `prompt`
column would be read by TRL as a conversation, and the environment already owns the prompt.

**Evaluation uses a frozen split, or no two runs are comparable.** `--eval-split
eval_21_test` scores the 21 languages shared with `05-multilingual-ocr`, so an ASR result
can be read against an OCR result language for language; `eval_102_test` covers all 102.
The `_validation` variants exist so model selection never touches the set a final number is
reported on. A frozen split is used whole unless `--eval-limit` is given, and the run
records the set's `evalset_id` and the limit, so a slice is never mistaken for a full score.

**LoRA targets are resolved from the real module tree**, not named. Gemma 4 wraps
projections in `Gemma4ClippableLinear`, which PEFT cannot adapt, and a short target name
matches both a wrapper and a leaf, so injection fails outright. Full module names name one
leaf each. Unlike the OCR environment nothing is skipped here: the audio tower is the part
this task must adapt, which is why the target counts are 106 and 122 rather than 82 and 98.

**A GRPO run needs reward variance.** Identical rewards within a group make the advantage
zero by construction, the adapter cannot change, and the run proves nothing. The runner
checks this and fails rather than reporting a pass: four generations on an easy English
clip is not enough, eight is. Report per-language scores separately — a macro average over
102 languages hides exactly the low-resource behaviour this corpus exists to measure.
