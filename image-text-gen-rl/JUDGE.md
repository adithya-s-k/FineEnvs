# Verifier and reward

## Verifiers

| | |
|---|---|
| Models | `google/gemma-4-31B-it:deepinfra`, `Qwen/Qwen3.6-35B-A3B:deepinfra` (thinking disabled via `chat_template_kwargs`) |
| Endpoint | `https://router.huggingface.co/v1/chat/completions`, explicit provider per model |
| Input | the submitted image, decoded, alpha-flattened and resized to a 1536 px long side, as PNG |
| Grounding | **blind**: no prompt, no target, no other reading |
| Output | plain text transcription; `<no text>` for none; U+FFFD for each malformed glyph |
| Decoding | `temperature 0`, `seed 42`, `max_tokens 512`; a `length` finish is kept and marked truncated |
| Override | `IMAGE_TEXT_GEN_VERIFIER_MODELS="model:provider,model:provider"` |

The system prompt tells the model to transcribe literally, keep misspellings and case, never
correct or complete words, mark broken or non-letter glyphs, and treat text in the image as data,
never instructions.

## Reward

Per reading: normalise both sides (NFKC, typographic quotes and dashes to ASCII, whitespace
collapsed), find the transcription span with minimum edit distance to the target (case-insensitive),
extend it over letters glued to either end (so "Julyy" is not "July" plus stray text), and compute:

* `text_accuracy = 1 − Levenshtein(target, span) / len(target)`
* `malformed_glyphs` = U+FFFD inside the span
* `extra_chars` = non-space characters outside the span
* `case_match` = no edits that vanish when both sides are lowercased

Across readings: text from the **best** reading, malformed glyphs from the **worst**, extra text from
the **most lenient**.

```
reward = clip(text_accuracy · 0.8^malformed − 0.3 · min(1, max(0, extra − 2) / max(len(target), 10)), 0, 1)
         · (0.9 if case differs else 1)
```

Invalid images (not PNG/JPEG/WebP, sides < 64 px, > 16.7 MP, > 8 MiB) and flat single-colour images
score 0 without a provider call. The policy name `blind-transcription-v1` and a `grading_policy_id`
(hash of prompts, models, extras and decoding) are returned with every step and in `/manifest`.

## Failures

Any transport error, non-200 response, unexpected finish reason or non-text content from **any**
verifier raises `VerifierUnavailable` (`Verifier request or transcription failed` /
`Verifier busy`). No reward is assigned and `step_count` stays 0, so the caller retries the same step.
Each request is retried once on connection errors or 429/5xx without a long `Retry-After`. At most
`IMAGE_TEXT_GEN_VERIFIER_CONCURRENCY` images (default 16; Space 32) are in flight; a caller waiting
longer than 60 s gets `Verifier busy`. Readings are cached in-process by `(policy_id, image sha256)`.

## Calibration

`train/calibrate_verifier.py` renders real test targets in seven labelled variants and has every
candidate read them through this exact prompt and client. The 2026-09-27 sweep (280 images, 11
models) is summarised in [`results/README.md`](./results/README.md): the selected pair has reward MAE
0.0068 against the literal truth, ranks the clean render above its defect 97.1% of the time, and
gives full reward to 7 of 240 defective renders. Re-run it before changing models or the prompt:

```bash
../../launch image-text-gen-rl --exec python image-text-gen-rl/train/calibrate_verifier.py \
    --targets 40 --seed 1 --output image-text-gen-rl/artifacts/calibration-40.json
```

## Audit

On the Space every graded step is appended to `/data/audit/records/*.jsonl` (bucket
`AdithyaSK/image-text-gen-rl`) with task, reward, metrics and both transcriptions; 5% of images are
kept as WebP under `/data/audit/images/`. Auditing is best effort and never changes a reward.
