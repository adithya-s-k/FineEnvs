# Data contract

| | |
|---|---|
| Source | [`leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset`](https://huggingface.co/datasets/leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset) |
| Revision | `0ec4bce48b13570f95609a32fe72c38c40b39c33` (pinned in `envs/image_text_gen/data/catalog.py`) |
| License | MIT |
| Files read | `data_with_ocr_reward_train.csv`, `data_with_ocr_reward_test.csv` |
| Fields used | `id`, `prompt`, `text` (the target), `v1..v5_qwen_ocr_levenstein_score` (difficulty prior) |

Every prompt quotes exactly one target string, verbatim. The `480p_v*.zip` files are SD3
latents (`.pt`), not images, and are not used.

## Published derivative

The environment serves [`AdithyaSK/image-text-gen-rl-prompts`](https://huggingface.co/datasets/AdithyaSK/image-text-gen-rl-prompts)
at the commit pinned as `PUBLISHED_REVISION` in `catalog.py` (built by `train/publish_dataset.py`;
its card credits the original). The Space mounts it read-only at `/dataset`.

## Screening

`train/screen_prompts.py` removed 300 of 14,047 source rows (2.1%) for sexual (79), profanity (68),
graphic violence (92), hate (21), self-harm (17), drugs (13), sexual violence (7) and word-list-only
(3) content: a strict whole-word list over prompt and target, unioned with a Gemma 4 31B rubric
classifier over batches of 20 prompts. Llama-Guard-4 was rejected: it rated a lingerie prompt
rendering "FUCKBOOK" as safe. The screen errs toward removal. `envs/image_text_gen/data/exclusions.json`
lists removed row IDs and categories only; the flagged text is not republished.

## Splits

| Split | Tasks | Derivation |
|---|---:|---|
| `train` | 11,982 | screened source train, de-duplicated by prompt, minus prompts that also occur in test, hash bucket ≥ 4 |
| `validation` | 466 | the same rows with `sha256(schema, revision, id) % 100 < 4` |
| `test` | 986 | screened source test |

Task IDs are `itg-` + a content hash of the schema version, revision, source id, prompt and
target, so a replayed ID always means the same task. `baseline_ocr` is the mean Qwen OCR
Levenshtein score of the dataset's five SD3 samples (0.875 on average; 20% of prompts score 1.0
on all five), exposed for curricula. 306 targets contain a literal `?`, which is why the
malformed-glyph marker is U+FFFD rather than `?`.

Locally the published splits are downloaded at the pinned revision on first use;
`IMAGE_TEXT_GEN_SOURCE_DIR` derives tasks from leffff-format CSVs instead (tests and smoke).
