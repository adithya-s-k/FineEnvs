# Candidate comparison: which model to fine-tune

Four checkpoints scored on `eval_21_validation` — the frozen 504-task set over the 21
languages shared with `05-multilingual-ocr`, all three task families, 24 tasks per language.
The **validation** variant is used deliberately: `eval_21_test` stays untouched so a
post-training number still means something.

Every model saw the identical prompt from the environment and was graded by it, greedily
sampled through vLLM. Per-model files hold every prediction; `leaderboard.json` holds the
summary.

| Model | Macro | `transcription` | `verbatim` | `language_id` |
|---|---|---|---|---|
| **google/gemma-4-E4B-it** | **0.6636** | 0.6772 | 0.6529 | 0.6607 |
| google/gemma-4-E2B-it | 0.5513 | 0.6279 | 0.6093 | 0.4167 |
| Qwen/Qwen2.5-Omni-7B | 0.4800 | 0.5209 | 0.5082 | 0.4107 |
| Qwen/Qwen2.5-Omni-3B | 0.4499 | 0.4783 | 0.4725 | 0.3988 |

## The macro average hides the finding

On high-resource languages the four are close, and Qwen often wins: German 0.921 for
Qwen-7B against 0.812 for E4B, Italian 0.924 against 0.836, Mandarin 0.520 against 0.386,
Egyptian Arabic 0.683 against 0.543.

The ranking is decided by the Indic languages, where the gap is not close:

| | E4B | E2B | Qwen-7B | Qwen-3B |
|---|---|---|---|---|
| `kn_in` | 0.710 | 0.326 | 0.069 | 0.030 |
| `ta_in` | 0.622 | 0.246 | 0.099 | 0.090 |
| `te_in` | 0.672 | 0.316 | 0.192 | 0.037 |
| `ml_in` | 0.360 | 0.290 | 0.027 | 0.004 |
| `or_in` | 0.246 | 0.105 | 0.044 | 0.013 |

Qwen2.5-Omni is at or near zero on several of these. Its predictions are well formed —
clean transcripts and correctly shaped locale codes — so this is coverage, not a prompt or
parsing failure.

## What this says about fine-tuning

**Gemma 4 E4B is the base to train**, for two reasons that point the same way. It is
strongest overall, and it is the only candidate with non-trivial Indic ability to build on;
a model scoring 0.03 on Kannada has nothing for GRPO to sharpen. It also has the most room
left: its Indic scores sit between 0.25 and 0.71, far from the 0.80–0.90 it reaches on
European languages, and that headroom is on exactly the languages this corpus exists to
measure.

E4B over E2B is worth a second look if the target is transcription alone. The two differ by
only ~0.05 on both transcription families; nearly all of E4B's lead is `language_id`
(0.66 against 0.42), which is exact match on the FLEURS locale rather than a transcription
skill. E2B remains much weaker on Indic transcription, so the conclusion holds — but the
headline gap overstates it.

## Reading these numbers honestly

Each language carries 24 tasks, 8 per family, so a per-language figure moves easily:
Qwen-3B beating Qwen-7B on Russian (0.875 against 0.538) and English (0.927 against 0.833)
is noise at this size, not a finding. The macro figures over 504 tasks are the comparable
ones; treat a single language as an indication.

`language_id` is scored by exact match on the FLEURS locale, so answering `ar_jo` for
`ar_eg` scores zero despite identifying Arabic. That is the documented policy — the corpus
distinguishes those variants — and it is a large part of why every model scores lower on
this family than on transcription.

## Reproduce

```bash
REV=$(git rev-parse HEAD)          # must be pushed: the job fetches this exact commit
uv run --frozen --project envs/multilingual_asr hf jobs uv run \
  --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  --volume hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  train/hf_job.py --revision "$REV" --mode eval-vllm --source-root /fleurs \
  --models google/gemma-4-E2B-it google/gemma-4-E4B-it \
  --eval-split eval_21_validation --workers 32
```

Add `--max-model-len 8192` for Qwen2.5-Omni, whose audio encoder needs more room than
Gemma's. One A100 scores a model in about a minute — 504 clips are fetched once for all
candidates (66s, 384 MB), generation runs at 9.5 task/s for Gemma and 4.2–5.1 for Qwen, and
grading takes 2–3s. The rest of the wall clock is model download and vLLM startup.
