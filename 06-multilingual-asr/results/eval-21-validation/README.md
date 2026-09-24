# Candidate comparison: which model to fine-tune

Four checkpoints scored on `eval_21_validation` — the frozen **1,050-task** set over the 21
languages shared with `05-multilingual-ocr`, all three task families, **50 tasks per language**.
The **validation** variant is used deliberately: `eval_21_test` stays untouched so a
post-training number still means something.

Every model saw the identical prompt from the environment and was graded by it, greedily
sampled through vLLM. Per-model files hold every prediction; `leaderboard.json` the summary.

| Model | Macro | `transcription` | `verbatim` | `language_id` |
|---|---|---|---|---|
| **google/gemma-4-E4B-it** | **0.6614** | 0.6768 | 0.6395 | 0.6714 |
| google/gemma-4-E2B-it | 0.5413 | 0.6218 | 0.5926 | 0.4114 |
| Qwen/Qwen2.5-Omni-7B | 0.4703 | 0.5024 | 0.4869 | 0.4229 |
| Qwen/Qwen2.5-Omni-3B | 0.4365 | 0.4678 | 0.4553 | 0.3886 |

## The macro average hides the finding

Split the 21 languages into the Indic group and everything else and the comparison falls
apart into two different pictures:

| | gemma-4-E4B-it | gemma-4-E2B-it | Qwen2.5-Omni-7B | Qwen2.5-Omni-3B |
|---|---|---|---|---|
| **Non-Indic** (11 languages) | 0.7383 | 0.7149 | 0.7319 | 0.7175 |
| **Indic** (10 languages) | 0.5793 | 0.3516 | 0.1834 | 0.1289 |

On the non-Indic half the four sit within **0.023** of one another — that is a tie. On the
Indic half they spread over **4.5×**, and that gap alone produces the ranking above. Qwen2.5-Omni
is at or near zero on several Indic languages; its predictions are well formed — clean
transcripts and correctly shaped locale codes — so this is coverage, not a parsing failure.

| | E4B | E2B | Qwen-7B | Qwen-3B |
|---|---|---|---|---|
| `kn_in` | 0.721 | 0.336 | 0.060 | 0.027 |
| `ta_in` | 0.612 | 0.243 | 0.080 | 0.088 |
| `te_in` | 0.659 | 0.309 | 0.229 | 0.033 |
| `ml_in` | 0.446 | 0.279 | 0.033 | 0.003 |
| `or_in` | 0.256 | 0.072 | 0.043 | 0.012 |

## What this says about fine-tuning

**Gemma 4 E4B is the base to train.** It is strongest overall, and it is the only candidate
with non-trivial Indic ability to build on — a model scoring 0.03 on Kannada has nothing for
GRPO to sharpen. It also has the most room left: its Indic scores sit far below what it
reaches on European languages, and that headroom is on exactly the languages this corpus
exists to measure.

E4B over E2B is worth a second look if the target is transcription alone: much of E4B's lead
is `language_id`, which is exact match on the FLEURS locale rather than a transcription skill.
E2B remains much weaker on Indic transcription, so the conclusion holds.

## Reading these numbers honestly

An earlier run of this same comparison on a 504-task set put every model within **0.014** of
where it lands here, so the ranking is not a sampling artefact. Each language still carries
only 50 tasks, so treat a single language as an indication and the macro as the comparable
figure.

`language_id` is scored by exact match on the FLEURS locale, so answering `ar_jo` for `ar_eg`
scores zero despite identifying Arabic. That is the documented policy — the corpus
distinguishes those variants — and it is much of why every model scores lower on this family.

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

Add `--max-model-len 8192` for Qwen2.5-Omni. One A100 scores a model in about two minutes:
1,050 clips are fetched once for all candidates (106s, 783 MB), generation runs at 9.6–10.4
task/s for Gemma and 4.3–5.0 for Qwen, and grading takes seconds. The rest of the wall clock
is model download and vLLM startup.
