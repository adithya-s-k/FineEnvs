# Gemma 4 on the frozen evaluation set

Both Gemma 4 sizes scored on all 500 tasks of `eval-500.json` — 22 languages, of which
**11 are Indic and carry exactly half the set (250 tasks)** — across all five families.
Every model saw the identical prompt from the environment and was graded by it, sampled
greedily through vLLM so a re-run reproduces the score. No task failed.

| | E2B | E4B |
|---|---|---|
| **Macro reward** | 0.3497 | **0.4051** |
| Micro reward | 0.3538 | 0.4107 |
| Indic (11 langs, 250 tasks) | 0.2860 | **0.3494** |
| Non-Indic (11 langs, 250 tasks) | 0.4216 | 0.4720 |

| Family | E2B | E4B |
|---|---|---|
| `section_ocr` | 0.6302 | 0.6654 |
| `mcq_vqa` | 0.5800 | 0.6600 |
| `page_ocr` | 0.4586 | 0.5582 |
| `descriptive_vqa` | 0.1000 | 0.1700 |
| `layout_detection` | **0.0000** | **0.0000** |

E4B wins on every family and on both script groups, and both models are markedly weaker on
Indic scripts than on the rest — the same shape the ASR comparison found, though here the
gap is narrower because even the non-Indic numbers leave plenty of room.

## `layout_detection` scores zero for a reason worth acting on

Not a grading fault and not an inability to see layout. The models emit **well-formed
layout JSON in the wrong schema**, unanimously:

- **200 of 200 predictions used `box_2d`**, never the `bbox` the prompt asks for.
- **All 9,692 coordinate values fall in 0–1000**, Gemma's normalized convention, where the
  prompt explicitly asks for absolute pixel coordinates on a canvas whose width and height
  it states.

The prompt is unambiguous — it gives the exact object shape, names the allowed labels and
says "no extra keys or prose" — and the grader is right to reject what came back. Gemma 4
is falling back to the layout format it was trained to emit and ignoring the requested one.
A further 17 of 100 E4B answers were cut off mid-array by the token budget, so raising
`--max-new-tokens` is worth doing too, but it is not the cause: the 83 complete answers
were rejected on schema alone.

**This matters more for training than for the score.** A family where every rollout scores
exactly 0.0 has no reward variance, so GRPO gets no gradient from it — the same failure
that stopped the first ASR smoke. As it stands `layout_detection` cannot be trained by
this reward. Before including it in a run, either give the reward partial credit for a
recoverable schema, or get format compliance off the floor first so there is something for
the advantage to separate.

## Cost, which is the point

Built to run after every checkpoint, so the costs that would otherwise repeat are paid once
per session:

| | Time | Paid |
|---|---|---|
| Fetch 500 pages (395 MB) | 1357s | once per session — reused by every checkpoint |
| vLLM boot + weights | ~4–8 min | once per session, with adapters swapped over the live engine |
| **Score one checkpoint** | **50–70s** | **per checkpoint** |

Generation ran at 11.6 task/s with grading overlapped behind it rather than after it, which
is what keeps a judge-graded family from adding its latency to the total: 500 tasks,
including the 100 that leave the machine for a judge, complete in under 70 seconds.

The page fetch is now the dominant cost and is the one thing still worth attacking if
evaluation runs as a separate job per checkpoint rather than as one session over several.

## Reproduce

```bash
REV=$(git rev-parse HEAD)          # must be pushed: the job fetches this exact commit
uv run --frozen --project envs/nayana_ocr hf jobs uv run \
  --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  --volume hf://buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  train/hf_job.py --revision "$REV" --mode eval-vllm \
  --corpus-manifest repo --evalset eval-500.json \
  --models google/gemma-4-E2B-it google/gemma-4-E4B-it \
  --workers 48 --grade-workers 8
```

For checkpoints of one run, swap `--models` for `--base` plus `--adapters`, which boots one
engine and loads each checkpoint over it:

```bash
  --base google/gemma-4-E4B-it \
  --adapters step50=/outputs/run/checkpoint-50 step100=/outputs/run/checkpoint-100
```

`--gpus 2` puts one engine on each card via data parallelism. It is wired but unused here:
at 11.6 task/s a checkpoint already scores in under a minute, so the second card would
only shorten the part that is no longer the bottleneck.

**These numbers are on the test split.** `eval-500.json` is the only frozen set this
environment has, so scoring it here means a post-training number on the same set is
measured on data already used to compare models. That was a deliberate call to get numbers
quickly; a separate validation set would have kept the final number clean.
