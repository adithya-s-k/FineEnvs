# Training

One GRPO trainer, `grpo_nayana.py`, for a local GPU and for HF Jobs alike. It talks to the
environment over HTTP, either a server it starts from the corpus manifest or one already running,
including the deployed Space (`--env-url`). On a bucket-backed server it streams the corpus: every
step gets crops it has not seen, in a shuffled order that a seed reproduces.

## The Kannada run

Two jobs, launched one after the other from a pushed commit. The first trains. The second follows
the first's output bucket and scores every checkpoint on Sarvam Indic OCR Bench as soon as it is saved.

```bash
# Train: 500 steps x 8 crops = 4,000 distinct Kannada section crops from the corpus.
hf jobs uv run -d --flavor a100-large -s HF_TOKEN --timeout 14h \
  -e NAYANA_INDIC_OCR_BENCH_ROOT=/indic-ocr-bench \
  -v hf://buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  -v hf://buckets/FineEnvs/indic-ocr-bench-bucket:/indic-ocr-bench:ro \
  -v hf://buckets/<you>/fineenvs-ocr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode train --corpus-manifest repo --source-root /corpus \
  --output-root /outputs --evalset eval-nayana-kn-section-validation.json \
  --model google/gemma-4-E4B-it --languages kn --families section_ocr \
  --max-steps 500 --num-generations 8 --gradient-accumulation-steps 8 --max-completion-length 512 \
  --learning-rate 5e-5 --lr-scheduler-type cosine --warmup-ratio 0.05 \
  --eval-limit 48 --save-steps 25 --save-total-limit 30 \
  --trackio-space <you>/fineenvs-ocr-trackio --run-name ocr-kn-v2

# Score every checkpoint on the Kannada part of Sarvam Indic OCR Bench as it lands.
hf jobs uv run -d --flavor a100-large -s HF_TOKEN --timeout 16h \
  -e NAYANA_INDIC_OCR_BENCH_ROOT=/indic-ocr-bench \
  -v hf://buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  -v hf://buckets/FineEnvs/indic-ocr-bench-bucket:/indic-ocr-bench:ro \
  -v hf://buckets/<you>/fineenvs-ocr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode eval-vllm --corpus-manifest repo --source-root /corpus \
  --output-root /outputs --split indic_ocr_bench_test --languages kn \
  --base google/gemma-4-E4B-it --watch <you>/fineenvs-ocr-runs/<commit> --follow-job <train job id> \
  --output-dir /outputs/<commit>/evals \
  --trackio-space <you>/fineenvs-ocr-trackio --run-name ocr-kn-v2-eval
```

Training took 10.5 hours. The scoring job holds one vLLM engine and loads each checkpoint into it as
an adapter, so it sits idle most of the time. Any GPU with 24 GB or more works. An L40S is cheaper,
but none were free when this ran. If the scoring job dies, relaunch it with the same arguments: it
picks up from its own `curve.json`.

Its output, under `<commit>/evals/` in the bucket:

| | |
|---|---|
| `curve.json` | each checkpoint's means, and its change from the base model, paired per crop, with a 95% interval |
| `step-N.json` | every crop's prediction, the reward, and Sarvam's CER and WER for checkpoint N, plus Sarvam's own report |
| `plots/` | the held-out curve, the paired change, and the training signals, redrawn after every checkpoint |

The same points go to Trackio, next to the training run. To score any finished checkpoint on the
benchmark instead, use `--adapters name=/path/to/checkpoint` in place of `--watch`.

## A quick check first

Two optimizer steps against the deployed Space, on any CUDA GPU:

```bash
uv run --frozen --project envs/nayana_ocr --extra train python train/grpo_nayana.py \
  --env-url https://fineenvs-nayana-ocr-env.hf.space --smoke \
  --output-dir artifacts/local-gpu-smoke
```

## What each script does

| Script | |
|---|---|
| `grpo_nayana.py` | the GRPO trainer. `--languages` and `--families` choose what to train on; `--evalset` and `--eval-limit` set the small in-run evaluation |
| `hf_job.py` | runs any of these on HF Jobs from a pushed commit, with that commit's lockfile |
| `eval_vllm.py` | scores models or checkpoints through vLLM, on a frozen set or a served split; `--watch` follows a live run |
| `plot_run.py` | draws a run's figures from its `curve.json` and trainer log |
| `publish_indic_ocr_bench.py` | builds the benchmark's serving copy once and publishes it to a bucket |
| `deploy_space.py` | deploys the environment to a Space, with the corpus and benchmark buckets mounted |
| `verify_judge.py` | calibrates the descriptive-VQA judge before a deployment uses it |
| `verify_corpus.py`, `verify_playground.py`, `verify_evalset.py` | check the served corpus, the playground, and a frozen evaluation set |
| `benchmark_corpus.py`, `benchmark_job.py` | measure the data path locally, against the Space, or on a CPU job |

[REPRODUCE.md](../REPRODUCE.md) has the exact commands and defaults for each. Run outputs go to
`artifacts/`, which git ignores.

## Notes

**Only task IDs reach the trainer.** The environment owns the prompt and the image, so the dataset
carries a task id, its language and family, and a few spare ids in case a page cannot be rendered.

**Both the language model and the vision tower are adapted.** Images do reach TRL's loss, so LoRA on
the vision tower trains. Audio does not, which is the bug the [ASR half](../../asr/) had to fix.

**A larger learning rate gets a warmup.** `--lr-scheduler-type cosine --warmup-ratio 0.05` keeps
early steps small, when the advantage estimates are noisiest. `--beta` adds a KL penalty to the base
model; it is 0 by default, and the Kannada run used 0.

**Check images before a long run on new data.** A page past 50 megapixels cannot be served, and that
is only known once it is decoded. Frozen evaluation sets record which pages they replaced; check a
new set with `verify_evalset.py` before relying on it.
