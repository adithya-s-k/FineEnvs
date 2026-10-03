# Training and deployment entry points

- `grpo_nayana.py`: one-GPU LoRA GRPO against a local catalog or OpenEnv URL. `--task-input auto`
  selects the full-corpus block iterator when the server advertises bucket-backed storage.
  `--task-input corpus` requires it explicitly. The evaluation sample is indexed and balanced;
  the full training pass keeps natural language/task proportions within the selected groups.
- `hf_job.py`: fetch a pushed Git commit and use its frozen lockfile. Supports CPU smoke,
  small-window smoke, and training. Pass the full Space with `--env-url` or a bucket manifest
  with `--corpus-manifest`; a job can attach the source bucket at `/corpus`.
- `eval_vllm.py`: score models or a run's checkpoints over one vLLM engine, on a frozen
  evalset or a served split (`--split indic_ocr_bench_test --languages kn` is the Kannada
  slice of Sarvam Indic OCR Bench). `--watch <bucket>/<run>` follows a run while it
  trains. Each save writes `ready.json`, the adapter's size and hash, and the watcher
  scores the base once and then each checkpoint when all its bytes have arrived. It writes
  `evals/curve.json` with means and the paired change from base (95% interval), redraws
  `evals/plots/` with `plot_run.py`, and logs every point to Trackio.
- `deploy_space.py`: publish code and a ready full-corpus manifest, remove the old bundled
  preview, and attach the existing bucket read-only. Indexes must already be published.
- `verify_judge.py`: calibrate the configured Gemma model/provider against multilingual
  references and strictness challenges before deployment.
- `verify_corpus.py`: check indexed discovery, prefetch, cache reuse, and reference scoring.
- `verify_playground.py`: check Gradio scoring, indexed navigation, and session isolation.
- `benchmark_corpus.py`: cold random evaluation, warm GRPO-style requests, sustained
  cached capacity, and multi-block prefetch, using the actual image/reward adapter.
- `benchmark_job.py`: run that benchmark at a pushed commit on a CPU Job with a bucket
  mount, returning its complete measured report through the job logs.

See [REPRODUCE.md](../REPRODUCE.md) for exact commands, defaults, and replay boundaries.
Write generated reports and checkpoints to `artifacts/`, which is ignored by Git. HF Jobs
can upload training outputs to an explicit artifact repository. GPU optimizer training
remains unverified; preflight image eligibility before a full-corpus run.

A Kannada run with live benchmark scoring, as launched on 2026-10-02:

```bash
# Training: 500 steps x 8 prompts = 4,000 distinct Kannada section crops streamed from the corpus.
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

# Watcher: every checkpoint on the Kannada slice of Sarvam Indic OCR Bench (test).
hf jobs uv run -d --flavor l40sx1 -s HF_TOKEN --timeout 16h \
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

The watcher needs about 20 GB of GPU memory and is idle between checkpoints. L40S capacity was
scarce when these runs launched: the OCR watcher waited 1.5 hours and was relaunched on
`a100-large`, which resumes from `evals/curve.json` without rescoring anything.
