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
eval_21_test` scores the 21 languages shared with `06-multilingual-ocr`, so an ASR result
can be read against an OCR result language for language; `eval_102_test` covers all 102.
The `_validation` variants exist so model selection never touches the set a final number is
reported on. A frozen split is used whole unless `--eval-limit` is given, and the run
records the set's `evalset_id` and the limit, so a slice is never mistaken for a full score.

**LoRA targets are resolved from the real module tree**, not named. Gemma 4 wraps
projections in `Gemma4ClippableLinear`, which PEFT cannot adapt, and a short target name
matches both a wrapper and a leaf, so injection fails outright. Full module names name one
leaf each. Only the language model is adapted. TRL never backpropagates through the audio
tower, so adapters there stayed exactly zero.

**The loss hears the clip.** TRL 1.13 builds its loss inputs from images only. The audio
features a completion was generated from were dropped before the log-prob forward, so
every run before this one optimised p(transcript | no audio). That is a language prior
over the training sentences, not listening, which is why training reward rose while
held-out ASR barely moved. `AudioGRPOTrainer` carries `input_features` and their mask
through the generation batch into every log-prob forward. On the first batch it checks
that the clip makes its own transcripts more likely. The first run with it measured
-0.20 nats/token with the clip against -4.53 without, a gap the old gradient never saw.

**Features are padded, not audio.** A batch of clips needs one feature shape. Padding the
waveform to 30 s made the extractor mark about 18 s of silence as valid speech, which vLLM
never sees. The extractor now pads features to the span with the padding masked, and only
real frames become audio tokens.

**A higher learning rate gets a warmup and a cosine tail.** `--lr-scheduler-type cosine
--warmup-ratio 0.05` keeps steps small while advantage estimates are noisiest. `--beta`
adds a KL penalty to the base model; it is 0 by default, as in TRL.

**Every checkpoint is scored while the run trains.** Each save writes `ready.json`, the
adapter's size and hash. A second job runs `eval_vllm.py --watch` against the run's
bucket. It holds one vLLM engine, scores the base model once, then scores each checkpoint
once all its bytes have arrived. Results go under `<run>/evals/`. `curve.json` holds each
checkpoint's means and its paired change from base with a 95% interval, `plots/` the
figures from `plot_run.py`, and every point is logged to Trackio beside the training run.

```bash
# The run: all 2,282 Kannada training clips, character-error reward.
hf jobs uv run -d --flavor a100-large -s HF_TOKEN --timeout 14h \
  -v hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  -v hf://buckets/<you>/fineenvs-asr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode train --source-root /fleurs --output-root /outputs \
  --model google/gemma-4-E4B-it --languages kn_in --families transcription \
  --train-per-group 2282 --max-steps 575 --num-generations 16 --gradient-accumulation-steps 4 \
  --max-completion-length 448 --learning-rate 5e-5 --lr-scheduler-type cosine --warmup-ratio 0.05 \
  --reward-unit cer --eval-split eval_1_validation --eval-limit 48 --save-steps 25 \
  --trackio-space <you>/fineenvs-asr-trackio --run-name asr-kn-full-v3

# Its watcher: the 838-clip Kannada test set, scored at every checkpoint.
hf jobs uv run -d --flavor l40sx1 -s HF_TOKEN --timeout 16h \
  -v hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  -v hf://buckets/<you>/fineenvs-asr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode eval-vllm --source-root /fleurs --output-root /outputs \
  --base google/gemma-4-E4B-it --watch <you>/fineenvs-asr-runs/<commit> --follow-job <train job id> \
  --eval-split eval_1_test --max-new-tokens 448 --reward-unit cer \
  --output-dir /outputs/<commit>/evals \
  --trackio-space <you>/fineenvs-asr-trackio --run-name asr-kn-full-v3-eval
```

The watcher needs about 20 GB of GPU memory and is idle between checkpoints. L40S capacity was
scarce when these runs launched: the OCR watcher waited 1.5 hours and was relaunched on
`a100-large`, which resumes from `evals/curve.json` without rescoring anything.

**A GRPO run needs reward variance.** Identical rewards within a group make the advantage
zero by construction, the adapter cannot change, and the run proves nothing. The runner
checks this and fails rather than reporting a pass: four generations on an easy English
clip is not enough, eight is. Report per-language scores separately — a macro average over
102 languages hides exactly the low-resource behaviour this corpus exists to measure.
