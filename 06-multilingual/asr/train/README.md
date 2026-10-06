# Training

One GRPO trainer, `grpo_asr.py`, for a local GPU and for HF Jobs alike. It talks to the
environment over HTTP, either a server it starts from the corpus manifest (`--corpus`) or one
already running, including the deployed Space (`--env-url`).

## The Kannada run

Two jobs, launched one after the other from a pushed commit. The first trains. The second follows
the first's output bucket and scores every checkpoint on the held-out clips as soon as it is saved.

```bash
# Train: every Kannada clip in FLEURS, rewarded on character error.
hf jobs uv run -d --flavor a100-large -s HF_TOKEN --timeout 14h \
  -v hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  -v hf://buckets/<you>/fineenvs-asr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode train --source-root /fleurs --output-root /outputs \
  --model google/gemma-4-E4B-it --languages kn_in --families transcription \
  --train-per-group 2282 --max-steps 575 --num-generations 16 --gradient-accumulation-steps 4 \
  --max-completion-length 448 --learning-rate 5e-5 --lr-scheduler-type cosine --warmup-ratio 0.05 \
  --reward-unit cer --eval-split eval_1_validation --eval-limit 48 --save-steps 25 \
  --trackio-space <you>/fineenvs-asr-trackio --run-name asr-kn-full-v3

# Score every checkpoint on the 838 Kannada test clips as it lands.
hf jobs uv run -d --flavor a100-large -s HF_TOKEN --timeout 16h \
  -v hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  -v hf://buckets/<you>/fineenvs-asr-runs:/outputs \
  train/hf_job.py --revision <commit> --mode eval-vllm --source-root /fleurs --output-root /outputs \
  --base google/gemma-4-E4B-it --watch <you>/fineenvs-asr-runs/<commit> --follow-job <train job id> \
  --eval-split eval_1_test --max-new-tokens 448 --reward-unit cer \
  --output-dir /outputs/<commit>/evals \
  --trackio-space <you>/fineenvs-asr-trackio --run-name asr-kn-full-v3-eval
```

Training took 4.7 hours. The scoring job holds one vLLM engine and loads each checkpoint into it as
an adapter, about 80 seconds per checkpoint, so it sits idle most of the time. Any GPU with 24 GB or
more works. An L40S is cheaper, but none were free when this ran. If the scoring job dies, relaunch
it with the same arguments: it picks up from its own `curve.json`.

Its output, under `<commit>/evals/` in the bucket:

| | |
|---|---|
| `curve.json` | each checkpoint's means, and its change from the base model, paired per clip, with a 95% interval |
| `step-N.json` | every clip's prediction and score for checkpoint N |
| `plots/` | the held-out curve, the paired change, and the training signals, redrawn after every checkpoint |

The same points go to Trackio, next to the training run.

## A quick check first

Four steps on two languages, against a server the script starts itself. It runs on any CUDA GPU
and checks the whole loop before you spend real hours:

```bash
uv run --frozen --project envs/multilingual_asr --extra train python train/grpo_asr.py \
  --corpus data/corpus-manifest.json \
  --languages en_us hi_in --families transcription \
  --eval-split eval_21_test --eval-limit 8 \
  --num-generations 8 --max-steps 4 --smoke
```

## How the trainer works

**The audio reaches the loss.** TRL 1.13 drops audio features from the forward pass its loss is
computed from, so a stock GRPO run trains a model that cannot hear. `AudioGRPOTrainer` carries
each batch's audio into every log-prob pass. On the first batch it checks that the clip makes the
model's own transcripts more likely, and stops if it does not. On the Kannada run the clip was
worth 5.8 nats per token. [LEARNINGS.md](../LEARNINGS.md) has the story.

**Padding is masked.** Clips of different lengths have to share one shape in a batch. The feature
extractor pads them to 30 seconds and marks the padding as not audio, so the model hears exactly
what vLLM serves at evaluation time.

**Only task IDs reach the trainer.** TRL reads a `prompt` column as a conversation, and the
environment already owns the prompt, so the dataset carries the task id, language and family and
nothing else.

**Sampling never pages a split.** The train split holds 815,226 tasks. The server answers "how many
Kannada transcription tasks" and "give me positions 12, 507 and 1,940" straight from its index, so
drawing a sample takes seconds. The draw is seeded, so a run reproduces.

**Held-out sets are frozen.** `--eval-split` scores a committed set, whole unless `--eval-limit` is
given, and the run records the set's id and any limit, so a slice is never mistaken for a full
score. Each set has a validation twin, so choosing a checkpoint never touches the set a final
number is reported on.

**Only the language model is adapted.** LoRA targets are read from the model's real module tree,
because Gemma 4 wraps some projections in a layer PEFT cannot adapt. The audio and vision towers
are left out: no gradient reaches them through TRL, and an adapter there would sit at zero.

**A larger learning rate gets a warmup.** `--lr-scheduler-type cosine --warmup-ratio 0.05` keeps
early steps small, when the advantage estimates are noisiest. `--beta` adds a KL penalty to the
base model. It is 0 by default, as in TRL, and the Kannada run used 0.

**A run needs spread in its rewards.** If every transcript in a group scores the same, GRPO's
advantage is zero and nothing can be learned. The trainer reports that rather than a pass. Four
samples per clip was not enough on easy English clips; eight was, and the Kannada run used sixteen.
