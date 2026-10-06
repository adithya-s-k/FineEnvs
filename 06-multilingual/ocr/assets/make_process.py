# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4", "pillow"]
# ///
"""'How it trains' GIF: each stage of one GRPO step beside the code that runs it.

    uv run --script make_process.py asr how-it-trains.gif
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# The published GIFs use macOS system faces; elsewhere matplotlib's own stand in.
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
if Path("/System/Library/Fonts/Menlo.ttc").exists():
    for path in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Menlo.ttc"):
        font_manager.fontManager.addfont(path)
    SANS, MONO = "Helvetica Neue", "Menlo"
INK, SUB, FAINT, LINE = "#111827", "#6b7280", "#9ca3af", "#e5e7eb"
CODE_BG, CODE_FG, CODE_DIM = "#0d1117", "#e6edf3", "#7d8590"
SYNTAX = {"comment": "#7d8590", "string": "#a5d6ff", "keyword": "#ff7b72", "number": "#ffa657",
          "name": "#d2a8ff"}
KEYWORDS = {"def", "return", "for", "in", "with", "class", "import", "from", "lambda", "if", "else"}
COLOUR = {"asr": "#6d28d9", "ocr": "#0f9f6e"}
URL = "huggingface.co/collections/FineEnvs/multilingual-multimodal-envs"

ASR = [
    ("Launch on HF Jobs", "one command, a pinned commit, one A100",
     "train/hf_job.py", """\
hf jobs uv run --flavor a100-large -s HF_TOKEN \\
  -v hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \\
  -v hf://buckets/<you>/fineenvs-asr-runs:/outputs \\
  train/hf_job.py --revision <commit> --mode train \\
  --model google/gemma-4-E4B-it --languages kn_in \\
  --train-per-group 2282 --max-steps 575 \\
  --num-generations 16 --gradient-accumulation-steps 4 \\
  --learning-rate 5e-5 --lr-scheduler-type cosine \\
  --reward-unit cer --save-steps 25"""),
    ("The environment serves a clip", "OpenEnv server over the FLEURS bucket",
     "envs/multilingual_asr/training.py", """\
class TrainingEnvironment:
    def reset(self, task_id, **kwargs):
        obs = self.client.reset(task_id=task_id).observation
        # 16 kHz FLEURS audio, fetched by hash and verified
        return [
            {"type": "audio", "audio": self.cache.audio(obs)},
            {"type": "text", "text": obs.prompt},
        ]"""),
    ("Gemma samples 16 transcripts", "TRL GRPOTrainer drives the rollouts",
     "train/grpo_asr.py", """\
trainer = audio_grpo_trainer()(
    model="google/gemma-4-E4B-it",
    processing_class=pad_audio_features(processor, 30),
    train_dataset=Dataset.from_list(train_rows),  # task ids only
    environment_factory=factory,
    reward_funcs=env_reward,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=lm),
    args=GRPOConfig(num_generations=16, temperature=0.9,
                    gradient_accumulation_steps=4, learning_rate=5e-5),
)"""),
    ("The server grades each one", "reward = 0.8 (1 - CER) + 0.2 exact",
     "envs/multilingual_asr/server/rewards.py", """\
def score(task, prediction, unit="cer"):
    rate, natural = error_rate(prediction, reference, language, family)
    cer = character_error_rate(prediction, reference, language, family)
    exact = normalize(prediction) == normalize(reference)
    rewarded = cer if unit == "cer" else rate
    reward = 0.8 * max(0.0, 1.0 - rewarded) + 0.2 * float(exact)
    return reward, {natural: rate, "cer": cer, "exact_match": exact}"""),
    ("The loss hears the clip", "-0.38 nats/token with the audio, -6.22 without",
     "envs/multilingual_asr/training.py", """\
class AudioGRPOTrainer(GRPOTrainer):
    # TRL 1.13 drops input_features before the loss forward,
    # so every earlier run trained p(transcript | no audio).
    def _get_per_token_logps_and_entropies(self, model, ids, *a, **kw):
        rows = self._logp_audio          # stashed when prompts tokenize
        with audio_forward(model, rows):  # attach each chunk's clips
            return super()._get_per_token_logps_and_entropies(
                model, ids, *a, **kw)"""),
    ("Every 25 steps, a checkpoint is scored", "838 held-out clips, paired against the base model",
     "train/eval_vllm.py", """\
# a second job, one vLLM engine, adapters hot-loaded as they land
hf jobs uv run ... train/hf_job.py --mode eval-vllm \\
  --base google/gemma-4-E4B-it \\
  --watch <you>/fineenvs-asr-runs/<commit> --follow-job <id> \\
  --eval-split eval_1_test --reward-unit cer \\
  --trackio-space <you>/fineenvs-asr-trackio

# step 575: CER 0.105 -> 0.057 (-45%), WER 0.312 -> 0.243"""),
]

OCR = [
    ("Launch on HF Jobs", "one command, a pinned commit, one A100",
     "train/hf_job.py", """\
hf jobs uv run --flavor a100-large -s HF_TOKEN \\
  -v hf://buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \\
  -v hf://buckets/<you>/fineenvs-ocr-runs:/outputs \\
  train/hf_job.py --revision <commit> --mode train \\
  --corpus-manifest repo --source-root /corpus \\
  --model google/gemma-4-E4B-it --languages kn \\
  --families section_ocr --max-steps 500 \\
  --num-generations 8 --gradient-accumulation-steps 8 \\
  --learning-rate 5e-5 --lr-scheduler-type cosine"""),
    ("The environment serves a crop", "streamed from 1M indexed pages in a bucket",
     "envs/nayana_ocr/training.py", """\
class TrainingEnvironment:
    def reset(self, task_id, spare_task_ids=(), **kwargs):
        obs = self.client.reset(task_id=task_id).observation
        image = self.cache.image(obs)  # verified by sha256
        return [
            {"type": "image", "image": image},
            {"type": "text", "text": obs.prompt},
        ]"""),
    ("Gemma samples 8 transcriptions", "TRL GRPOTrainer drives the rollouts",
     "train/grpo_nayana.py", """\
trainer = GRPOTrainer(
    model="google/gemma-4-E4B-it",
    processing_class=processor,
    train_dataset=build_corpus_dataset(url, ...),  # streamed, shuffled
    environment_factory=factory,
    reward_funcs=env_reward,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=targets),
    args=GRPOConfig(num_generations=8, temperature=0.9,
                    gradient_accumulation_steps=8, learning_rate=5e-5),
)"""),
    ("The server grades each one", "0.8 (1 - CER) + 0.2 exact; zero if overlong",
     "envs/nayana_ocr/server/rewards.py", """\
def score(family, prediction, reference):
    overlong = len(prediction) > max(1024, 4 * len(reference))
    cer = char_error_rate(predicted, target)   # NFC, rapidfuzz
    exact = predicted == target and not overlong
    reward = 0.0 if overlong else 0.8 * max(0.0, 1.0 - cer) + 0.2 * exact
    return reward, {"char_error_rate": cer, "exact_match": exact}"""),
    ("Benchmarks report beside the reward", "Sarvam's own metrics.py, vendored unmodified",
     "envs/nayana_ocr/server/bench_rewards.py", """\
def official_metrics(prediction, reference):
    gt, pred = official.preprocess(reference, prediction, normalize=True)
    rates = official.calculate_ocr_metrics(gt, pred)
    looped, _ = official.is_loop_or_catastrophic(reference, prediction)
    return {"official_cer": rates["cer"], "official_wer": rates["wer"],
            "loop_or_catastrophic": looped}   # reported, never rewarded"""),
    ("Every 25 steps, a checkpoint is scored", "Sarvam bench, Kannada, paired against base",
     "train/eval_vllm.py", """\
# a second job, one vLLM engine, adapters hot-loaded as they land
hf jobs uv run ... train/hf_job.py --mode eval-vllm \\
  -v hf://buckets/FineEnvs/indic-ocr-bench-bucket:/indic-ocr-bench:ro \\
  --split indic_ocr_bench_test --languages kn \\
  --base google/gemma-4-E4B-it \\
  --watch <you>/fineenvs-ocr-runs/<commit> --follow-job <id>

# step 300: Sarvam CER 0.428 -> 0.360 (-16%)"""),
]

TITLE = {"asr": ("How the Kannada ASR run trains", "one GRPO step, end to end, and the code that runs it"),
         "ocr": ("How the Kannada OCR run trains", "one GRPO step, end to end, and the code that runs it")}


def tokens(line):
    """Very small Python/shell highlighter: (text, colour) pieces."""
    if line.lstrip().startswith("#"):
        return [(line, SYNTAX["comment"])]
    out = []
    pattern = re.compile(r'(#.*$)|("[^"]*")|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|(\b[A-Za-z_]\w*\b)|(\s+)|(.)')
    for m in pattern.finditer(line):
        text = m.group(0)
        if m.group(1):
            out.append((text, SYNTAX["comment"]))
        elif m.group(2):
            out.append((text, SYNTAX["string"]))
        elif m.group(3):
            out.append((text, SYNTAX["number"]))
        elif m.group(4):
            if text in KEYWORDS:
                out.append((text, SYNTAX["keyword"]))
            elif re.match(r"[A-Z]", text) or text in {"score", "reset", "official_metrics"}:
                out.append((text, SYNTAX["name"]))
            else:
                out.append((text, CODE_FG))
        else:
            out.append((text, CODE_FG))
    return out


def build(run, out_gif):
    stages = ASR if run == "asr" else OCR
    colour = COLOUR[run]
    fig = plt.figure(figsize=(12, 5.6), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    title, sub = TITLE[run]
    fig.text(0.018, 0.925, title, fontsize=21, fontweight="bold", color=INK, family=SANS)
    fig.text(0.018, 0.875, sub, fontsize=10.5, color=SUB, family=MONO)
    fig.text(0.982, 0.03, URL, fontsize=9.5, color=SUB, family=MONO, ha="right")

    # Code card: left 62%.
    fig.patches.append(FancyBboxPatch((0.018, 0.08), 0.6, 0.76, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=CODE_BG, edgecolor=CODE_BG,
                                      zorder=-5))
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        fig.add_artist(plt.Circle((0.035 + i * 0.016, 0.81), 0.0055, color=c, transform=fig.transFigure))
    filename = fig.text(0.09, 0.802, "", fontsize=10, color=CODE_DIM, family=MONO)

    # Pipeline: right column of stage boxes.
    boxes, labels, notes = [], [], []
    top, height, gap = 0.80, 0.088, 0.03
    for i, (name, note, _, _) in enumerate(stages):
        y = top - i * (height + gap) - height
        box = FancyBboxPatch((0.64, y), 0.342, height, boxstyle="round,pad=0,rounding_size=0.012",
                             transform=fig.transFigure, facecolor="#ffffff", edgecolor=LINE, lw=1.2,
                             zorder=-5)
        fig.patches.append(box)
        fig.text(0.652, y + height * 0.56, f"{i + 1}", fontsize=13, fontweight="bold", color=FAINT,
                 family=MONO, va="center")
        labels.append(fig.text(0.676, y + height * 0.62, name, fontsize=11.5, fontweight="bold",
                               color=SUB, family=SANS, va="center"))
        notes.append(fig.text(0.676, y + height * 0.27, note, fontsize=8.2, color=FAINT, family=MONO,
                              va="center"))
        boxes.append(box)
        if i < len(stages) - 1:
            fig.add_artist(plt.Line2D([0.81, 0.81], [y - gap + 0.004, y - 0.004], color="#d1d5db", lw=1.2,
                                      transform=fig.transFigure))

    code_texts = []
    # Measure one monospace character once, so tokens butt up exactly.
    probe = fig.text(0, 0, "M" * 40, fontsize=10.2, family=MONO)
    char_width = probe.get_window_extent(fig.canvas.get_renderer()).width / 40 / fig.bbox.width
    probe.remove()
    per_stage, typing = 34, 14  # frames per stage, frames spent "typing"

    def show(stage, frame_in_stage):
        for t in code_texts:
            t.remove()
        code_texts.clear()
        _, _, path, code = stages[stage]
        filename.set_text(path)
        lines = code.split("\n")
        visible = len(lines) if frame_in_stage >= typing else max(1, round(len(lines) * (frame_in_stage + 1) / typing))
        for row, line in enumerate(lines[:visible]):
            x = 0.035
            y = 0.745 - row * 0.05
            for text, col in tokens(line):
                if not text:
                    continue
                t = fig.text(x, y, text, fontsize=10.2, color=col, family=MONO, va="center")
                code_texts.append(t)
                x += char_width * len(text)
        for i, box in enumerate(boxes):
            active = i == stage
            done = i < stage
            box.set_edgecolor(colour if active else ("#c4b5fd" if run == "asr" and done else
                                                      "#a7f3d0" if done else LINE))
            box.set_linewidth(2.2 if active else 1.2)
            box.set_facecolor("#ffffff")
            labels[i].set_color(INK if active or done else SUB)
            notes[i].set_color(colour if active else FAINT)

    total = per_stage * len(stages) + 10

    def draw(frame):
        stage = min(frame // per_stage, len(stages) - 1)
        show(stage, frame - stage * per_stage if frame < per_stage * len(stages) else per_stage)
        return []

    FuncAnimation(fig, draw, frames=total, interval=80).save(out_gif, writer=PillowWriter(fps=12))
    print("wrote", out_gif)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
