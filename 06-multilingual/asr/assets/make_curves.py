# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4", "pillow"]
# ///
"""Animated training + held-out curves, in the style of 04-smoldataenvs/curves.gif.

Reads one run folder of FineEnvs/multilingual-multimodal-rl-runs:

    hf download FineEnvs/multilingual-multimodal-rl-runs --repo-type dataset --local-dir runs
    uv run --script make_curves.py asr runs/asr-kannada curves.gif curves.png
"""

import json
import pathlib
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# The published GIFs use macOS system faces; elsewhere matplotlib's own stand in.
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
if pathlib.Path("/System/Library/Fonts/Menlo.ttc").exists():
    for path in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Menlo.ttc"):
        font_manager.fontManager.addfont(path)
    SANS, MONO = "Helvetica Neue", "Menlo"
INK, SUB, FAINT, GRID, CARD, LINE = "#111827", "#6b7280", "#9ca3af", "#e5e7eb", "#ffffff", "#e5e7eb"
COLOUR = {"asr": "#6d28d9", "ocr": "#0f9f6e"}
URL = "huggingface.co/collections/FineEnvs/multilingual-multimodal-envs"


def capped(samples, key):
    return st.mean(min(float(s[key]), 1.0) for s in samples)


def load(run, d):
    d = pathlib.Path(d)
    lines = (d / "trainer_log_history.jsonl").read_text().splitlines()
    history = [h for h in map(json.loads, lines) if "reward" in h]
    base_file = d / "evals" / "base.json"
    files = [(0, base_file)] + sorted(
        (int(p.stem.split("-")[1]), p) for p in (d / "evals").glob("step-*.json"))
    evals = []
    for step, path in files:
        samples = json.loads(path.read_text())["samples"]
        if run == "asr":
            evals.append((step, capped(samples, "cer"), capped(samples, "wer")))
        else:
            evals.append((step, st.mean(float(s["official_cer"]) for s in samples),
                          st.mean(float(s["official_wer"]) for s in samples)))
    train_key = "asr/kn_in/transcription/cer" if run == "asr" else "nayana/kn/section_ocr/char_error_rate"
    train = {
        "reward": [(h["step"], h["reward"]) for h in history],
        "cer": [(h["step"], h[train_key]) for h in history if train_key in h],
    }
    return train, evals


def smooth(points, window=20):
    out, acc = [], []
    for s, v in points:
        acc = (acc + [v])[-window:]
        out.append((s, sum(acc) / len(acc)))
    return out


SPEC = {
    "asr": {
        "title": "Gemma 4 learns to hear Kannada",
        "sub": "GRPO · 575 steps · all 2,282 FLEURS train clips · reward = 1 − character error",
        "train": [("reward", "reward", "the training signal"), ("cer", "train CER", "on the clips it trains on")],
        "eval": [("CER", "character error, per clip"), ("WER", "word error, per clip")],
        "held": "838 FLEURS test clips it never trains on",
        "legend": "gemma-4-E4B-it + LoRA · Kannada speech",
    },
    "ocr": {
        "title": "Gemma 4 learns to read Kannada",
        "sub": "GRPO · 500 steps · 4,000 Nayana section crops · reward = 1 − character error",
        "train": [("reward", "reward", "the training signal"), ("cer", "train CER", "on the crops it trains on")],
        "eval": [("Sarvam CER", "official metric, per crop"), ("Sarvam WER", "official metric, per crop")],
        "held": "Sarvam Indic OCR Bench · 300 Kannada crops",
        "legend": "gemma-4-E4B-it + LoRA · Kannada documents",
    },
}


def card(fig, x, y, w, h):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor=LINE,
                                      linewidth=1.1, zorder=-5))


def build(run, folder, out_gif, out_png=None):
    spec, colour = SPEC[run], COLOUR[run]
    train, evals = load(run, folder)
    last = max(s for s, _ in train["reward"])
    fig = plt.figure(figsize=(12, 4.1), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    fig.text(0.018, 0.905, spec["title"], fontsize=21, fontweight="bold", color=INK, family=SANS)
    fig.text(0.018, 0.84, spec["sub"], fontsize=10.5, color=SUB, family=MONO)
    counter = fig.text(0.982, 0.885, "", fontsize=22, fontweight="bold", color=INK, family=MONO, ha="right")
    fig.text(0.018, 0.755, "TRAINING", fontsize=10.5, color=SUB, family=MONO)
    fig.text(0.482, 0.755, "what it optimises", fontsize=9, color=FAINT, family=MONO, ha="right")
    fig.text(0.518, 0.755, "HELD-OUT EVAL", fontsize=10.5, color=SUB, family=MONO)
    fig.text(0.982, 0.755, spec["held"], fontsize=9, color=FAINT, family=MONO, ha="right")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.12, 0.78], color="#d1d5db", lw=1, transform=fig.transFigure))
    fig.text(0.018, 0.045, "━━", fontsize=11, color=colour, family=MONO)
    fig.text(0.05, 0.045, spec["legend"], fontsize=10, color=INK, family=MONO)
    fig.text(0.40, 0.045, "┄┄ untuned base model", fontsize=10, color=FAINT, family=MONO)
    fig.text(0.982, 0.045, URL, fontsize=9.5, color=SUB, family=MONO, ha="right")

    lefts = [0.018, 0.255, 0.518, 0.755]
    panels = []
    names = [(n, t) for _, n, t in spec["train"]] + list(spec["eval"])
    for left, (name, note) in zip(lefts, names, strict=True):
        card(fig, left, 0.12, 0.227, 0.6)
        fig.text(left + 0.016, 0.645, name, fontsize=13, fontweight="bold", color=INK, family=SANS)
        fig.text(left + 0.016, 0.592, note, fontsize=8.5, color=SUB, family=MONO)
        ax = fig.add_axes([left + 0.035, 0.165, 0.18, 0.40])
        ax.set_facecolor("none")
        ax.set_zorder(5)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, labelsize=8, colors=SUB)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_family(MONO)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xticks([])
        panels.append(ax)

    raw = [train["reward"], train["cer"]]
    smoothed = [smooth(series) for series in raw]
    eval_series = [[(s, c) for s, c, _ in evals], [(s, w) for s, _, w in evals]]
    lines = []
    for ax, rs, sm in zip(panels[:2], raw, smoothed, strict=True):
        ys = sorted(v for _, v in rs)
        # Percentiles, so a single outlier step does not flatten the curve.
        lo, hi = ys[len(ys) // 100], ys[-1 - len(ys) // 100]
        pad = (hi - lo) * 0.08
        ax.set_xlim(0, last)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([round(lo, 2), round(hi, 2)])
        faint, = ax.plot([], [], color=colour, alpha=0.18, lw=0.8, clip_on=True)
        bold, = ax.plot([], [], color=colour, lw=2.2)
        lines.append((faint, bold, rs, sm))
    marks = []
    for ax, series in zip(panels[2:], eval_series, strict=True):
        ys = [v for _, v in series]
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.12
        ax.set_xlim(-last * 0.02, last * 1.02)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([round(lo, 3), round(hi, 3)])
        ax.axhline(series[0][1], color=FAINT, lw=1.2, ls=(0, (3, 3)))
        line, = ax.plot([], [], color=colour, lw=2.2, marker="o", ms=4)
        value = ax.text(0.98, 0.9, "", transform=ax.transAxes, ha="right", fontsize=10,
                        family=MONO, color=INK, fontweight="bold",
                        bbox={"facecolor": CARD, "edgecolor": "none", "pad": 1.5})
        marks.append((line, series, value, ys[0]))

    frames = 72
    hold = 26

    def draw(frame):
        step = last * min(frame, frames) / frames
        for faint, bold, rs, sm in lines:
            faint.set_data([s for s, _ in rs if s <= step], [v for s, v in rs if s <= step])
            bold.set_data([s for s, _ in sm if s <= step], [v for s, v in sm if s <= step])
        for line, series, value, base in marks:
            seen = [(s, v) for s, v in series if s <= step]
            line.set_data([s for s, _ in seen], [v for _, v in seen])
            if seen and seen[-1][0] > 0:
                change = (seen[-1][1] - base) / base * 100
                value.set_text(f"{seen[-1][1]:.3f}  ({change:+.0f}%)")
            else:
                value.set_text(f"{base:.3f}  base")
        counter.set_text(f"step {int(step)}")
        return []

    animation = FuncAnimation(fig, draw, frames=frames + hold, interval=70)
    animation.save(out_gif, writer=PillowWriter(fps=14))
    if out_png:
        draw(frames)
        fig.savefig(out_png, dpi=100, facecolor=fig.get_facecolor())
    print("wrote", out_gif, out_png or "")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
