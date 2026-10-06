# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Draw a run's figures from its evaluation curve and its trainer log history.

Run by the checkpoint watcher after every score, and by hand on any run's files:

    uv run --script train/plot_run.py --curve curve.json \
        --trainer-state checkpoint-500/trainer_state.json --out plots/

Writes eval_curve.png (each metric's mean against step, the base model dashed),
eval_change.png (each metric's paired change from base with its 95% interval), and
training.png (the trainer's own signals, smoothed, raw values faint behind).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, FAINT, GRID = "#1f2937", "#9ca3af", "#e5e7eb"
LOWER_IS_BETTER = {"cer", "wer", "char_error_rate", "official_cer", "official_wer"}
TRAINING = (
    ("reward", "reward"),
    ("reward_std", "reward std within a group"),
    ("frac_reward_zero_std", "groups with no reward spread"),
    ("loss", "loss"),
    ("grad_norm", "grad norm"),
    ("learning_rate", "learning rate"),
    ("kl", "KL to base"),
    ("entropy", "entropy"),
    ("completions/mean_length", "completion length"),
    ("completions/clipped_ratio", "completions truncated"),
)


def style(axis, title):
    axis.set_title(title, fontsize=10, color=INK, loc="left")
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=8, colors=INK)


def grid(count, width=4.2, height=2.8):
    columns = min(count, 3)
    rows = -(-count // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows), squeeze=False
    )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return figure, list(axes.flat)


def smooth(values, window):
    out, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def eval_figures(curve, out, title):
    points = sorted(curve, key=lambda p: p["step"])
    base = next((p for p in points if p["step"] == 0), None)
    metrics = [k for k in points[0] if isinstance(points[0][k], float)] if points else []
    written = []
    if metrics:
        figure, axes = grid(len(metrics))
        for axis, metric in zip(axes, metrics, strict=False):
            steps = [p["step"] for p in points if metric in p]
            axis.plot(steps, [p[metric] for p in points if metric in p], color=INK,
                      marker="o", markersize=3, linewidth=1.5)
            if base and metric in base:
                axis.axhline(base[metric], color=FAINT, linestyle="--", linewidth=1)
            arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
            style(axis, f"{metric} ({arrow})")
            axis.set_xlabel("step", fontsize=8)
        figure.suptitle(f"{title}: held-out mean (dashed: base model)", fontsize=11,
                        color=INK, x=0.01, ha="left")
        figure.tight_layout()
        figure.savefig(out / "eval_curve.png", dpi=130)
        plt.close(figure)
        written.append(out / "eval_curve.png")
    changed = [m for m in metrics if any(f"{m}_change" in p for p in points)]
    if changed:
        figure, axes = grid(len(changed))
        for axis, metric in zip(axes, changed, strict=False):
            rows = [(p["step"], p[f"{metric}_change"]) for p in points
                    if f"{metric}_change" in p]
            steps = [s for s, _ in rows]
            axis.fill_between(steps, [c["low"] for _, c in rows],
                              [c["high"] for _, c in rows], color=GRID, linewidth=0)
            axis.plot(steps, [c["delta"] for _, c in rows], color=INK, marker="o",
                      markersize=3, linewidth=1.5)
            axis.axhline(0, color=FAINT, linewidth=1)
            arrow = "below 0 is better" if metric in LOWER_IS_BETTER else "above 0 is better"
            style(axis, f"{metric} change ({arrow})")
            axis.set_xlabel("step", fontsize=8)
        figure.suptitle(f"{title}: paired change from base, 95% interval shaded",
                        fontsize=11, color=INK, x=0.01, ha="left")
        figure.tight_layout()
        figure.savefig(out / "eval_change.png", dpi=130)
        plt.close(figure)
        written.append(out / "eval_change.png")
    return written


def training_figure(state, out, title):
    history = [h for h in state.get("log_history", []) if "step" in h]
    panels = [(key, label) for key, label in TRAINING if any(key in h for h in history)]
    if not panels:
        return []
    figure, axes = grid(len(panels))
    for axis, (key, label) in zip(axes, panels, strict=False):
        rows = [(h["step"], h[key]) for h in history if key in h]
        steps, values = [s for s, _ in rows], [float(v) for _, v in rows]
        axis.plot(steps, values, color=FAINT, linewidth=0.6)
        axis.plot(steps, smooth(values, 20), color=INK, linewidth=1.5)
        style(axis, label)
        axis.set_xlabel("step", fontsize=8)
    figure.suptitle(f"{title}: training (20-step mean over raw)", fontsize=11,
                    color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(out / "training.png", dpi=130)
    plt.close(figure)
    return [out / "training.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written = eval_figures(json.loads(args.curve.read_text()), args.out, args.title)
    if args.trainer_state and args.trainer_state.exists():
        state = json.loads(args.trainer_state.read_text())
        written += training_figure(state, args.out, args.title)
    for path in written:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
