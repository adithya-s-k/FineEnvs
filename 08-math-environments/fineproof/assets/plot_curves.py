"""Draw the README figure from results/fp-rl2-2b.json: training reward, held-out eval, cut-off proofs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#172033", "#64748b", "#e2e8f0"
BLUE, ORANGE, GREEN, DOT = "#2a78d6", "#eb6834", "#1baf7a", "#cbd5e1"


def rolling(values, window=10):
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1): i + 1] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def main():
    run = json.loads((ROOT.parent / "results" / "fp-rl2-2b.json").read_text())
    curve, points = run["train_curve"], run["eval"]["points"]
    regrades = run["regrade_gpt_oss_120b_high"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "svg.fonttype": "none", "svg.hashsalt": "fineproof-rl2-2b",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.14, wspace=0.26)
    updates = curve["update"]
    last = updates[-1]

    ax = axes[0]
    ax.scatter(updates, curve["reward"], s=6, color=DOT, linewidths=0, label="per update (128 rollouts)")
    ax.plot(updates, rolling(curve["reward"]), color=BLUE, lw=2, label="10-update mean")
    ax.set_title("Training reward", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, 0.75)

    ax = axes[1]
    steps = [p["step"] for p in points]
    ax.errorbar(steps, [p["mean_reward"] for p in points], yerr=[p["mean_reward_se"] for p in points],
                color=ORANGE, lw=2, marker="D", ms=5, capsize=2, elinewidth=1, label="gpt-oss-20b (training judge)")
    if regrades:
        ax.plot([r["step"] for r in regrades], [r["mean_reward"] for r in regrades], color=GREEN, lw=1.6,
                ls="--", marker="s", ms=5, label="gpt-oss-120b regrade")
    ax.set_title("Held-out eval: 64 problems × 4 attempts", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, 0.75)

    ax = axes[2]
    cut = [100 * v for v in curve["truncated_frac"]]
    ax.scatter(updates, cut, s=6, color=DOT, linewidths=0)
    ax.plot(updates, rolling(cut), color=BLUE, lw=2)
    ax.set_title("Proofs cut off at 32k tokens (%)", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, 50)

    for ax in axes:
        ax.set_xlim(0, max(500, last))
        ax.set_xlabel("update")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.axvline(100, color=MUTED, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=9, loc="lower right") if ax.get_legend_handles_labels()[0] else None
    first, final = points[0], points[-1]
    fig.text(0.055, 0.93, f"Qwen3.5-2B, RL from the base model on FineProofEnvs: held-out rubric score "
             f"{first['mean_reward']:.2f} → {final['mean_reward']:.2f} in {final['step']} updates",
             fontsize=14, color=INK, weight="bold")
    fig.text(0.055, 0.875, "Reward = gpt-oss-20b rubric score / 7; a proof cut off at the cap scores 0. "
             "Dotted line: the run resumed in a new HF Job at update 100.", fontsize=10.5, color=MUTED)
    fig.savefig(ROOT / "fp-rl2-2b-curves.svg")


if __name__ == "__main__":
    main()
