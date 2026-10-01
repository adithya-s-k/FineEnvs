"""Rebuild the article comparison from its pinned data excerpt."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "historical-lfm-curves.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "text.color": "#172033",
            "axes.labelcolor": "#475569",
            "svg.fonttype": "none",
            "svg.hashsalt": "smoldataenvs-history",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.69, bottom=0.26, wspace=0.30)
    fig.text(
        0.065,
        0.94,
        "LFM2.5-2.6B: earlier multi-harness RL results",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.065,
        0.885,
        "Historical article runs | 250 test tasks x 4 harnesses | Pass@1",
        color="#475569",
    )
    for (key, run), color in zip(data["runs"].items(), ["#2563eb", "#d97706"]):
        steps, rewards = run["train"]["step"], run["train"]["reward"]
        smooth = [
            sum(rewards[max(0, i - 49) : i + 1]) / min(i + 1, 50)
            for i in range(len(rewards))
        ]
        axes[0].plot(steps, rewards, color=color, alpha=0.08, linewidth=0.6)
        axes[0].plot(steps, smooth, color=color, linewidth=2, label=run["label"])
        points = run["eval"]
        axes[1].plot(
            [p["step"] for p in points],
            [p["overall"] for p in points],
            color=color,
            linewidth=2,
        )
        for p in points:
            axes[1].plot(
                p["step"],
                p["overall"],
                "o",
                color=color,
                markerfacecolor=color if p["complete"] else "white",
                markersize=5,
            )
        p = points[-1]
        axes[1].annotate(
            f"{p['overall']:.1f}%",
            (p["step"], p["overall"]),
            xytext=(0, 10 if key.endswith("multi") else -18),
            textcoords="offset points",
            ha="right",
            color=color,
            weight="bold",
        )
        savings = run["efficiency"]
        axes[2].plot(
            [p["step"] for p in savings],
            [p["tool_savings"] for p in savings],
            color=color,
            linewidth=2,
        )
        coverage = {p["step"]: p["complete"] for p in points}
        for p in savings:
            axes[2].plot(
                p["step"],
                p["tool_savings"],
                "o",
                color=color,
                markerfacecolor=color if coverage[p["step"]] else "white",
                markersize=5,
            )
        p = savings[-1]
        axes[2].annotate(
            f"{p['tool_savings']:.1f}%",
            (p["step"], p["tool_savings"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            color=color,
            weight="bold",
        )
    for ax, title, ylabel in zip(
        axes,
        ["Training reward", "Held-out answer quality", "Tool-call savings"],
        [
            "Correctness + efficiency bonus",
            "Pass@1 on graded pairs (%)",
            "Savings on matched successes (%)",
        ],
    ):
        ax.set_title(title, loc="left", weight="bold", pad=14)
        ax.set_xlabel("Optimizer step / checkpoint")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_xlim(-20, 1030)
    axes[0].set_ylim(0, 1.15)
    axes[1].set_ylim(35, 60)
    axes[2].set_ylim(-2, 36)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.06, 0.845),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.065,
        0.15,
        "Training: trailing 50-update mean; faint lines are raw rewards. Evaluation: measured checkpoints, no smoothing.",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.105,
        "Hollow markers: incomplete coverage. Savings use pairs solved by baseline and checkpoint. Both runs used Harbor.",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.06,
        "Source: multi-harness RL article, pinned snapshot a1e03a6. These are not the new non-thinking pilot results.",
        fontsize=10,
        color="#64748b",
    )
    output = root / "historical-lfm-curves.svg"
    fig.savefig(output, metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
