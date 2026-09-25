"""Build training diagnostics from extracted captures and optimizer-step telemetry."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = ["Harbor multi-harness", "Native OpenCode", "Harbor OpenCode-only"]
COLORS = ["#4f46e5", "#e85c41", "#059669"]


def summarize(frame, keys):
    result = frame.groupby(keys).agg(
        admitted_rollouts=("rollout_id", "size"),
        unique_tasks=("task_index", "nunique"),
        binary_reward_mean=("binary_reward", "mean"),
        completion_tokens_mean=("completion_tokens", "mean"),
        completion_tokens_median=("completion_tokens", "median"),
        completion_tokens_p90=("completion_tokens", lambda x: x.quantile(.9)),
        supervised_tokens=("supervised_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        text_only_turn_tokens=("completion_tokens_in_text_only_turns", "sum"),
        agent_turns_mean=("agent_turns", "mean"),
        tool_calls_mean=("emitted_tool_calls", "mean"),
        tool_calls_p90=("emitted_tool_calls", lambda x: x.quantile(.9)),
        exact_repeated_calls_mean=("exact_repeated_calls", "mean"),
        rollout_over4096_fraction=("has_response_over4096", "mean"),
        rollout_finish_length_fraction=("has_finish_length", "mean"),
        training_rows=("rows", "sum"),
        receipt_token_mismatches=("receipt_token_delta", lambda x: (x != 0).sum()),
    ).reset_index()
    result["text_only_turn_token_share"] = result.text_only_turn_tokens / result.completion_tokens
    result["rows_per_rollout"] = result.training_rows / result.admitted_rollouts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    captures = pd.read_csv(args.evidence / "admitted_training_behavior.csv")
    steps = pd.read_csv(args.evidence / "training_metrics.csv")
    for frame in (captures, steps):
        frame["window_end"] = ((frame.step - 1) // 100 + 1) * 100
    assert not captures.duplicated(["run", "rollout_id"]).any()
    assert len(steps) == 3000 and not steps.duplicated(["run", "step"]).any()
    assert np.isfinite(steps.grad_norm).all()
    captures["has_response_over4096"] = captures.responses_over_4096 > 0
    captures["has_finish_length"] = captures.responses_finish_length > 0
    # Admission receipts cover steps 31 onward for multi-harness, every step otherwise.
    for run, cohort in captures.groupby("run"):
        first = 31 if run == "Harbor multi-harness" else 1
        recorded = steps[(steps.run == run) & (steps.step >= first)]
        assert cohort.supervised_tokens.sum() == recorded["batch/trained_tokens_per_step"].sum()
    for name, keys in [
        ("training_rollout_totals", ["run"]),
        ("training_behavior_windows", ["run", "window_end"]),
        ("training_behavior_by_harness", ["run", "harness", "window_end"]),
        ("training_behavior_by_outcome", ["run", "window_end", "binary_reward"]),
    ]:
        summarize(captures, keys).to_csv(output / f"{name}.csv", index=False)
    windows = summarize(captures, ["run", "window_end"])
    steps["zero_gradient"] = steps.grad_norm == 0
    assert (steps.zero_gradient == (steps.reward_std == 0)).all()
    step_keys = ["reward", "reward_std", "tools/call_frequency", "tools/failure_frequency",
                 "rollout/turns_mean", "rollout/fork_frac", "rollout/samples_per_rollout",
                 "completions/mean_length", "completions/clipped_ratio", "zero_gradient",
                 "perf/step_s", "perf/fwd_bwd_s", "perf/rollout_wait_s"]
    telemetry = steps.groupby(["run", "window_end"])[step_keys].mean().reset_index()
    totals = steps.groupby(["run", "window_end"])[["batch/forwarded_tokens_per_step", "batch/trained_tokens_per_step"]].sum().reset_index()
    telemetry = telemetry.merge(totals, on=["run", "window_end"], validate="one_to_one")
    telemetry["forwarded_per_supervised_token"] = telemetry["batch/forwarded_tokens_per_step"] / telemetry["batch/trained_tokens_per_step"]
    telemetry.to_csv(output / "training_step_diagnostics.csv", index=False)
    accounting = []
    for run, group in steps.groupby("run"):
        forwarded = group["batch/forwarded_tokens_per_step"]
        supervised = group["batch/trained_tokens_per_step"]
        zero = group.zero_gradient
        accounting.append({
            "run": run, "steps": len(group), "zero_gradient_steps": int(zero.sum()),
            "forwarded_tokens": forwarded.sum(), "supervised_tokens": supervised.sum(),
            "forwarded_per_supervised_token": forwarded.sum() / supervised.sum(),
            "supervised_tokens_in_zero_gradient_steps": supervised[zero].sum(),
            "zero_gradient_supervised_token_fraction": supervised[zero].sum() / supervised.sum(),
            "zero_gradient_forwarded_token_fraction": forwarded[zero].sum() / forwarded.sum(),
            "mean_step_s": group["perf/step_s"].mean(),
            "step_timing_observations": group["perf/step_s"].count(),
            "receipt_token_mismatches": int((captures.loc[captures.run == run, "receipt_token_delta"] != 0).sum()),
        })
    pd.DataFrame(accounting).to_csv(output / "training_accounting.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for run, color in zip(RUNS, COLORS):
        w = windows[windows.run == run].sort_values("window_end")
        t = telemetry[telemetry.run == run].sort_values("window_end")
        options = dict(color=color, marker="o", markersize=4, label=run)
        axes[0, 0].plot(w.window_end, w.completion_tokens_mean, **options)
        axes[0, 1].plot(w.window_end, w.tool_calls_mean, **options)
        axes[0, 2].plot(w.window_end, 100 * w.text_only_turn_token_share, **options)
        axes[1, 0].plot(w.window_end, 100 * w.rollout_over4096_fraction, **options)
        axes[1, 1].plot(t.window_end, t.forwarded_per_supervised_token, **options)
        axes[1, 2].plot(t.window_end, 100 * t.zero_gradient, **options)
    labels = ["Mean completion tokens / admitted rollout", "Mean emitted tool calls / admitted rollout",
              "Completion tokens in text-only turns (%)", "Admitted rollouts with a response >4,096 tokens (%)",
              "Forwarded / supervised tokens", "Steps with zero fresh gradient (%)"]
    for ax, label in zip(axes.flat, labels):
        ax.set_title(label, fontsize=11, pad=10)
        ax.grid(alpha=.18)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("End of 100-step window")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.suptitle("Training behavior, token exposure and available learning signal", fontsize=19, y=.995)
    fig.text(.05, .025, "Captured turns from 13,625 admitted rollouts; multi-harness receipts start at step 31. Bottom-right panels use all 3,000 step records.\nTool calls are emitted requests, not confirmed executions. Text-only includes reasoning and final answers. Zero fresh gradient does not rule out optimizer momentum.", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .075, 1, .90))
    fig.savefig(output / "training_diagnostics.png", dpi=180)
    fig.savefig(output / "training_diagnostics.pdf")
    plt.close(fig)
    print(windows[(windows.window_end.isin([500, 700, 1000]))].to_string(index=False))
    print(pd.DataFrame(accounting).to_string(index=False))


if __name__ == "__main__":
    main()
