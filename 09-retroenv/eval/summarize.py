#!/usr/bin/env python3
"""Turn run_eval.py output directories into one results table.

    uv run python eval/summarize.py runs/test_id/* --output runs/test_id/results

Writes RESULTS.md (the board plus breakdowns) and board.json (every summary),
sorted by pass@1 and then mean reward. Transcripts stay in the run directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for run in args.runs:
        path = run / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            summary["identity"] = json.loads((run / "identity.json").read_text())
            summaries.append(summary)
    if not summaries:
        parser.error("no summary.json found")
    summaries.sort(key=lambda s: (-(s["pass_at_1"] or 0), -(s["mean_reward"] or 0)))

    header = (
        "| Model | Tasks | Pass@1 (95% CI) | Exact route | Reward | Steps | Stock | Graph | "
        "Tool calls | No emit | Refused | Cost |"
    )
    lines = [header, "|" + "---|" * 12]
    for s in summaries:
        c = s["components"]
        ci = s["pass_at_1_ci95"]
        cost = "—" if s["cost_usd"] is None else f"${s['cost_usd']:.2f}"
        lines.append(
            f"| {s['label']} | {s['tasks']}/{s['episodes_expected']} | {pct(s['pass_at_1'])} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | {pct(s['exact_route_rate'])} | {pct(s['mean_reward'])} | "
            f"{pct(c.get('step_correctness'))} | {pct(c.get('stock_correctness'))} | {pct(c.get('graph_validity'))} | "
            f"{s['mean_tool_calls']:.1f} | {pct(s['no_emit_rate'])} | {pct(s.get('refusal_rate'))} | {cost} |"
        )

    def section(title: str, key: str) -> list[str]:
        groups = sorted({name for s in summaries for name in (s.get(key) or {})})
        if not groups:
            return []
        out = [
            f"\n### {title} (exact route rate)\n",
            "| Model | " + " | ".join(groups) + " |",
            "|---|" + "---|" * len(groups),
        ]
        for s in summaries:
            cells = []
            for name in groups:
                row = (s.get(key) or {}).get(name)
                cells.append("—" if not row else f"{pct(row['exact_route_rate'])} (n={row['tasks']})")
            out.append(f"| {s['label']} | " + " | ".join(cells) + " |")
        return out

    first = summaries[0]["identity"]
    body = [
        "# RetroEnv v2 eval board",
        "",
        f"Split `{first['split']}`, {len(first['task_ids'])} tasks, {first['attempts']} attempt(s), "
        f"{first['max_turns']} model turns, toolset `{first['toolset']}`, tool budget {first['max_tool_calls']}. "
        "Every tool call and reward came from the OpenEnv server. Pass needs every required route verified "
        "against a hidden patent route; exact route is the share of tasks where a submitted route matches one "
        "step for step.",
        "",
        *lines,
        *section("By required routes", "by_kind"),
        *section("By step cap", "by_depth"),
        *section("By heuristic tier", "by_tier"),
        "",
        "## Reading the table",
        "",
        "**No emit** is the share of episodes the model never closed with `emit_routes`; the harness "
        "then submits an empty route set, which scores the 0.05 floor. **Refused** is the share the "
        "provider declined on safety grounds, which counts as a failed episode and is never re-routed "
        "to another model. Where the two columns match, every unclosed episode was a refusal, and the "
        "model's reward is held down by requests it did not answer rather than by its chemistry.",
        "",
        "**Steps**, **Stock** and **Graph** are the step-correctness, stock-correctness and "
        "graph-validity reward components. A model can score well on stock and graph while failing the "
        "task: those measure that the submitted tree is well formed and its leaf claims are truthful, "
        "not that the route matches the patent.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    (args.output / "board.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
