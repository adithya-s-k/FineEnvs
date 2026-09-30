"""Baseline, 100 training updates and checkpoint evaluation in one allocation."""
import argparse
import json
from pathlib import Path
import time

from recipe import digest, eval_rows, mean, task_rows, write_json
from runtime.checkpoints import make_ready


def pilot(args, cfg):
    from run import launch_local

    cfg = {**cfg, "max_steps": 100, "save_steps": 50, "eval_steps": 100, "eval_selection": "stratified"}
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    limit = args.limit if args.limit is not None else 25
    selected = eval_rows(task_rows("test"), limit, stratified=True, seed=cfg["seed"])
    identity = {"config": cfg, "eval_tasks": selected}
    signature = digest(identity)
    manifest = root / "pilot.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {
        "sha256": signature, **identity, "phases": {}}
    if state["sha256"] != signature:
        raise ValueError("Pilot output belongs to a different configuration or test selection")
    if args.resume and not state["phases"].get("baseline", {}).get("complete"):
        raise ValueError("Resume requires this pilot's completed baseline")

    for phase in ("baseline", "train", "checkpoint-100"):
        if state["phases"].get(phase, {}).get("complete"):
            continue
        entry = {"complete": False, "started_at": time.time()}
        state["phases"][phase] = entry
        write_json(manifest, state)
        current = argparse.Namespace(**vars(args))
        current.action = "train" if phase == "train" else "eval"
        current.limit = limit
        current.checkpoint = None
        current.resume = args.resume if phase == "train" else None
        current.smoke_eval = False
        phase_cfg = {**cfg, "output": str(root / phase), "run_name": cfg["run_name"] + "-" + phase}
        try:
            if current.resume:
                make_ready(current.resume)
            if phase == "checkpoint-100":
                current.checkpoint = root / "train/checkpoint-100"
                make_ready(current.checkpoint)
            launch_local(current, phase_cfg)
            if phase != "train":
                report = json.loads((Path(phase_cfg["output"]) / "eval/summary.json").read_text())
                if not report["complete"]:
                    raise ValueError("Pilot requires complete evaluation coverage")
                entry["evaluation"] = report
            else:
                marker = root / "train/checkpoint-100/checkpoint.saved.json"
                if not marker.exists():
                    raise ValueError("Training ended before checkpoint 100")
            entry.update(complete=True, finished_at=time.time())
            write_json(manifest, state)
        except BaseException as exc:
            entry.update(error=type(exc).__name__, finished_at=time.time())
            write_json(manifest, state)
            raise
    write_json(root / "comparison.json", comparison(root, state))


def comparison(root, state):
    before = state["phases"]["baseline"]["evaluation"]
    after = state["phases"]["checkpoint-100"]["evaluation"]
    metrics = [json.loads(line) for line in (root / "train/metrics.jsonl").read_text().splitlines()]
    committed = {row["step"]: row for row in metrics if "grad_norm" in row}
    updates = [committed[step] for step in sorted(committed)]
    result = {"baseline": before, "checkpoint_100": after,
              "pass_at_1_change_pp": 100 * (after["pass_at_1"] - before["pass_at_1"]),
              "updates_with_gradient_metrics": len(updates),
              "nonzero_gradient_updates": sum(row["grad_norm"] > 0 for row in updates),
              "zero_gradient_updates": sum(row["grad_norm"] == 0 for row in updates),
              "note": "Mean tool usage includes different success sets; it is not matched-success efficiency."}
    result["harnesses"] = {h: {"baseline": before["harnesses"][h], "checkpoint_100": values,
        "pass_at_1_change_pp": 100 * (values["pass_at_1"] - before["harnesses"][h]["pass_at_1"])}
        for h, values in after["harnesses"].items()}
    for label, rows in (("first_20_updates", updates[:20]), ("last_20_updates", updates[-20:])):
        result[label] = {"count": len(rows), "mean_reward": mean(row.get("reward") for row in rows),
                         "mean_grad_norm": mean(row["grad_norm"] for row in rows)}
    base_pairs = {p.name: json.loads(p.read_text()) for p in (root / "baseline/eval/pairs").glob("*.json")}
    matches = []
    for path in (root / "checkpoint-100/eval/pairs").glob("*.json"):
        a, b = base_pairs[path.name], json.loads(path.read_text())
        if a.get("correctness") == b.get("correctness") == 1 and not a.get("error") and not b.get("error"):
            matches.append((a, b))
    result["matched_successes"] = {}
    for harness in ["all", *after["harnesses"]]:
        pairs = [(a, b) for a, b in matches if harness == "all" or a["harness"] == harness]
        values = {"pairs": len(pairs)}
        for key in ("tool_calls", "generated_tokens"):
            valid = [(a[key], b[key]) for a, b in pairs if a.get(key) is not None and b.get(key) is not None]
            denominator = sum(a for a, _ in valid)
            values[key + "_pairs"] = len(valid)
            values[key + "_savings_pct"] = 100 * (1 - sum(b for _, b in valid) / denominator) if denominator else None
        result["matched_successes"][harness] = values
    return result
