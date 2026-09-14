"""Build reproducible comparison views from locally downloaded Trackio archives.

Only derived databases are written. Source archives and their recorded values
remain unchanged. See README.md for selection and evaluation-mapping rules.
"""

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import uuid
from collections import defaultdict
from pathlib import Path

MODELS = {
    "qwen3-vl-2b": "Qwen3-VL-2B",
    "qwen3.5-2b": "Qwen3.5-2B",
    "glm-ocr": "GLM-OCR",
    "gemma4-e2b": "Gemma4-E2B",
}
SAFE_CONFIG = {
    "_name_or_path",
    "model_type",
    "transformers_version",
    "max_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "optim",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "bf16",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "save_steps",
    "num_generations",
    "temperature",
    "top_p",
    "top_k",
    "beta",
    "epsilon",
    "loss_type",
    "seed",
    "max_completion_length",
    "use_vllm",
    "vllm_mode",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_length",
    "vllm_tensor_parallel_size",
    "mask_truncated_completions",
    "model/num_parameters",
    "scale_rewards",
    "vllm_importance_sampling_mode",
    "vllm_importance_sampling_clip_max",
}


def numeric_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if type(value) in (int, float) and math.isfinite(value)
    }


def training_segments(rows):
    """Internal log order survives asynchronous uploads; timestamps may not."""
    segments = []
    for row in sorted(rows, key=lambda r: (r["step"], r["id"])):
        metrics = row["metrics"]
        if "train/reward" not in metrics:
            continue
        step = metrics.get("train/global_step")
        if type(step) not in (int, float) or step != int(step):
            raise ValueError("Training reward has no integral optimizer step")
        if not segments or step < segments[-1][-1]["metrics"]["train/global_step"]:
            segments.append([])
        segments[-1].append(row)
    return segments


def merge_scalars(training, evaluation):
    """Align by optimizer step, retaining values and rejecting conflicting logs."""
    merged = {}
    for rows, is_eval in [(training, False), (evaluation, True)]:
        for row in rows:
            step = int(row["step"] if is_eval else row["metrics"]["train/global_step"])
            dest = merged.setdefault(
                step, {"timestamp": row["timestamp"], "metrics": {}}
            )
            values = numeric_metrics(row["metrics"])
            if is_eval:
                values = {k: v for k, v in values.items() if k.startswith("eval/")}
            for key, value in values.items():
                if key in dest["metrics"] and dest["metrics"][key] != value:
                    raise ValueError(f"Conflicting {key} at optimizer step {step}")
                dest["metrics"][key] = value
    baseline = merged.get(0, {}).get("metrics", {}).get("eval/test_reward")
    if baseline is not None:
        for row in merged.values():
            reward = row["metrics"].get("eval/test_reward")
            if reward is not None:
                row["metrics"]["eval/reward_gain"] = reward - baseline
    return merged


def create_database(path, template):
    conn = sqlite3.connect(path)
    for name, sql in template.execute(
        "SELECT name,sql FROM sqlite_master WHERE type IN ('table','index') "
        "AND sql IS NOT NULL ORDER BY type DESC"
    ):
        if not name.startswith("sqlite_"):
            conn.execute(sql)
    return conn


def selected_view(project, name, count):
    if project == "latex-ocr-colocate-5k":
        return "latex-ocr-comparison", (
            "unstable" if name == "gemma4-e2b" else "colocated"
        )
    if project == "latex-ocr-colocate-5k-stable":
        return "latex-ocr-comparison", "stabilized"
    if count >= 100 and project in {
        "latex-ocr-ablation-scale",
        "latex-ocr-grpo-redhat",
        "latex-ocr-grpo-demo",
    }:
        return "latex-ocr-history", {
            "latex-ocr-ablation-scale": "scale",
            "latex-ocr-grpo-redhat": "Red Hat run",
            "latex-ocr-grpo-demo": "200-step demo",
        }[project]
    return None


def build(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    databases = {}
    report = {"sources": [], "runs": [], "excluded": []}
    for path in sorted(source_dir.glob("*/*.db")):
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report["sources"].append(
            {
                "bucket": f"AdithyaSK/{path.parent.name}",
                "project": path.stem,
                "sha256": digest,
            }
        )
        for config_row in source.execute("SELECT * FROM configs ORDER BY id"):
            original_config = json.loads(config_row["config"])
            rows = [
                {**dict(row), "metrics": json.loads(row["metrics"])}
                for row in source.execute(
                    "SELECT * FROM metrics WHERE run_id=?", (config_row["run_id"],)
                )
            ]
            segments = training_segments(rows)
            for segment_index, training in enumerate(segments, 1):
                selection = selected_view(
                    path.stem, config_row["run_name"], len(training)
                )
                if selection is None:
                    report["excluded"].append(
                        {
                            "project": path.stem,
                            "run": config_row["run_name"],
                            "attempt": segment_index,
                            "records": len(training),
                            "reason": "Preserved overnight view"
                            if "overnight" in path.stem
                            else "Short preflight or restart (fewer than 100 training records)",
                        }
                    )
                    continue
                project, experiment = selection
                name = config_row["run_name"]
                model = MODELS.get(
                    name,
                    {"qwen3_vl": "Qwen3-VL-2B", "qwen3_5": "Qwen3.5-2B"}.get(
                        original_config.get("model_type"), name
                    ),
                )
                display_name = f"{model} / {experiment}"
                if len(segments) > 1:
                    display_name += f" / attempt {segment_index}"
                identity = f"{path.parent.name}/{path.stem}/{config_row['run_id']}/{segment_index}"
                run_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
                # Reused run IDs contain evaluations whose attempt is ambiguous.
                # Keep their training curves separate; do not guess an eval pairing.
                evaluation = (
                    [r for r in rows if "eval/test_reward" in r["metrics"]]
                    if len(segments) == 1
                    else []
                )
                merged = merge_scalars(training, evaluation)
                if project not in databases:
                    temp = output_dir / f"{project}.building"
                    temp.unlink(missing_ok=True)
                    databases[project] = (create_database(temp, source), temp)
                dest = databases[project][0]
                last_step = max(
                    int(r["metrics"]["train/global_step"]) for r in training
                )
                config = {k: v for k, v in original_config.items() if k in SAFE_CONFIG}
                config.update(
                    {
                        "model": model,
                        "experiment": experiment,
                        "_Group": model,
                        "_Created": training[0]["timestamp"],
                        "source_bucket": f"AdithyaSK/{path.parent.name}",
                        "source_project": path.stem,
                        "source_run_id": config_row["run_id"],
                        "source_sha256": digest,
                        "attempt": segment_index,
                        "observed_optimizer_steps": last_step,
                        "evaluation_sample_count": "not recorded in source",
                        "evaluation_mapping": "unambiguous single attempt"
                        if len(segments) == 1
                        else "omitted: source run ID reused across attempts",
                        "status": "recorded snapshot; no completion status stored",
                    }
                )
                dest.execute(
                    "INSERT INTO configs(run_id,run_name,config,created_at) VALUES (?,?,?,?)",
                    (
                        run_id,
                        display_name,
                        json.dumps(config),
                        training[0]["timestamp"],
                    ),
                )
                for step, row in sorted(merged.items()):
                    dest.execute(
                        "INSERT INTO metrics(run_id,run_name,timestamp,step,metrics) VALUES (?,?,?,?,?)",
                        (
                            run_id,
                            display_name,
                            row["timestamp"],
                            step,
                            json.dumps(row["metrics"]),
                        ),
                    )
                if len(segments) == 1:
                    for row in source.execute(
                        "SELECT timestamp,metrics FROM system_metrics WHERE run_id=?",
                        (config_row["run_id"],),
                    ):
                        dest.execute(
                            "INSERT INTO system_metrics(run_id,run_name,timestamp,metrics) VALUES (?,?,?,?)",
                            (run_id, display_name, row["timestamp"], row["metrics"]),
                        )
                window = min(50, len(training) // 2) or 1
                rewards = [r["metrics"]["train/reward"] for r in training]
                eval_points = [
                    {"step": step, "reward": row["metrics"]["eval/test_reward"]}
                    for step, row in sorted(merged.items())
                    if "eval/test_reward" in row["metrics"]
                ]
                report["runs"].append(
                    {
                        "project": project,
                        "run_id": run_id,
                        "name": display_name,
                        "model": model,
                        "source_project": path.stem,
                        "source_run_id": config_row["run_id"],
                        "attempt": segment_index,
                        "training_records": len(training),
                        "last_optimizer_step": last_step,
                        "planned_steps": config.get("max_steps"),
                        "window": window,
                        "first_window_reward": statistics.mean(rewards[:window]),
                        "last_window_reward": statistics.mean(rewards[-window:]),
                        "evaluation": eval_points,
                        "learning_rate": config.get("learning_rate"),
                        "beta": config.get("beta"),
                        "recorded_config": {
                            k: v for k, v in original_config.items() if k in SAFE_CONFIG
                        },
                        "last_window_clipped_ratio": statistics.mean(
                            r["metrics"].get("train/completions/clipped_ratio", 0)
                            for r in training[-window:]
                        ),
                    }
                )
        source.close()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for project, (conn, temp) in databases.items():
        conn.commit()
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        conn.close()
        temp.replace(output_dir / f"{project}.db")
    (output_dir / "comparison-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.source_dir, args.output_dir)
    counts = defaultdict(int)
    for run in summary["runs"]:
        counts[run["project"]] += 1
    print(json.dumps(dict(counts)))
