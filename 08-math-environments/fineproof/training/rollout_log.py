"""One record per rollout, and the Trackio metrics, traces and alerts made from them.

OpenEnv returns a rich result for every rollout (tokens per call, finish reasons, phase timings, findings, errors) and
the verifier leaves the judge's grading in the trial folder. `append` writes both as one line of `rollouts.jsonl` next
to the trials folder. During training, `make_callbacks` turns the lines written since the previous update into:

- metrics (`train/rollouts/*`, `train/judge/*`), merged into the trainer's own log so they share its Trackio step;
- one Trackio trace per graded rollout: the conversation, plus spans for the sandbox, each policy call and each
  judge grade (its model, tokens, cost and written assessment);
- Trackio alerts when many rollouts go ungraded, the judge fails, or proofs hit the output cap.

Everything here is best-effort: logging never stops a rollout or a training step.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return None


def record(result, trial_root: Path | None) -> dict:
    """A JSON-ready summary of one OpenEnv HarborRolloutResult, with the judge's grading when the trial has one."""
    rec = {
        "time": round(time.time(), 1),
        "trial": result.trial_name,
        "task": result.task_name,
        "reward": result.reward,
        "ok": result.ok,
        "error": (result.error or "")[:300] or None,
        "exception": result.exception_type,
        "rollout_type": result.rollout_type,
        "capture": result.capture_level,
        "turns": result.n_turns,
        "roots": result.n_roots,
        "trainable_tokens": result.n_trainable_tokens,
        "wall_s": round(result.wall_s or 0.0, 1),
        "phases": {k: round(v, 1) for k, v in (result.phase_timings or {}).items() if isinstance(v, (int, float))},
        "findings": [str(f)[:160] for f in (result.findings or [])],
        "budget_stops": getattr(result, "budget_stop_count", 0),
        "param_fixes": list(getattr(result, "param_fixes", None) or []),
        "calls": [{"finish": t.finish_reason, "prompt_tokens": len(t.prompt_token_ids or []),
                   "completion_tokens": len(t.completion_token_ids or []),
                   "trainable": getattr(t, "trainable", True) and not getattr(t, "discarded", False)}
                  for t in (result.turns or [])],
    }
    if not result.ok and getattr(result, "agent_log_tail", ""):
        rec["agent_log_tail"] = result.agent_log_tail[-600:]
    grading = _read_json(Path(trial_root) / result.trial_name / "verifier" / "grading.json") \
        if trial_root and result.trial_name else None
    if grading:
        usage = [u for u in grading.get("usage") or [] if isinstance(u, dict)]
        started, finished = grading.get("started_at"), grading.get("finished_at")
        rec["judge"] = {
            "models": [g.get("model") for g in grading.get("grades") or []],
            "scores": grading.get("scores"),
            "reason": (grading.get("reason") or "")[:200],
            "proof_chars": grading.get("proof_chars"),
            "seconds": round(finished - started, 1)
            if isinstance(started, (int, float)) and isinstance(finished, (int, float)) else None,
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 6),
        }
    return rec


def records_from_trials(trial_root: Path) -> list[dict]:
    """Records rebuilt from Harbor's own trial files, for runs made before rollouts.jsonl existed. Per-call detail is
    not in those files, so each rollout counts as one call with the agent's total tokens."""
    records = []
    for folder in sorted(Path(trial_root).iterdir()) if Path(trial_root).is_dir() else []:
        result = _read_json(folder / "result.json")
        if not isinstance(result, dict):
            continue
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        agent = result.get("agent_result") or {}
        exception = (result.get("exception_info") or {}).get("exception_type")
        grading = _read_json(folder / "verifier" / "grading.json") or {}
        usage = [u for u in grading.get("usage") or [] if isinstance(u, dict)]
        rec = {"trial": folder.name, "task": result.get("task_name"), "reward": rewards.get("reward"),
               "ok": exception is None, "exception": exception,
               "calls": [{"finish": None, "prompt_tokens": agent.get("n_input_tokens") or 0,
                          "completion_tokens": agent.get("n_output_tokens") or 0}] if agent else []}
        if grading:
            rec["judge"] = {"models": [g.get("model") for g in grading.get("grades") or []],
                            "scores": grading.get("scores"), "reason": grading.get("reason"),
                            "proof_chars": grading.get("proof_chars"),
                            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                            "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 6)}
        records.append(rec)
    return records


def append(path: Path, result, trial_root: Path | None) -> None:
    try:
        line = json.dumps(record(result, trial_root), default=str)
        with _LOCK, Path(path).open("a") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - a log line must never fail a rollout
        logger.warning("could not log a rollout", exc_info=True)


def summarize(records: list[dict]) -> dict:
    """Metrics over a batch of rollout records (the ones finished since the previous update)."""
    if not records:
        return {}
    mean = (lambda xs: statistics.fmean(xs) if xs else None)
    graded = [r["reward"] for r in records if isinstance(r.get("reward"), (int, float))]
    calls = [c for r in records for c in r.get("calls") or []]
    judged = [r["judge"] for r in records if r.get("judge")]
    providers = Counter((m or "none").rsplit(":", 1)[-1] for j in judged for m in j.get("models") or [None])
    spreads = [max(s) - min(s) for j in judged if len(s := [x for x in j.get("scores") or []
                                                            if isinstance(x, (int, float))]) > 1]
    out = {
        "rollouts/count": len(records),
        "rollouts/graded_frac": len(graded) / len(records),
        "rollouts/reward_mean": mean(graded),
        "rollouts/full_marks_frac": mean([float(x >= 0.999) for x in graded]),
        "rollouts/zero_frac": mean([float(x <= 0.0) for x in graded]),
        "rollouts/server_errors": sum(not r.get("ok") for r in records),
        "rollouts/agent_timeouts": sum(r.get("exception") == "AgentTimeoutError" for r in records),
        "rollouts/wall_s_mean": mean([r["wall_s"] for r in records if r.get("wall_s")]),
        "rollouts/wall_s_max": max((r["wall_s"] for r in records if r.get("wall_s")), default=None),
        "rollouts/calls_mean": mean([float(len(r.get("calls") or [])) for r in records]),
        "rollouts/completion_tokens_mean": mean([c["completion_tokens"] for c in calls]),
        "rollouts/completion_tokens_max": max((c["completion_tokens"] for c in calls), default=None),
        "rollouts/prompt_tokens_mean": mean([c["prompt_tokens"] for c in calls]),
        "rollouts/truncated_frac": mean([float(c["finish"] == "length") for c in calls if c.get("finish")]),
        "rollouts/findings": sum(len(r.get("findings") or []) for r in records),
        "rollouts/budget_stops": sum(r.get("budget_stops") or 0 for r in records),
        "rollouts/untrainable_calls": sum(not c.get("trainable", True) for c in calls),
        "judge/failures": sum("grading error" in (j.get("reason") or "") for j in judged),
        "judge/proof_chars_mean": mean([j["proof_chars"] for j in judged if j.get("proof_chars")]),
        "judge/seconds_mean": mean([j["seconds"] for j in judged if j.get("seconds")]),
        "judge/score_spread_mean": mean(spreads),
        "judge/prompt_tokens_sum": sum(j.get("prompt_tokens") or 0 for j in judged),
        "judge/completion_tokens_sum": sum(j.get("completion_tokens") or 0 for j in judged),
        "judge/cost_usd_sum": sum(j.get("cost_usd") or 0 for j in judged),
    }
    for key in sorted({k for r in records for k in r.get("phases") or {}}):
        out[f"rollouts/phase_{key}_s"] = mean([r["phases"][key] for r in records if key in (r.get("phases") or {})])
    total = sum(providers.values())
    for provider, n in providers.items():
        out[f"judge/{provider.replace('/', '_')}_frac"] = n / total
    return {k: v for k, v in out.items() if v is not None}


def read_new(path: Path, offset: int) -> tuple[list[dict], int]:
    """Complete lines appended since `offset`, and the new offset."""
    path = Path(path)
    if not path.is_file():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    end = chunk.rfind(b"\n") + 1
    records = []
    for line in chunk[:end].decode(errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, offset + end


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if v is not None}
    return value


def trace(rec: dict, trial_root: Path, model: str | None = None, kind: str = "train"):
    """A trackio.Trace of one rollout: the agent's conversation, and spans for the sandbox, policy calls and judge."""
    import trackio

    folder = Path(trial_root) / str(rec.get("trial"))
    messages = [m for m in _read_json(folder / "agent" / "conversation.json") or [] if isinstance(m, dict)]
    result = _read_json(folder / "result.json") or {}
    grading = _read_json(folder / "verifier" / "grading.json") or {}
    ok = bool(rec.get("ok")) and rec.get("exception") is None
    spans = [{"id": "rollout", "name": "rollout", "kind": "span", "start_time": result.get("started_at"),
              "end_time": result.get("finished_at"), "status": "success" if ok else "error",
              "error": rec.get("error") or rec.get("exception"),
              "duration_ms": None if result.get("started_at") else (rec.get("wall_s") or 0) * 1000}]
    for span_id, phase in (("sandbox", "environment_setup"), ("agent", "agent_execution"), ("verifier", "verifier")):
        timing = result.get(phase) or {}
        if timing.get("started_at"):
            spans.append({"id": span_id, "parent_id": "rollout", "name": phase, "kind": "span",
                          "start_time": timing["started_at"], "end_time": timing.get("finished_at")})
    parent = "agent" if any(s["id"] == "agent" for s in spans) else "rollout"
    for i, call in enumerate(rec.get("calls") or []):
        spans.append({"id": f"policy-{i}", "parent_id": parent, "name": "policy call", "kind": "generation",
                      "model": model, "usage": {"input_tokens": call.get("prompt_tokens"),
                                                "output_tokens": call.get("completion_tokens")},
                      "status": "error" if call.get("finish") == "length" else "success",
                      "metadata": {"finish_reason": call.get("finish")}})
    parent = "verifier" if any(s["id"] == "verifier" for s in spans) else "rollout"
    usage = [u if isinstance(u, dict) else {} for u in grading.get("usage") or []]
    for i, grade in enumerate(grading.get("grades") or []):
        used = usage[i] if i < len(usage) else {}
        spans.append({"id": f"judge-{i}", "parent_id": parent, "name": "judge", "kind": "generation",
                      "model": grade.get("model"), "output": {"text": grade.get("assessment")},
                      "usage": {"input_tokens": used.get("prompt_tokens"), "output_tokens": used.get("completion_tokens")},
                      "cost_usd": used.get("cost"), "metadata": {"score": grade.get("score")}})
    metadata = {
        "kind": kind, "task": rec.get("task"), "trial": rec.get("trial"), "reward": rec.get("reward"),
        "scores": grading.get("scores"), "reason": grading.get("reason"), "proof_chars": grading.get("proof_chars"),
        "exception": rec.get("exception"), "findings": rec.get("findings") or None, "wall_s": rec.get("wall_s"),
        "status": "success" if ok else "error",
    }
    return trackio.Trace(messages=messages, spans=[_strip(s) for s in spans], metadata=_strip(metadata))


def log_traces(log, key: str, records: list[dict], trial_root: Path, step: int, model: str | None = None,
               kind: str = "train", chunk: int = 64) -> int:
    """Log one trace per record with `log` (trackio.log or a Run's log), `chunk` per call. Returns how many."""
    traces = []
    for rec in records:
        try:
            traces.append(trace(rec, trial_root, model, kind))
        except Exception:  # noqa: BLE001
            logger.warning("could not build a trace for %s", rec.get("trial"), exc_info=True)
    for start in range(0, len(traces), chunk):
        log({key: traces[start:start + chunk]}, step=step)
    return len(traces)


def alerts(metrics: dict) -> list[tuple[str, str]]:
    """(title, text) for each sign of trouble in one update's rollout metrics."""
    out = []
    count = metrics.get("rollouts/count", 0)
    if count >= 4 and metrics.get("rollouts/graded_frac", 1) < 0.75:
        out.append(("Many ungraded rollouts", f"{metrics['rollouts/graded_frac']:.0%} of {count} rollouts got a grade "
                    f"({metrics.get('rollouts/server_errors', 0)} server errors, "
                    f"{metrics.get('rollouts/agent_timeouts', 0)} agent timeouts, "
                    f"{metrics.get('judge/failures', 0)} judge failures)"))
    elif metrics.get("judge/failures", 0):
        out.append(("Judge failures", f"{metrics['judge/failures']} of {count} rollouts could not be graded"))
    if metrics.get("rollouts/truncated_frac", 0) > 0.2:
        out.append(("Proofs hit the output cap", f"{metrics['rollouts/truncated_frac']:.0%} of model calls stopped at "
                    "the token cap"))
    return out


def make_callbacks(path: Path, trial_root: Path, model: str | None = None):
    """Two TrainerCallbacks. Insert the first before Trackio's (it adds rollout metrics to the trainer's log) and
    append the second after it (it logs the traces and alerts once the trainer's own row is written)."""
    from transformers import TrainerCallback

    # Start at the end of the file: a resumed run must not count the rollouts of the run it resumes.
    shared = {"offset": Path(path).stat().st_size if Path(path).is_file() else 0, "records": [], "metrics": {}}

    class RolloutMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or logs is None:
                return
            try:
                records, shared["offset"] = read_new(path, shared["offset"])
                shared["records"] += records
                shared["metrics"] = summarize(records)
                logs.update(shared["metrics"])
            except Exception:  # noqa: BLE001 - logging must never stop training
                logger.warning("could not read rollout records", exc_info=True)

    class RolloutTraces(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            records, metrics = shared["records"], shared["metrics"]
            shared["records"], shared["metrics"] = [], {}
            if not state.is_world_process_zero or not records or "trackio" not in (args.report_to or []):
                return
            try:
                import trackio

                log_traces(trackio.log, "graded_rollouts", records, trial_root, state.global_step, model)
                for title, text in alerts(metrics):
                    trackio.alert(title=title, text=text)  # Trackio stamps the step
            except Exception:  # noqa: BLE001
                logger.warning("could not log rollout traces", exc_info=True)

    return RolloutMetrics(), RolloutTraces()
