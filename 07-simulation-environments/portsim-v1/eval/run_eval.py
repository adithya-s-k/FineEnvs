"""Run models on the berth-planning tasks through the env server and save every rollout for the viewer.

    cd 07-simulation-environments/portsim-v1
    envs/berth_planning/openenv/.venv/bin/python eval/run_eval.py --run 2026-10-05-board \
        --models anthropic:claude-sonnet-5-5 openai:gpt-6-astra@low hf:Qwen/Qwen3.8-27B:cerebras \
        --split all --concurrency 6

Starts a server in this process unless --server is given. Rollouts go to results/rollouts/<run>/<model>/<task>.json
with an index.json per run; existing episodes are skipped (resume), --force reruns them. Keys come from the
environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envs" / "berth_planning" / "openenv"))

from berth_core import load_pack  # noqa: E402
from berth_openenv.agent import run_episode  # noqa: E402
from berth_openenv.api import model_slug  # noqa: E402

_LOCK = threading.Lock()


def start_server() -> str:
    import uvicorn

    from berth_openenv.server import create_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(create_server(), host="127.0.0.1", port=port, log_level="warning",
                            ws_ping_interval=60, ws_ping_timeout=600)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def summarize(record: dict, task) -> dict:
    g = (record.get("final") or {}).get("grade") or {}
    return {"model": record["model"], "task_id": record["task_id"], "split": task.split, "difficulty": task.difficulty,
            "quay": task.quay, "ships": len(task.ships), "reward": record["reward"],
            "submitted": bool(record["final"]["submitted"]), "feasible": bool(g.get("feasible")),
            "cost": g.get("cost"), "optimal_cost": task.reference["optimal_cost"],
            "naive_cost": task.reference["naive_cost"], "turns": record["turns"],
            "checks": sum(1 for s in record["steps"] if s["tool"] == "check_plan"), "seconds": record["seconds"],
            "input_tokens": record["usage"]["input_tokens"], "output_tokens": record["usage"]["output_tokens"],
            "end_reason": record["end_reason"], "errors": len(record.get("errors") or [])}


def write_index(run_dir: Path, rows: dict):
    with _LOCK:
        body = {"run": run_dir.name, "episodes": sorted(rows.values(), key=lambda r: (r["model"], r["task_id"]))}
        tmp = run_dir / "index.json.tmp"
        tmp.write_text(json.dumps(body, indent=1))
        tmp.replace(run_dir / "index.json")


def board(rows: list[dict]) -> str:
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    lines = ["| Model | n | mean reward | feasible | submitted | easy | medium | hard | beat naive | optimal |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, rs in sorted(by.items(), key=lambda kv: -statistics.mean(r["reward"] for r in kv[1])):
        def mean(d):
            xs = [r["reward"] for r in rs if r["difficulty"] == d]
            return f"{statistics.mean(xs):.3f}" if xs else "-"
        beat = sum(1 for r in rs if r["feasible"] and r["cost"] is not None and r["cost"] < r["naive_cost"])
        opt = sum(1 for r in rs if r["feasible"] and r["cost"] == r["optimal_cost"])
        lines.append(f"| {m} | {len(rs)} | {statistics.mean(r['reward'] for r in rs):.3f} | "
                     f"{100 * sum(r['feasible'] for r in rs) / len(rs):.0f}% | {100 * sum(r['submitted'] for r in rs) / len(rs):.0f}% | "
                     f"{mean('easy')} | {mean('medium')} | {mean('hard')} | {beat} | {opt} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "test", "eval"])
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the split")
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit task ids")
    ap.add_argument("--server", default=None)
    ap.add_argument("--concurrency", type=int, default=6, help="episodes in flight per model")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--wall-clock", type=float, default=7200.0, help="seconds per episode before it is cut off")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pack = load_pack()
    tasks = [t for t in pack.tasks if args.split == "all" or t.split == args.split]
    if args.tasks:
        tasks = [pack.get(t) for t in args.tasks]
    if args.limit:
        tasks = tasks[: args.limit]
    run_dir = ROOT / "results" / "rollouts" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if (run_dir / "index.json").is_file():
        for r in json.loads((run_dir / "index.json").read_text())["episodes"]:
            rows[f"{r['model']}|{r['task_id']}"] = r
    base = args.server or start_server()
    print(f"server {base}; {len(tasks)} tasks x {len(args.models)} models -> {run_dir}", flush=True)

    jobs = []
    for t in tasks:  # task-major, so every model has episodes in flight from the start
        for m in args.models:
            path = run_dir / model_slug(m) / f"{t.task_id}.json"
            if path.is_file() and not args.force:
                rec = json.loads(path.read_text())
                if not rec.get("errors") and rec.get("end_reason") != "wall_clock":
                    rows[f"{m}|{t.task_id}"] = summarize(rec, t)
                    continue
            jobs.append((m, t, path))
    write_index(run_dir, rows)
    def work(m, t, path):
        rec = run_episode(m, base, t.task_id, max_turns=args.max_turns, max_tokens=args.max_tokens,
                          wall_clock_s=args.wall_clock)
        rec["run"] = args.run
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return m, t, rec

    # One pool per model: a slow model's episodes never hold the workers a fast model needs.
    pools = {m: ThreadPoolExecutor(max_workers=args.concurrency) for m in args.models}
    futs = [pools[m].submit(work, m, t, path) for m, t, path in jobs]
    done = 0
    for f in as_completed(futs):
        m, t, rec = f.result()
        rows[f"{m}|{t.task_id}"] = summarize(rec, t)
        write_index(run_dir, rows)
        done += 1
        err = f" ERR {rec['errors'][0][:120]}" if rec.get("errors") else ""
        print(f"[{done}/{len(jobs)}] {m} {t.task_id} reward={rec['reward']:.3f} {rec['end_reason']} "
              f"turns={rec['turns']} {rec['seconds']:.0f}s{err}", flush=True)
    for p in pools.values():
        p.shutdown()
    table = board(list(rows.values()))
    (run_dir / "board.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
