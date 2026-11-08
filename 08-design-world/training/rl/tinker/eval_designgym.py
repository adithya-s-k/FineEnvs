"""Evaluate a Tinker model (base or checkpoint) on DesignGym, through the openenv_rl adapter.

Same episodes as `envs/designgym/rollout.py` (turn caps, 0-1000 GUI grid for Qwen, thinking off, one skin per
task), but sampled on Tinker, so a base model and its SFT/RL checkpoints are scored by the same code path the
trainer uses. Reward is whatever the server grades with: start it with DESIGN_CANVAS_REWARD=3.

    # server (reward v3; the GUI surface needs `playwright install chromium`)
    DESIGN_CANVAS_REWARD=3 designgym serve --port 8007

    python eval_designgym.py --model Qwen/Qwen3.8-27B --renderer qwen3_8_disable_thinking --out evals/q38-base
    python eval_designgym.py --model Qwen/Qwen3.6-35B-A3B --renderer qwen3_5_disable_thinking --out evals/q36-base
    python eval_designgym.py --model Qwen/Qwen3.8-27B --model-path tinker://<run>/sampler_weights/<step> ...
    python eval_designgym.py ... --limit 3 --surfaces tools,html,gui   # pilot: measure latency and cost first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

import workspace_env
from designgym_rl import initial_messages
from openenv_rl import ImageWindow, MCPSessionFactory, OpenEnvEnvGroupBuilder

TURNS = {"tools": 60, "html": 20, "gui": 200}  # rollout.py DEFAULT_TURNS
SKINS = ("figma", "canva", "photoshop", "slides", "powerpoint", "inkscape")
EVALSET = Path(__file__).resolve().parents[3] / "bench" / "recreation-cua-v1.json"  # the frozen DesignWorldBench set


def evalset_tasks(subset: str, limit: int | None, evalset: Path = EVALSET) -> list[str]:
    record = json.loads(Path(evalset).read_text())
    ids = sorted(record["subsets"][subset]) if subset != "full" else sorted(e["task_id"] for e in record["tasks"])
    return ids[:limit] if limit else ids


async def run_one(sampler, a, task_id: str, surface: str, k: int, factory, sem) -> dict:
    task = {"task_id": task_id, "mode": "reference", "surface": surface, "max_steps": TURNS[surface]}
    if surface == "gui":
        task |= {"skin": SKINS[k % len(SKINS)], "coordinates": "normalized"}
    builder = OpenEnvEnvGroupBuilder(
        factory, task=task, group_size=1, model_name=a.model, renderer_name=a.renderer,
        max_turns=TURNS[surface], finalize_tool="submit", missing_reward=0.0,
        # drop screenshots in chunks so each prompt extends the last one and Tinker bills it as cached prefill
        image_window=ImageWindow(8, 3) if surface == "gui" else ImageWindow(6, 2),
        tags=["designgym", surface],
    )
    policy = TinkerTokenCompleter(sampler, max_tokens=a.max_tokens, temperature=a.temperature)
    async with sem:
        t0 = time.time()
        try:
            group = await do_group_rollout(builder, policy)
            traj = group.trajectories_G[0]
            last = traj.transitions[-1].metrics if traj.transitions else {}
            return {"task_id": task_id, "surface": surface, "reward": sum(t.reward for t in traj.transitions),
                    "turns": len(traj.transitions), "submitted": bool(last.get("openenv/done")),
                    "tool_errors": last.get("openenv/tool_errors"), "stop": str(traj.stop_reason),
                    "prompt_tokens": sum(t.ob.length for t in traj.transitions),
                    "sampled_tokens": sum(len(t.ac.tokens) for t in traj.transitions),
                    "seconds": round(time.time() - t0, 1), "error": None}
        except Exception as exc:  # noqa: BLE001 - one failed episode must not stop the eval
            return {"task_id": task_id, "surface": surface, "reward": 0.0, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - t0, 1)}


async def main(a) -> None:
    if not workspace_env.load():
        raise SystemExit("TINKER_API_KEY not found")
    service = tinker.ServiceClient()
    sampler = (service.create_sampling_client(model_path=a.model_path) if a.model_path
               else service.create_sampling_client(base_model=a.model))
    factory = MCPSessionFactory(a.server, exclude_tools=("start_task",), initial_messages_builder=initial_messages,
                                reset_kwargs_builder=lambda t: dict(t))
    tasks = evalset_tasks(a.subset, a.limit, a.evalset)
    surfaces = a.surfaces.split(",")
    sem = asyncio.Semaphore(a.concurrency)
    gui_sem = asyncio.Semaphore(a.gui_concurrency)  # each GUI episode drives its own Chromium on the server

    async def job(tid, s, k):
        if s == "gui":
            async with gui_sem:
                return await run_one(sampler, a, tid, s, k, factory, sem)
        return await run_one(sampler, a, tid, s, k, factory, sem)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "trace.jsonl").open("w") as fh:
        jobs = [job(t, s, k + i * len(tasks)) for i in range(a.samples) for k, t in enumerate(tasks) for s in surfaces]
        for fut in asyncio.as_completed(jobs):
            r = await fut
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{len(rows):4d}/{len(jobs)} {r['surface']:5s} {r['task_id']} "
                  f"reward {r['reward']:.3f} turns {r.get('turns', '-')} {r['seconds']:.0f}s {r['error'] or ''}",
                  flush=True)
    summary = {"model": a.model, "model_path": a.model_path, "renderer": a.renderer, "subset": a.subset}
    for s in surfaces:
        xs = [r["reward"] for r in rows if r["surface"] == s]
        per_task = {}
        for r in rows:
            if r["surface"] == s:
                per_task.setdefault(r["task_id"], []).append(r["reward"])
        task_means = [statistics.fmean(v) for v in per_task.values()]
        summary[s] = {"n": len(xs), "mean": round(statistics.fmean(xs), 4),
                      # over tasks (each task's mean over its samples): the spread a paired comparison uses
                      "stderr": round(statistics.stdev(task_means) / len(task_means) ** 0.5, 4) if len(task_means) > 1 else None,
                      "per_task": {t: round(statistics.fmean(v), 4) for t, v in sorted(per_task.items())},
                      "failed": sum(1 for r in rows if r["surface"] == s and r["error"]),
                      "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows if r["surface"] == s),
                      "sampled_tokens": sum(r.get("sampled_tokens", 0) for r in rows if r["surface"] == s)}
    summary["mean"] = round(statistics.fmean(summary[s]["mean"] for s in surfaces), 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Tinker base model, e.g. Qwen/Qwen3.8-27B")
    ap.add_argument("--model-path", default=None, help="tinker://... sampler checkpoint (SFT/RL); default: base")
    ap.add_argument("--renderer", default=None, help="e.g. qwen3_8_disable_thinking, qwen3_5_disable_thinking")
    ap.add_argument("--server", default="http://127.0.0.1:8007")
    ap.add_argument("--evalset", type=Path, default=EVALSET, help="frozen evalset JSON (task ids and subsets)")
    ap.add_argument("--subset", default="core30")
    ap.add_argument("--surfaces", default="tools,html,gui")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only (pilot)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--gui-concurrency", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1, help="episodes per task (several cut the sampling noise)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
