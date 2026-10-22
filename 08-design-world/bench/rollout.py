"""Run DesignWorldBench Recreation-CUA v2: frontier and open models on frozen computer-use episodes.

Every episode in the set is fixed in full (task, editor, seed, theme, file window, and a lived-in workspace: the
task's images under everyday names in everyday folders, among decoy pictures and documents), so every model meets
the same one. The prompt asks for efficient work and names no action count; a generous limit is still enforced.

    # 1. a DesignGym server with the eval pack (and the private apps installed, for all 14 editors)
    DESIGN_CANVAS_PACK=<eval pack> DESIGN_CANVAS_REWARD=3 DESIGN_CANVAS_APPS=daytona designgym serve --port 8000
    # 2. the run
    python bench/rollout.py --base-url http://localhost:8000 --models anthropic:claude-sonnet-5-5 --editors all
    python bench/rollout.py ... --editors builtin --tasks 3          # a quick smoke: 3 tasks x 6 editors
    python bench/rollout.py ... --out runs/<name>                    # rerun the same command to resume

Models are `provider:model[@thinking]` (envs/designgym/rollout.py: anthropic, openai, openai-chat, hf, vllm).
Editors: `all` (14), `public` (the 6 built-in + Inkscape, Impress, Draw), `builtin`, `apps`, or a comma list.

Writes `<out>/trace.jsonl` (every turn of every episode, with the system prompt and tool schemas), `frames/`,
`renders/`, `run.json` (the set, settings, models, server) and `summary.json` / `summary.md` (mean reward per model
and editor, per tier, 95% bootstrap intervals, tokens). `bench/publish.py` turns a run into the cua-v1 dataset
format the SFT data uses and uploads it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "envs" / "designgym"))
sys.path.insert(0, str(HERE.parent / "envs"))

import rollout as core  # noqa: E402  (the agents, providers and the episode loop)

EVALSET = HERE / "evalsets" / "recreation-cua-v2.json"


def pick_editors(record: dict, spec: str) -> list[str]:
    e = record["editors"]
    groups = {"builtin": e["builtin"], "public": e["builtin"] + e["public_apps"], "apps": e["public_apps"] + e["private_apps"],
              "private": e["private_apps"], "all": e["builtin"] + e["public_apps"] + e["private_apps"]}
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        out += groups.get(part, [part])
    known = set(groups["all"])
    unknown = [x for x in out if x not in known]
    if unknown:
        raise SystemExit(f"unknown editors: {', '.join(unknown)} (have {', '.join(sorted(known))})")
    return list(dict.fromkeys(out))


def server_check(base_url: str, editors: list[str], settings: dict) -> None:
    """Fail early when the server cannot play what the run asks for: an app it lacks, or another reward."""
    try:
        info = json.loads(core.fetch(f"{base_url}/api/info") or b"{}")
    except ValueError:
        info = {}
    have = set(info.get("skins", [])) | set(info.get("real_skins", []))
    if have:
        missing = [e for e in editors if e not in have]
        if missing:
            raise SystemExit(f"the server at {base_url} cannot play: {', '.join(missing)}. Install the private apps "
                             "(pip install designworld-private) and start it with DESIGN_CANVAS_APPS set, or choose "
                             "--editors public / builtin.")
    rv = info.get("reward_version")
    if rv is not None and int(rv) != int(settings.get("reward_version", 3)):
        raise SystemExit(f"the server grades with reward v{rv}; this set is scored with v{settings['reward_version']}: "
                         f"start it with DESIGN_CANVAS_REWARD={settings['reward_version']}")


def summarise(records: list[dict], tiers: dict[str, str]) -> tuple[dict, str]:
    # a rerun appends a new record: keep one per episode, the graded one if any, else the latest
    last: dict[tuple, dict] = {}
    for r in records:
        k = (r["model"], r.get("evalset_episode") or (r.get("task_id"), r.get("skin")))
        if k not in last or r.get("reward") is not None or last[k].get("reward") is None:
            last[k] = r
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in last.values():
        by[(r["model"], r.get("skin"))].append(r)
        by[(r["model"], "all")].append(r)
    rows = []
    for (model, skin), recs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1] or "")):
        # graded episodes count (a late provider failure was handed in as it stood); ungraded ones are errors,
        # left out of the mean and listed, to be run again
        graded = [r for r in recs if r.get("reward") is not None]
        vals = [r["reward"] for r in graded] or [0.0]
        lo, hi = core.bootstrap_ci(vals)
        tier = {t: (statistics.mean(v) if (v := [r["reward"] for r in graded if tiers.get(r.get("task_id")) == t]) else None)
                for t in ("easy", "medium", "hard")}
        rows.append({"model": model, "editor": skin, "n": len(graded), "mean": round(statistics.mean(vals), 4) if graded else None,
                     "ci": [round(lo, 4), round(hi, 4)], "errors": len(recs) - len(graded),
                     "tiers": {k: (round(v, 4) if v is not None else None) for k, v in tier.items()},
                     "turns": round(statistics.mean([r.get("n_turns", 0) for r in recs]), 1),
                     "input_tokens": sum((r.get("usage") or {}).get("input", 0) for r in recs),
                     "output_tokens": sum((r.get("usage") or {}).get("output", 0) for r in recs)})
    lines = ["| model | editor | n | reward | 95% CI | easy | medium | hard | turns | errors |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    pc = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    for x in rows:
        ci = "—" if x["mean"] is None else f"{100 * x['ci'][0]:.1f}–{100 * x['ci'][1]:.1f}"
        lines.append(f"| {x['model']} | {x['editor']} | {x['n']} | {pc(x['mean'])} | {ci} | "
                     f"{pc(x['tiers']['easy'])} | {pc(x['tiers']['medium'])} | {pc(x['tiers']['hard'])} | "
                     f"{x['turns']} | {x['errors']} |")
    return {"rows": rows}, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=None, help="a running DesignGym server; default: start one in-process")
    ap.add_argument("--evalset", type=Path, default=EVALSET)
    ap.add_argument("--models", required=True, help="comma list of provider:model[@thinking]")
    ap.add_argument("--thinking", default="default", choices=core.THINKING)
    ap.add_argument("--editors", default="all")
    ap.add_argument("--tasks", type=int, default=None, help="only the first N tasks of the set (a smoke)")
    ap.add_argument("--task-ids", default=None, help="only these task ids (comma list)")
    ap.add_argument("--concurrency", type=int, default=6, help="episodes at once")
    ap.add_argument("--app-concurrency", type=int, default=2, help="episodes at once in each real app")
    ap.add_argument("--coordinates", default=None, help="override the set's choice (auto, pixels, normalized)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--keep-images", type=int, default=4)
    ap.add_argument("--image-window", default=None, type=lambda v: tuple(int(x) for x in v.split(",")))
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    record = json.loads(a.evalset.read_text())
    if record.get("evalset_version") != 2:
        raise SystemExit(f"{a.evalset} is not a v2 set (for v1 use envs/designgym/rollout.py --evalset)")
    settings = record["settings"]
    editors = pick_editors(record, a.editors)
    task_ids = [t["task_id"] for t in record["tasks"]]
    if a.task_ids:
        task_ids = [t for t in task_ids if t in set(a.task_ids.split(","))]
    if a.tasks:
        task_ids = task_ids[: a.tasks]
    episodes = [e for e in record["episodes"] if e["skin"] in editors and e["task_id"] in task_ids]
    tiers = {t["task_id"]: t["tier"] for t in record["tasks"]}

    core.REQUEST_TIMEOUT, core.VERBOSE = a.request_timeout, a.verbose
    env_file = core.load_env_file(a.env_file)
    models = []
    for m in (m.strip() for m in a.models.split(",") if m.strip()):
        base, mode = core.split_spec(m, a.thinking)
        models.append(base if mode == "default" else f"{base}@{mode}")
    need = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openai-chat": "OPENAI_API_KEY",
            "hf": "HF_TOKEN", "vllm": "VLLM_BASE_URL"}
    missing = [m for m in models if not os.environ.get(need.get(m.split(":")[0], ""), "")]
    if missing:
        print(f"missing API keys for: {', '.join(missing)} (env file: {env_file})", file=sys.stderr)
        return 2
    base_url = a.base_url or core.start_local_server()
    server_check(base_url, editors, settings)
    out = a.out or HERE.parent / "results" / "v2" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "evalset": record["name"], "evalset_id": record["evalset_id"], "settings": settings, "models": models,
        "editors": editors, "tasks": task_ids, "base_url": base_url, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))

    # the episode loop takes its limits from these, as the v1 harness's flags
    args = argparse.Namespace(coordinates=a.coordinates or settings.get("coordinates", "auto"), max_steps=settings["max_steps"],
                              max_turns=settings["max_turns"], episode_timeout=settings["episode_timeout"],
                              max_tokens=a.max_tokens, keep_images=a.keep_images, image_window=a.image_window,
                              no_frames=a.no_frames)
    trace_path = out / "trace.jsonl"
    previous = [json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()] if trace_path.exists() else []
    # an episode is done once it was graded (even if the provider failed late and it was handed in as it stood);
    # one that never got a grade (setup or provider failure) runs again
    done = {(r["model"], r.get("evalset_episode")) for r in previous if r.get("reward") is not None}
    jobs = [(m, e) for m in models for e in episodes if (m, e["id"]) not in done]
    print(f"{record['name']} ({record['evalset_id'][:12]}): {len(task_ids)} tasks x {len(editors)} editors x "
          f"{len(models)} models = {len(models) * len(episodes)} episodes; {len(jobs)} to run -> {out}", flush=True)

    gates = defaultdict(lambda: threading.Semaphore(a.app_concurrency))
    lock = threading.Lock()
    trace = open(trace_path, "a")
    records: list[dict] = []

    def job(j):
        model, ep = j
        gate = gates[ep["skin"]] if ep.get("real") else None
        if gate:
            gate.acquire()
        try:
            rec = core.run_episode(base_url, model, "gui", settings["mode"], record["split"], 0, args, out,
                                   entry=next(t for t in record["tasks"] if t["task_id"] == ep["task_id"]),
                                   episode=ep, settings=settings)
        except Exception as exc:
            rec = {"model": model, "surface": "gui", "mode": settings["mode"], "skin": ep["skin"], "task_id": ep["task_id"],
                   "evalset_episode": ep["id"], "reward": None, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:], "turns": [], "n_turns": 0, "tool_calls": 0,
                   "tool_errors": 0, "usage": {}}
        finally:
            if gate:
                gate.release()
        rec["evalset"] = record["name"]
        rec["evalset_id"] = record["evalset_id"]
        with lock:
            trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace.flush()
            records.append(rec)
            rw = rec.get("reward")
            print(f"  {model:40s} {ep['skin']:16s} {ep['task_id']:11s} reward {'—' if rw is None else f'{rw:.3f}'}  "
                  f"turns {rec.get('n_turns', 0):3d}  {rec.get('seconds', 0)}s"
                  + (f"  ERROR {rec['error'][:140]}" if rec.get("error") else ""), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        list(pool.map(job, jobs))
    trace.close()
    latest = {}
    for r in previous + records:
        latest[(r["model"], r.get("evalset_episode"))] = r
    summary, table = summarise(list(latest.values()), tiers)
    summary.update(evalset=record["name"], evalset_id=record["evalset_id"], models=models, editors=editors)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
