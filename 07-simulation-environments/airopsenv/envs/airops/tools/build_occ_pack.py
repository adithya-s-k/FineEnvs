"""The ops control center's task pack: per date of the split the crew filter kept (eval and train) and the two
controls, the whole-day task, its crew world frozen at the start and its references, built once so the env never
rebuilds units. No LLM is called; the crew and aircraft solvers run (HiGHS, one thread each).

    python tools/build_occ_pack.py --out cache/<build> --run cache/<crew_check run> --split <split.json>
                                     --tasks cache/<world_check run> [--only DATE,...] [--workers 4] [--resume]
                                     [--scale 1.0]
    python tools/build_occ_pack.py --assemble cache/<build> --pack tasks/b6-v2 [--scripted cache/scripted]

Per date, in its own process:
1. the trips and the units at the start (crewday.storm_units on --run's own budgets, the middle band), frozen with
   every airport a unit may reach (its time zone, for the station clocks and crews.Ends) and the round trips of the
   planned rotations (occ.round_trips);
2. F1, the same world as the crew run: the morning replay's objective and flights without a crew equal --run's, and
   --run's stored best plan (notice-priced) replays with its crews named (fill off) at its best (within 0.5) with a
   legal crew on every flight it flies;
3. if F1 holds, --run's best plan and bound stand; otherwise crewplan.joint runs again on the frozen units with --run's
   budgets (scaled by --scale), the standard crew desk's day and the airline's among its candidates, and its best,
   bound and gap are published;
4. the references on the frozen units: B0 (the standard crew desk, aircraft doing nothing; notice and flat), idle
   (the desk asking ops to cancel where it is stuck and nobody answering), the airline's replay by the trip rule and
   --run's crew-aware calm day; the bound must not exceed any of them.
It writes worlds/<date>.json.gz and dates/<date>.json as each date finishes (`--resume` keeps those written without an
error), then summary.json and report.md. `--assemble` writes the pack: dates/<date>.json.gz (the task, its split set,
and its frozen world) and manifest.json (the sha256 of every file, the commit, the framework and crew_params.json);
with `--scripted`, each date tools/scripted_desks.py played gets the env's scripted desks' costs as references
(joint: the crew desk forwarding its system's words and ops accepting its cancellations; one: one desk running both by
script).
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import resource
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from airops_core.programs import crewbound, crewday, crewplan, crews, crewsim, occ, references, sim
from airops_core.programs.model import Task

ROOT = Path(__file__).resolve().parents[1]
INPUTS, DERIVED = ROOT / "cache" / "inputs", ROOT / "cache" / "derived"
CONTROLS = ("2024-07-03", "2025-01-15")
SKIPPED = ("2023-06-02",)
TOL = 0.5
RSS_UNIT = 1 << 30 if sys.platform == "darwin" else 1 << 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--run", type=Path, help="the crew_check run: its budgets, best plans and bounds")
    ap.add_argument("--split", type=Path, help="the split: the eval dates (key eval) and the train dates (key train)")
    ap.add_argument("--tasks", type=Path, help="the world_check run whose tasks.jsonl the crew run read")
    ap.add_argument("--only", default="", help="comma-separated dates (default: every eval, train and control date)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the joint search's budgets on a rerun")
    ap.add_argument("--assemble", type=Path, help="a build's out directory: write the pack from it")
    ap.add_argument("--pack", type=Path, default=ROOT / "tasks" / "b6-v2")
    ap.add_argument("--scripted", type=Path,
                    help="a tools/scripted_desks.py run: its scripted desks' costs become references")
    a = ap.parse_args()
    if a.assemble:
        print(json.dumps(assemble(a.assemble, a.pack, a.scripted), indent=1))
        return
    if not (a.out and a.run and a.split and a.tasks):
        sys.exit("ERROR: --out, --run, --split and --tasks are required")
    split = json.loads(a.split.read_text())
    parts = {**{d: "eval" for d in split["eval"]}, **{d: "train" for d in split["train"]},
             **dict.fromkeys(CONTROLS, "control")}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    dates = {d: p for d, p in parts.items() if d not in SKIPPED and (not only or d in only)}
    for sub in ("dates", "worlds"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    lines = {}
    for line in (a.tasks / "tasks.jsonl").open():
        head = json.loads(line[:line.index(',"tz"')] + "}")
        if head["date"] in dates:
            lines[head["date"]] = line
    if missing := sorted(set(dates) - set(lines)):
        sys.exit(f"ERROR: no task for {', '.join(missing)}")
    todo = [d for d in sorted(lines) if not (a.resume and done(a.out / "dates" / f"{d}.json"))]
    todo.sort(key=lambda d: -len(lines[d]))
    t0 = time.monotonic()
    if todo:
        with ProcessPoolExecutor(min(a.workers, len(todo)), max_tasks_per_child=1) as pool:
            futures = {pool.submit(job, lines[d], dates[d], a.run, a.out, a.scale): d for d in todo}
            for n, f in enumerate(as_completed(futures), 1):
                print(f"[{time.monotonic() - t0:8.0f}s] {n}/{len(todo)} {futures[f]} {f.result()}", flush=True)
    rows = [json.loads((a.out / "dates" / f"{d}.json").read_text()) for d in sorted(dates)
            if (a.out / "dates" / f"{d}.json").exists()]
    report = summary(rows, a, time.monotonic() - t0)
    (a.out / "summary.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (a.out / "report.md").write_text(markdown(report, rows))
    print(json.dumps({k: report[k] for k in ("dates", "errors", "f1_failed", "reruns", "bound_problems")},
                     indent=1), flush=True)


def done(path: Path) -> bool:
    return path.exists() and "error" not in json.loads(path.read_text())


def job(line: str, part: str, run: Path, out: Path, scale: float) -> str:
    t0 = time.monotonic()
    task = replace(Task.from_dict(json.loads(line)), split=part)
    row = {"date": task.date, "task_id": task.task_id, "split": part}
    try:
        ran = json.loads((run / "dates" / f"{task.date}.json").read_text())
        plans = json.loads((run / "plans" / f"{task.date}.json").read_text())
        world, info = freeze(task, ran, plans, scale)
        with gzip.open(out / "worlds" / f"{task.date}.json.gz", "wt") as fh:
            json.dump(world, fh, separators=(",", ":"))
        row |= info
    except Exception:
        row["error"] = traceback.format_exc()
    row["seconds"] = round(time.monotonic() - t0, 1)
    row["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT, 2)
    (out / "dates" / f"{task.date}.json").write_text(json.dumps(row, indent=1, default=str))
    return "error" if "error" in row else f"{row['refs']['best_from']} {row['seconds']}s {row['peak_rss_gb']}GB"


def freeze(task: Task, ran: dict, plans: dict, scale: float) -> tuple[dict, dict]:
    """The date's frozen world and references, and what the build found on the way."""
    if "error" in ran:
        raise RuntimeError("error in the crew run on this date")
    laps = {}
    t0 = time.monotonic()
    budget = ran["budget"]
    w = crewday.world(task, INPUTS, DERIVED)
    units, before = crewday.storm_units(w, "middle", budget["morning"], budget["polish"])
    laps["units"] = round(time.monotonic() - t0, 1)
    world = sim.truth(task)
    j = ran["bands"]["middle"]["joint"]["notice"]
    morning = ran["bands"]["middle"]["record"]["before_start"]
    stored = plan_of(plans["notice"])
    absent = sorted(names(stored) - {u.id for u in units})
    f1 = {"units": {k: [before.get(k), morning.get(k)] for k in ("objective", "uncrewed")}, "absent": absent[:20]}
    if absent:
        f1 |= {"stored": [None, j["best"]], "uncrewed": None, "ok": False}
    else:
        replay = crewsim.execute(task, stored, units, world, w.local, fill=False, airports=w.airports)
        replayed = crewplan.worth(replay, False)
        f1 |= {"stored": [round(replayed, 1), j["best"]], "uncrewed": list(replay.uncrewed)}
        f1["ok"] = all(a == b for a, b in f1["units"].values()) and abs(replayed - j["best"]) <= TOL and \
            not replay.uncrewed
    t1 = time.monotonic()
    if f1["ok"]:
        best, bound, plan, rerun = j["best"], j["bound"], stored, None
    else:
        rerun = joint(task, w, units, {k: v * scale for k, v in budget.items() if k != "width"} |
                      {"width": budget["width"]})
        best, bound, plan = rerun.pop("best"), rerun.pop("bound"), rerun.pop("plan")
    laps["best"] = round(time.monotonic() - t1, 1)
    t1 = time.monotonic()
    b0 = crewsim.execute(task, {}, units, world, w.local, airports=w.airports, desk=True)
    idle = crewsim.execute(task, {}, units, world, w.local, airports=w.airports, desk=True, drop=False)
    airline = crewsim.execute(task, references.airline_plan(task), units, world, w.local, airports=w.airports)
    laps["references"] = round(time.monotonic() - t1, 1)
    calm = ran["bands"]["middle"]["calm"]
    refs = {"best": round(best, 1), "bound": round(bound, 1), "gap_pct": round(100 * (best - bound) / best, 4),
            "best_from": "run" if rerun is None else "rerun", "B0": {"notice": round(crewplan.worth(b0, False), 1),
                                                                      "flat": round(crewplan.worth(b0, True), 1)},
            "idle": round(crewplan.worth(idle, False), 1), "airline": round(crewplan.worth(airline, False), 1),
            "calm": {k: calm.get(k) for k in ("base", "crew_aware", "rise_pct", "bound")},
            "bands": {"pilot": crews.SHARES["pilot"]["middle"], "cabin": crews.SHARES["cabin"]["middle"]}}
    played = {"B0": refs["B0"]["notice"], "idle": refs["idle"], "airline": refs["airline"], "best": best}
    below = {k: v for k, v in played.items() if v < bound - TOL}
    trips = occ.round_trips(w.legs, w.flight_of())
    codes = sorted(w.airports)
    world_json = {"date": task.date, "task_id": task.task_id, "units": [u.to_dict() for u in units],
                  "airports": {c: w.airports[c].to_dict() for c in codes}, "trips": [list(p) for p in trips],
                  "refs": refs, "best_plan": {str(k): v for k, v in plan.items()},
                  "morning": {k: before.get(k) for k in ("objective", "uncrewed", "real_layovers")}}
    info = {"flights": len(task.flights), "units": len(units), "trips": len(trips), "F1": f1, "rerun": rerun,
            "refs": refs, "below_bound": below, "laps": laps,
            "statuses": {k: counts(d) for k, d in (("B0", b0), ("idle", idle))}}
    return world_json, info


def counts(day: crewsim.CrewDay) -> dict[str, int]:
    return dict(sorted(Counter(r.status for r in day.run.flights.values()).items()))


def plan_of(raw: dict) -> dict:
    return {int(k) if k != "reports" else k: v for k, v in raw.items()}


def names(plan: dict) -> set[str]:
    """Every unit a plan names: crews, riders and reports."""
    out = {k.split("#")[0] for k in plan.get("reports", {})}
    for k, e in plan.items():
        if k != "reports":
            out |= set(e.get("crew", {}).values()) | set(e.get("deadhead", ()))
    return out


def joint(task: Task, w: crewday.World, units, budget: dict) -> dict:
    """crewplan.joint on the frozen units, notice-priced, as the crew run ran it (tools/crew_check.py)."""
    world = sim.truth(task)
    air = crewplan.Aircraft(task, world)
    fl = crewbound.Flights.of(task, world.block, air.net.copies)
    reached = crewbound.reach(task, fl, units, w.local)
    least = crewbound.least(task, fl, units, w.local, crews.Ends(units, w.airports, task.date).least, reached)
    desk = crewsim.execute(task, {}, units, world, w.local, airports=w.airports, desk=True)
    j = crewplan.joint(task, world, units, w.local, w.airports, air, fl, reached, least, budget["search"],
                       budget["cover"], crew_limit=max(60.0, budget["search"] / 4),
                       days=(("standard crew desk", crewday.named(desk)),
                             ("airline replay", references.airline_plan(task))), width=budget["width"])
    replay = crewsim.execute(task, j.found.plan, units, world, w.local, fill=False, airports=w.airports)
    best = crewplan.worth(replay, False)
    return {"best": best, "bound": j.bound, "plan": j.found.plan, "exact": j.found.exact and abs(best - j.found.value)
            <= TOL, "claimed": round(j.found.claimed, 1), "pieces": {k: None if v is None else round(v, 1) for k, v
                                                                     in j.pieces.items()},
            "attempts": [{k: v for k, v in a.items() if k in ("try", "claimed", "replayed")} for a in
                         j.found.attempts], "budget": budget}


def summary(rows: list[dict], a, seconds: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    return {"dates": len(rows), "errors": sorted(r["date"] for r in rows if "error" in r),
            "f1_failed": sorted(r["date"] for r in ok if not r["F1"]["ok"]),
            "reruns": {r["date"]: {"best": r["refs"]["best"], "bound": r["refs"]["bound"],
                                   "gap_pct": r["refs"]["gap_pct"], "exact": r["rerun"]["exact"]}
                       for r in ok if r["rerun"] is not None},
            "bound_problems": {r["date"]: r["below_bound"] for r in ok if r["below_bound"]},
            "seconds": round(seconds, 1), "per_date_s": sorted(r.get("seconds", 0) for r in rows)[-5:],
            "peak_rss_gb": max((r.get("peak_rss_gb", 0) for r in rows), default=0), "commit": commit(),
            "inputs": {"run": str(a.run), "split": str(a.split), "tasks": str(a.tasks), "scale": a.scale}}


def markdown(report: dict, rows: list[dict]) -> str:
    out = ["# The ops control center's pack build", "", f"Commit {report['commit']}; {report['dates']} dates, "
           f"{len(report['errors'])} errors, F1 failed on {len(report['f1_failed'])}, bound problems on "
           f"{len(report['bound_problems'])}.", "",
           "| Date | Split | Flights | Units | F1 | Best from | Best | Bound | Gap % | B0 | Idle | Airline | s |",
           "|" + "---|" * 13]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['date']} | {r['split']} | error |" + " |" * 10)
            continue
        f = r["refs"]
        out.append(f"| {r['date']} | {r['split']} | {r['flights']} | {r['units']} | {r['F1']['ok']} | "
                   f"{f['best_from']} | {f['best']} | {f['bound']} | {f['gap_pct']} | {f['B0']['notice']} | "
                   f"{f['idle']} | {f['airline']} | {r['seconds']} |")
    return "\n".join(out) + "\n"


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(build: Path, pack: Path, scripted: Path | None = None) -> dict:
    """The pack from a build: per date the build wrote without an error, one file with its task (its split set) and
    its frozen world, and the manifest."""
    rows = {p.stem: json.loads(p.read_text()) for p in sorted((build / "dates").glob("*.json"))}
    good = {d: r for d, r in rows.items() if "error" not in r}
    report = json.loads((build / "summary.json").read_text())
    tasks = Path(report["inputs"]["tasks"])
    tasks = tasks if tasks.is_absolute() else ROOT / tasks
    if pack.exists():
        shutil.rmtree(pack)
    (pack / "dates").mkdir(parents=True)
    for line in (tasks / "tasks.jsonl").open():
        head = json.loads(line[:line.index(',"tz"')] + "}")
        if (d := head["date"]) not in good:
            continue
        task = replace(Task.from_dict(json.loads(line)), split=good[d]["split"])
        world = json.loads(gzip.decompress((build / "worlds" / f"{d}.json.gz").read_bytes()))
        if scripted and (z := scripted / "dates" / f"{d}.json").is_file():
            row = json.loads(z.read_text())
            world["refs"]["scripted"] = {"joint": row["channel"]["cost"], "one": row["one"]["cost"]}
        with gzip.GzipFile(pack / "dates" / f"{d}.json.gz", "wb", mtime=0) as fh:
            fh.write(json.dumps({"task": task.to_dict(), **world}, separators=(",", ":")).encode())
    files = {str(p.relative_to(pack)): sha256(p) for p in sorted(pack.rglob("*")) if p.is_file()}
    manifest = {"pack": pack.name, "built_at_commit": report["commit"], "assembled_at_commit": commit(),
                "framework": importlib.metadata.version("agentenv-framework"),
                "crew_params_sha256": sha256(crews.SOURCES),
                "tasks": {r["task_id"]: {"date": d, "split": r["split"], "best_from": r["refs"]["best_from"]}
                          for d, r in sorted(good.items())},
                "skipped": {d: "error in the build" for d in sorted(set(rows) - set(good))} | dict.fromkeys(
                    SKIPPED, "its joint search crashed in the crew run; outside the pool by rule"), "files": files}
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"dates": len(good), "skipped": manifest["skipped"], "files": len(files),
            "bytes": sum(p.stat().st_size for p in pack.rglob("*") if p.is_file())}


if __name__ == "__main__":
    sys.exit(main())
