"""Does the storm-day world reproduce the real record well enough to grade against? Its gates were set before the run.
No LLM is called; only the reference solvers run (HiGHS).

    python tools/world_check.py --out cache/<run> [--workers 4] [--only DATE,...] [--resume] [--seed 0]
                                  [--refs naive,airline,recorded,anchor,flat,calm] [--prove-s 3600] [--watch-s 2400]
                                  [--scale JFK=0.5,BOS=0.5]

1. every storm date over the 19 cached months (programs.network: a date whose per-hub candidate rule holds at JFK or
   BOS, those being its storm hubs, and whose programs control or hold at least 5 of the task's JetBlue flights) built
   as a whole-network task at scale 1 and the 03:00 cutoff, plus the control dates (the control rule's, each with its
   hubs' windows); the day-hubs that can't be candidates (a neighbour day not loaded) are counted with their reason,
   and so are the dates an advisory-archive outage drops and those too few flights qualify;
2. the seeded whole-event split over the built storm dates, stratified by storm hub (network.split: per hub 12 eval
   dates, 8 summer and 4 winter, one per storm event, a date drawn for one hub at most; train every built date of an
   event without one). Being built is the only clause;
3. the unseen-traffic scale per airport fitted on the train dates' storm hubs alone (unseen.fit_scale); `--scale` sets
   it instead, and fidelity then fails as not out of sample. A hub without train dates stops the run unless `--scale`
   is given;
4. the references `--refs` names on every task at that scale, one JSON per task under refs/ as it finishes (`--resume`
   keeps each whose signature is unchanged). By default: naive, the airline replay, the recorded day, the anchor
   (notice-priced, proven and canonical), the flat anchor and the calm day (from the positions the schedule assumes,
   network.scheduled); the follower, the program-blind follower, the rule and the calm day from the record's morning
   positions (`morning`, a finding beside the per-task shortfall, network.shortfall) run when named;
5. a task is kept when every MIP it ran proved to GAP and no reference failed; any other is left out of every gate's
   pool and counted, and a drawn eval date so left out is dropped, not replaced;
6. gates.json (the gates with values, thresholds and pass/fail, the parameters and the diagnostics), candidates.jsonl
   (every date's build, split and proof) and report.md (numbers only).

The gates, as registered, a task being a storm date:
- airline_replay: on >= 90% of the kept storm dates, the airline replay's auto-cancels are at most 1% of the flights
  JetBlue flew by their deadline and at most 2;
- fidelity: the other carriers' modelled program delay over their BTS NAS delay at each storm hub of a date, pooled
  per hub over the kept dates whose own ratio there is within 0.5-2 (the others dropped and counted), within 0.8-1.25
  on eval and on train separately, at a scale fitted on train;
- calm_day: at most 1 forced cancellation and 2 ferries on >= 90% of the kept storm dates;
- controls: on both control dates, JetBlue's recorded day costs at least the anchor, and naive is within 3% of the
  flat anchor;
- references: every kept task's anchor proven and replayed exactly, no reference below a proven anchor, pax <= seats
  on every flight a reference flies, the tasks left out and the references compared with the anchors counted;
- days, reported only: the built storm dates by storm hub(s) and season, and the split.
Reported beside them: per date the record's morning against the schedule's positions (the aircraft out of position,
the departures they leave without an aircraft on time) and the tails away from a gateway or out of service.

--only takes dates (2024-08-19) or task ids (b6-20240819-0400), storm and control dates alike.
"""

import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from airops_core.build import pack
from airops_core.core.sim import AUTO_CANCELLED, FLOWN
from airops_core.programs import candidates, network, rbs, references, sim, unseen
from airops_core.programs.advisories import HUBS
from airops_core.programs.model import Task, write_tasks
from airops_core.programs.repairs import CUTOFF, DMAX

ROOT = Path(__file__).resolve().parents[1]
INPUTS, DERIVED = ROOT / "cache" / "inputs", ROOT / "cache" / "derived"
AUTO_SHARE, AUTO_MAX = 0.01, 2
POOLED, PER_DAY = (0.8, 1.25), (0.5, 2.0)
CONTROL_MARGIN = 0.03
CONTROL_DATES = 2
CALM_CANCELS, CALM_FERRIES = 1, 2
GATE_SHARE = 0.9
LEVERAGE = 0.10
CUTOFF_SHARE = 0.9
PROVE_S, WATCH_S = 3600.0, 2400.0
RSS_UNIT = 1 << 20 if sys.platform == "darwin" else 1 << 10
REFERENCES = ("naive", "airline", "recorded", "anchor", "flat", "follower", "blind", "rule", "calm", "morning")
DEFAULT_REFS = ("naive", "airline", "recorded", "anchor", "flat", "calm")
SOLVED = ("anchor", "flat", "calm")
PLAYED = ("naive", "airline", "anchor", "follower", "blind", "rule")
COLUMNS = ("task_id", "split", "hubs", "start", "kept", "flights", "tails", "moves", "gateway", "out_of_service",
           "anchor", "naive", "follower", "blind", "rule", "airline", "recorded", "flat_anchor", "naive_flat", "calm",
           "replay_auto", "fidelity", "shortfall", "morning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated dates or task ids (default: all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", default=",".join(DEFAULT_REFS),
                    help=f"comma-separated references of {', '.join(REFERENCES)}")
    ap.add_argument("--prove-s", type=float, default=PROVE_S, help="time limit of each anchor, flat or calm solve")
    ap.add_argument("--watch-s", type=float, default=WATCH_S, help="time limit of each follower watch solve")
    ap.add_argument("--scale", default="", help="HUB=scale,... in place of the train fit (fidelity then fails)")
    args = ap.parse_args()
    refs = chosen(args.refs)
    limits = (args.prove_s, args.watch_s)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    only = wanted(args.only)
    ones = dict.fromkeys(HUBS, 1.0)

    fits = candidates.e_table(INPUTS, DERIVED, workers=args.workers)
    e_fit, e_median = unseen.table(fits)
    storms, counts = network.storms(INPUTS)
    controls = network.control_dates(INPUTS, DERIVED)
    jobs = [(d, hubs, None, "") for d, hubs in storms.items() if only(d)]
    jobs += [(d, tuple(h for h in HUBS if h in w), w, "control") for d, w in controls.items() if only(d)]
    built = network.build_all(jobs, INPUTS, DERIVED, e_fit, e_median, ones, workers=args.workers)
    tasks = {b.task.task_id: b.task for b in built if b.task and b.task.split != "control"}
    held = {b.task.task_id: b.task for b in built if b.task and b.task.split == "control"}
    clock.lap("build", f"{len(tasks)} of {sum(not j[3] for j in jobs)} storm dates built, {len(held)} control dates")

    event = candidates.events(storms)
    drawn = network.split({t.date: t.storms for t in tasks.values()}, event, args.seed)
    train = [t for t in tasks.values() if drawn.get(t.date) == "train"]
    fitted = unseen.fit_scale(train, fits)
    scale = parsed(args.scale) if args.scale else fitted
    if set(scale) != set(HUBS):
        sys.exit(f"ERROR: no train dates to fit the scale at {sorted(set(HUBS) - set(scale))} "
                 f"({len(train)} train dates); give --scale to run anyway, and fidelity fails")
    shipped_e = unseen.table(fits, scale)
    shipped = {i: rbs.rescale(replace(t, split=drawn.get(t.date, "") if i in tasks else "control"), scale, *shipped_e)
               for i, t in {**tasks, **held}.items()}
    write_tasks(list(shipped.values()), out / "tasks.jsonl")
    clock.lap("split", f"eval {sum(v == 'eval' for v in drawn.values())}, train {len(train)}, scale {scale}")

    results = run_pass(out / "refs", shipped, refs, args.workers, args.resume, limits, clock)
    clock.lap("references")

    kept = {i for i, r in results.items() if not unproven(r)}
    final = network.settle(drawn, [t.date for i, t in tasks.items() if i in kept], event)
    split = {i: final[t.date] for i, t in tasks.items() if t.date in final}
    fid = {(i, h): unseen.program_delay(t, hub=h) for i, t in shipped.items() if i in tasks for h in t.storms}
    params = {"seed": args.seed, "references": list(refs), "scale": scale, "scale_fitted_on_train": fitted,
              "out_of_sample": not args.scale, "train_dates_fitted": len(train), "cutoff": CUTOFF, "dmax": DMAX,
              "time_limits_s": limits, "e_median": {"scale_1": e_median, "shipped": shipped_e[1]},
              "advisories_fitted": len(fits), "months": list(candidates.MONTHS),
              "late_departures": late_departures([(t.date, h) for t in train for h in t.storms])}
    report = summary(tasks, held, shipped, results, kept, drawn, split, fid, built, counts, params, clock, jobs)
    rows = candidate_rows(built, results, kept, drawn, split, fid)
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (out / "gates.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (out / "report.md").write_text(markdown(report))
    print(json.dumps(report["gates"], indent=1, default=str), flush=True)


class Clock:
    def __init__(self):
        self.t0 = self.last = time.monotonic()
        self.laps: dict[str, float] = {}

    def lap(self, name: str, note: str = "") -> None:
        now = time.monotonic()
        self.laps[name] = round(now - self.last, 1)
        self.last = now
        print(f"[{now - self.t0:8.1f}s] {name} {note}".rstrip(), flush=True)


def chosen(text: str) -> tuple[str, ...]:
    """The references named, in REFERENCES order."""
    names = {x.strip() for x in text.split(",") if x.strip()}
    if names - set(REFERENCES):
        sys.exit(f"ERROR: unknown references {sorted(names - set(REFERENCES))}; choose from {', '.join(REFERENCES)}")
    return tuple(k for k in REFERENCES if k in names)


def wanted(text: str) -> Callable[[str], bool]:
    """Whether a date is among --only's dates or task ids (everything when empty)."""
    items = [x.strip() for x in text.split(",") if x.strip()]
    dates = set()
    for x in items:
        if x.lower().startswith("b6-"):
            ymd = x.split("-")[1]
            dates.add(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
        else:
            dates.add(x)
    return lambda d: not items or d in dates


def parsed(text: str) -> dict[str, float]:
    return {h.strip().upper(): float(v) for h, _, v in (x.partition("=") for x in text.split(",") if x.strip())}


def late_departures(keys: Iterable[tuple[str, str]]) -> dict:
    """JetBlue's recorded hub departures after local midnight on the train dates' storm hubs, and the first whole hour
    by which CUTOFF_SHARE of them had left: reported only, the cutoff stays at its default."""
    deps = []
    for d, hub in sorted(keys):
        x = pack.local(candidates.frame(INPUTS, DERIVED, d), candidates.midnight(d))
        ours = x[(x.Reporting_Airline == candidates.CARRIER) & (x.FlightDate == d) & (x.Origin == hub)
                 & (x.Cancelled == 0) & (x.ad >= 1440)]
        deps += [int(a) for a in ours.ad]
    hour = next((60 * h for h in range(25, 49) if sum(d < 60 * h for d in deps) >= CUTOFF_SHARE * len(deps)),
                None) if deps else None
    return {"n": len(deps), "hour_holding_share": hour}


def run_pass(folder: Path, tasks: Mapping[str, Task], refs: tuple[str, ...], workers: int, resume: bool,
             limits: tuple[float, float], clock: Clock) -> dict[str, dict]:
    """The references on every task in a process pool, heaviest first, each result written to folder/<task id>.json
    as it finishes; with `resume`, a result whose signature matches is kept."""
    folder.mkdir(parents=True, exist_ok=True)
    results, jobs = {}, []
    for i, t in tasks.items():
        sig = signature(t, refs, limits)
        path = folder / f"{i}.json"
        if resume and path.exists():
            kept = json.loads(path.read_text())
            if kept.get("signature") == sig:
                results[i] = kept
                continue
        jobs.append((i, t.to_json(), sig, refs, limits))
    jobs.sort(key=lambda j: weight(tasks[j[0]]), reverse=True)
    if jobs:
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            futures = {pool.submit(job, j): j[0] for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                (folder / f"{r['task_id']}.json").write_text(json.dumps(r, default=str))
                results[r["task_id"]] = r
                print(f"  [{time.monotonic() - clock.t0:8.1f}s] {len(results)}/{len(tasks)} {r['task_id']} "
                      f"{brief(r)}", flush=True)
    return results


def signature(task: Task, refs: tuple[str, ...], limits: tuple[float, float]) -> str:
    return hashlib.sha256(f"{task.to_json()}|{refs}|{limits}".encode()).hexdigest()[:16]


def weight(task: Task) -> int:
    """The anchor MIP's departure copies, which its size and solve time follow."""
    return sum(max(0, f.deadline - task.earliest(f, f.lb) + 1) for f in task.flights)


def brief(r: dict) -> str:
    parts = [f"{k}={v}s" for k, v in r.get("seconds", {}).items()] + [f"rss={r.get('peak_rss_mb')}MB"]
    if unproven(r):
        parts.append(f"LEFT OUT {unproven(r)}")
    if r.get("below_anchor"):
        parts.append(f"BELOW ANCHOR {r['below_anchor']}")
    return " ".join(parts)


def job(j: tuple) -> dict:
    """The references of one task: its costs, counts, proofs and solve times."""
    tid, line, sig, refs, (prove, watch) = j
    task = Task.from_dict(json.loads(line))
    out = {"task_id": tid, "signature": sig, "refs": list(refs), "split": task.split, "hubs": list(task.storms),
           "date": task.date, "start": task.start, "flights": len(task.flights), "tails": len(task.tails),
           "moves": sum(len(t.repositions) for t in task.tails), "watches": len(references.watch_times(task)),
           "connections": {"flown": sum(len(t.connections) for t in task.tails), "faster": len(task.quick)},
           "gateway": task.clauses.get("network", {}).get("gateway busy", 0),
           "out_of_service": task.clauses.get("network", {}).get("out of service", 0),
           "shortfall": network.shortfall(task), "seconds": {}}
    anchor = {}
    plays: dict[str, Callable[[], dict]] = {
        "naive": lambda: played(task, references.naive(task)),
        "airline": lambda: replayed(task),
        "recorded": lambda: recorded(task),
        "anchor": lambda: anchored(task, prove, anchor),
        "flat": lambda: solved(references.flat_anchor(task, prove, start=anchor.get("run"))),
        "follower": lambda: followed(task, references.follow(task, True, watch)),
        "blind": lambda: followed(task, references.follow(task, False, watch)),
        "rule": lambda: followed(task, references.rule(task)),
        "calm": lambda: calmed(references.calm(task, prove)),
        "morning": lambda: calmed(references.calm(task, prove, record=True)),
    }
    for name in refs:
        attempt(out, name, plays[name])
    a = out.get("anchor", {})
    if a.get("proven"):
        costs = {k: out[k]["total"] for k in PLAYED if k != "anchor" and "total" in out.get(k, {})}
        out["below_anchor"] = references.below_anchor(costs, a["objective"])
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT)
    return out


def attempt(out: dict, name: str, play: Callable[[], dict]) -> None:
    t0 = time.monotonic()
    try:
        out[name] = play()
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    out["seconds"][name] = round(time.monotonic() - t0, 1)


def played(task: Task, run: sim.Execution) -> dict:
    return {**run.cost.to_dict(), **run.counts, "seats_ok": references.seats_ok(task, run)}


def replayed(task: Task) -> dict:
    air = references.airline(task)
    return {**played(task, air.run), "auto": air.auto, "flew": air.flew, "causes": causes(task, air.run)}


def causes(task: Task, run: sim.Execution) -> list[dict]:
    """Where the record says why, each flight JetBlue flew by its deadline that the replay auto-cancels: its tail's
    first leg, the tail not ready at its origin by JetBlue's departure (`start position`); the leg before it reached
    another station (`repositioning BTS lacks`), carried its aircraft through a Move (`carried by a Move`) or
    auto-cancelled too (`leg before auto-cancelled`); else `other` (the pooled aircraft went to other flights)."""
    carried, _ = sim.routes(task, {f.id for f in task.flights if f.actual.dep is None})
    before = {}
    for t in task.tails:
        for k, i in enumerate(t.legs):
            before[i] = t.legs[k - 1] if k else None
    tails = {t.tail: t for t in task.tails}
    out = []
    for i in references.counted(task):
        if run.flights[i].status != AUTO_CANCELLED:
            continue
        f, t = task.flight(i), tails.get(task.flight(i).actual.tail)
        p = task.flight(before[i]) if before.get(i) is not None else None
        if p is None:
            cause = "start position" if t is None or t.pos != f.orig or t.ready > f.actual.dep else "other"
        elif p.actual.end != f.orig:
            cause = "repositioning BTS lacks"
        elif p.id in carried:
            cause = "carried by a Move"
        elif run.flights[p.id].status == AUTO_CANCELLED:
            cause = "leg before auto-cancelled"
        else:
            cause = "other"
        out.append({"id": i, "flight": f.flight, "orig": f.orig, "tail": f.actual.tail, "cause": cause})
    return out


def recorded(task: Task) -> dict:
    cost, ferries = references.recorded(task)
    return {**cost.to_dict(), "ferries": ferries}


def anchored(task: Task, prove: float, keep: dict) -> dict:
    sol, run = references.anchor(task, prove)
    keep["run"] = run
    exact = run.cost.total == sol.objective and all(
        (run.flights[i].dep if run.flights[i].status == FLOWN else None) == d for i, d in sol.deps.items())
    return {**sol.to_dict(), **run.cost.to_dict(), **run.counts, "proven": sol.proven, "exact": exact,
            "seats_ok": references.seats_ok(task, run), "past_midnight": references.past_midnight(task, run)}


def solved(sol) -> dict:
    return {**{k: v for k, v in sol.to_dict().items() if k in ("status", "objective", "gap_pct", "wall_s")},
            "proven": sol.proven}


def followed(task: Task, f: references.Followed) -> dict:
    cancels = [v for v in f.plan.values() if v.get("cancel")]
    walls = [w.get("wall_s", 0) for w in f.watches]
    return {**played(task, f.run), "watch_solves": sum("wall_s" in w for w in f.watches),
            "max_watch_s": max(walls, default=0), "sum_watch_s": round(sum(walls), 1),
            "decided": dict(Counter(v["decided"] for v in cancels))}


def calmed(c: references.Calm) -> dict:
    """The calm day's forced cancellations, each with why (`short`: on the calm schedule from its start positions its
    origin has fewer aircraft of its family ready than departures by its scheduled departure; `knock-on` otherwise)
    and the wait the calm naive day gives it; how many a Move binds or follow one (`moved`); and its ferries."""
    task = c.task
    ids = sorted(i for i, d in c.solution.deps.items() if d is None)
    near = {m.leg for t in task.tails for m in t.repositions} | {
        j for t in task.tails for k, j in enumerate(t.legs) if k and t.legs[k - 1] in task.moved}
    naive = sim.execute(task, {}, sim.calm(task))
    out = []
    for i in ids:
        f, n = task.flight(i), naive.flights[i]
        b = balance(task, f.family, f.orig, task.earliest(f, 0))
        out.append({"id": i, "flight": f.flight, "route": f"{f.orig}-{f.dest}", "family": f.family,
                    "sched_dep": f.sched_dep, "jetblue": "cancelled" if f.actual.dep is None else "flew",
                    "balance": b, "cause": "short" if b < 0 else "knock-on", "moved": i in near,
                    "naive_wait": None if n.dep is None else n.dep - f.sched_dep})
    return {"cancelled": c.cancelled, "ferries": c.ferries, "ids": ids, "moved": sum(i in near for i in ids),
            "causes": out, "ferry_list": [[x.family, x.orig, x.dest] for x in c.run.ferries], **solved(c.solution)}


def balance(task: Task, family: str, station: str, t: int) -> int:
    """On the calm schedule from the start positions (every flight leaving at its scheduled departure, no earlier
    than the start), the aircraft of `family` ready at `station` by minute t less its departures from there by t."""
    n = sum(x.family == family and x.pos == station and x.ready <= t for x in task.tails)
    for f in task.flights:
        if f.family == family:
            dep = task.earliest(f, 0)
            back, m = task.back(f, dep + f.sched_block), task.moved.get(f.id)
            n += (m.dest if m else f.dest) == station and (max(back, m.ready) if m else back) <= t
            n -= f.orig == station and dep <= t
    return n


def unproven(r: dict) -> list[str]:
    """The references of a task's result that failed or whose MIP did not prove: the reasons it is left out."""
    if r.get("error"):
        return ["task"]
    refs = r.get("refs", REFERENCES)
    bad = [k for k in refs if "error" in r.get(k, {"error": "missing"})]
    return bad + [k for k in SOLVED if k in refs and k not in bad and not r[k]["proven"]]


def summary(tasks: Mapping[str, Task], held: Mapping[str, Task], shipped: Mapping[str, Task],
            results: Mapping[str, dict], kept: set[str], drawn: Mapping[str, str], split: Mapping[str, str],
            fid: Mapping[tuple[str, str], tuple], built: list, counts: dict, params: dict, clock: Clock,
            jobs: list) -> dict:
    pool = {i: r for i, r in results.items() if i in tasks and i in kept}
    autos = [(r["airline"]["auto"], r["airline"]["flew"]) for r in pool.values() if "auto" in r.get("airline", {})]
    replay = [replays(a, n) for a, n in autos]
    calm = [r["calm"]["cancelled"] <= CALM_CANCELS and r["calm"]["ferries"] <= CALM_FERRIES for r in pool.values()
            if "cancelled" in r.get("calm", {})]
    ctl = {i: control_checks(results.get(i, {}), i in kept) for i in held}
    left = {i: unproven(r) for i, r in results.items() if i not in kept}
    gates = {
        "airline_replay": gate(share(replay), len(replay),
                               f"auto-cancels <= {AUTO_SHARE:.0%} of the flights JetBlue flew by their deadline and "
                               f"<= {AUTO_MAX}, per kept storm date",
                               {"rounded_up_share": share([a <= min(AUTO_MAX, math.ceil(AUTO_SHARE * n))
                                                           for a, n in autos]),
                                "auto_cancels": sum(a for a, _ in autos), "flew": sum(n for _, n in autos)}),
        "fidelity": fidelity_gate({k: v for k, v in fid.items() if k[0] in pool}, split, params["out_of_sample"]),
        "calm_day": gate(share(calm), len(calm), f"<= {CALM_CANCELS} forced cancellation and <= {CALM_FERRIES} "
                         "ferries, per kept storm date"),
        "controls": {"controls": ctl, "n": len(ctl), "threshold": f"both control dates: recorded >= anchor and naive "
                     f"within {CONTROL_MARGIN:.0%} of the flat anchor", "pass": len(ctl) == CONTROL_DATES
                     and all(c["ok"] for c in ctl.values())},
        "references": references_gate({i: r for i, r in results.items() if i in kept}, left),
        "days": days_report(tasks, drawn, split, kept, counts, built, jobs),
    }
    gates["go"] = all(g["pass"] for k, g in gates.items() if k != "days")
    return {"gates": gates, "parameters": params,
            "diagnostics": {**diagnostics(tasks, results, pool, split, shipped), **findings(tasks, results)},
            "tasks": [row_of(i, r, i in kept, {h: v for (j, h), v in fid.items() if j == i})
                      for i, r in sorted(results.items())],
            "seconds": clock.laps, "timing": timing(results)}


def replays(auto: int, flew: int) -> bool:
    """The airline replay gate's literal reading of one day: its auto-cancels at most AUTO_SHARE of the flights flown
    and at most AUTO_MAX."""
    return auto <= AUTO_SHARE * flew and auto <= AUTO_MAX


def gate(value: float | None, n: int, rule: str, alt: dict | None = None) -> dict:
    return {"value": value, "threshold": GATE_SHARE, "n": n, "rule": rule,
            "pass": value is not None and value >= GATE_SHARE, **(alt or {})}


def share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def control_checks(r: dict, kept: bool) -> dict:
    if not kept or not all(k in r for k in ("naive", "recorded", "anchor", "flat")):
        return {"ok": False, "left_out": unproven(r) if r else ["missing"]}
    naive_ok = r["naive"]["flat"] <= (1 + CONTROL_MARGIN) * r["flat"]["objective"]
    recorded_ok = r["recorded"]["total"] >= r["anchor"]["objective"]
    return {"ok": naive_ok and recorded_ok, "recorded": r["recorded"]["total"], "anchor": r["anchor"]["objective"],
            "recorded_ge_anchor": recorded_ok, "naive_flat": r["naive"]["flat"], "flat_anchor": r["flat"]["objective"],
            "naive_flat_over_anchor": round(r["naive"]["flat"] / r["flat"]["objective"] - 1, 4),
            "naive_within_margin": naive_ok,
            "pax_delay": {"recorded": r["recorded"]["delay"], "anchor": r["anchor"]["delay"]}}


def fidelity_gate(fid: Mapping[tuple[str, str], tuple], split: Mapping[str, str], out_of_sample: bool) -> dict:
    """Per split and hub, the modelled over the observed other-carrier program delay, pooled over the (task, storm
    hub) days whose own ratio is within PER_DAY; passes only when every pool is within POOLED and the scale was fitted
    on train."""
    out = {"threshold": {"pooled": POOLED, "per_day_filter": PER_DAY}, "out_of_sample": out_of_sample}
    ok = out_of_sample
    for s in ("eval", "train"):
        for h in HUBS:
            p = pooled([v for (i, hub), v in fid.items() if split.get(i) == s and hub == h])
            out[f"{s}_{h}"] = p
            ok = ok and p["ratio"] is not None and POOLED[0] <= p["ratio"] <= POOLED[1]
    return {**out, "pass": ok}


def pooled(days: list[tuple[float, float, int]]) -> dict:
    """Modelled over observed program delay over the days whose own ratio is within PER_DAY; the others, and those
    without observed delay, counted."""
    kept = [(m, o) for m, o, _ in days if o > 0 and PER_DAY[0] <= m / o <= PER_DAY[1]]
    observed = sum(o for _, o in kept)
    measured = sum(1 for _, o, _ in days if o > 0)
    return {"ratio": round(sum(m for m, _ in kept) / observed, 4) if observed else None, "days": len(kept),
            "dropped": measured - len(kept), "no_delay": len(days) - measured}


def references_gate(kept: Mapping[str, dict], left: Mapping[str, list[str]]) -> dict:
    """Over the kept tasks: anchors proven and exact, no reference below a proven anchor, pax <= seats; the gate
    covers only the references that ran, counted per reference."""
    with_anchor = {i: r for i, r in kept.items() if "anchor" in r}
    compared = Counter(k for r in with_anchor.values() for k in PLAYED if k != "anchor" and "total" in r.get(k, {}))
    not_exact = sorted(i for i, r in with_anchor.items() if not r["anchor"]["exact"])
    below = {i: r["below_anchor"] for i, r in kept.items() if r.get("below_anchor")}
    seats = sorted(i for i, r in kept.items() if not all(r[k]["seats_ok"] for k in PLAYED if k in r))
    gaps = [r["anchor"]["gap_pct"] for r in with_anchor.values()]
    return {"kept": len(kept), "anchored": len(with_anchor), "left_out": len(left),
            "left_out_by": dict(Counter(k for v in left.values() for k in v)),
            "left_out_tasks": dict(sorted(left.items())), "anchor_not_exact": not_exact, "below_anchor": below,
            "compared_with_anchor": {k: compared[k] for k in PLAYED if compared[k]}, "pax_over_seats": seats,
            "max_anchor_gap_pct": max(gaps, default=None),
            "pass": bool(with_anchor) and len(with_anchor) == len(kept) and not not_exact and not below and not seats}


def days_report(tasks: Mapping[str, Task], drawn: Mapping[str, str], split: Mapping[str, str], kept: set[str],
                counts: dict, built: list, jobs: list) -> dict:
    """The built storm dates by storm hub(s) and season, the seeded split as drawn and as kept, and why dates or
    day-hubs failed."""
    def by(ids: Iterable[str]) -> dict:
        return dict(sorted(Counter(network.label(tasks[i]) for i in ids).items()))

    eval_drawn = [i for i, t in tasks.items() if drawn.get(t.date) == "eval"]
    return {"candidates": counts["candidates"], "dates_run": sum(not j[3] for j in jobs), "built": len(tasks),
            "built_by": by(tasks), "fallback_windows": sum(t.clauses["fallback"] for t in tasks.values()),
            "kept": len(kept & set(tasks)),
            "drawn": {"eval": by(eval_drawn), "train": sum(v == "train" for v in drawn.values()),
                      "neither": len(tasks) - len(drawn)},
            "split": {"eval": by(i for i, v in split.items() if v == "eval"),
                      "train": by(i for i, v in split.items() if v == "train")},
            "eval_dropped": sorted(i for i in eval_drawn if i not in kept),
            "not_built": dict(Counter(b.row.get("reason") for b in built if b.task is None)),
            "rejected": counts["rejected"], "archive_gap_dates": counts["archive_gap_dates"],
            "counts": {k: v for k, v in counts.items() if isinstance(v, int)}, "pass": True}


def diagnostics(tasks: Mapping[str, Task], results: Mapping[str, dict], pool: Mapping[str, dict],
                split: Mapping[str, str], shipped: Mapping[str, Task]) -> dict:
    """What the report reads beside the gates: leverage, the replay's auto-cancels by cause, the calm day's
    cancellations by cause, the connections JetBlue flew faster than the rule, the references over the anchor, the
    construction and the solve times."""
    priced = {i: r for i, r in pool.items() if all(k in r for k in ("naive", "flat", "anchor"))}
    flat = [r["naive"]["flat"] / r["flat"]["objective"] - 1 for r in priced.values()]
    notice = [r["naive"]["total"] / r["anchor"]["objective"] - 1 for r in priced.values()]
    every = [r for r in results.values() if "auto" in r.get("airline", {})]
    calms = [c for r in pool.values() for c in r.get("calm", {}).get("causes", ())]
    over = {k: dist1([100 * (r[k]["total"] / r["anchor"]["objective"] - 1) for r in priced.values()
                      if "total" in r.get(k, {})]) for k in ("naive", "follower", "blind", "rule", "airline")}
    return {
        "leverage": {"flat": {**dist1(flat), "share_ge_10pct": share([v >= LEVERAGE for v in flat])},
                     "notice": {**dist1(notice), "share_ge_10pct": share([v >= LEVERAGE for v in notice])}},
        "auto_cancel_causes": {"all_tasks": dict(Counter(c["cause"] for r in every for c in r["airline"]["causes"])),
                               "kept_storm_dates": dict(Counter(c["cause"] for r in pool.values()
                                                                for c in r.get("airline", {}).get("causes", ()))),
                               "by_task": {r["task_id"]: r["airline"]["causes"] for r in every
                                           if r["airline"]["causes"]}},
        "calm_cancels": {"total": len(calms), "by_cause": dict(Counter(c["cause"] for c in calms)),
                         "jetblue_cancelled": sum(c["jetblue"] == "cancelled" for c in calms),
                         "next_to_a_move": sum(c["moved"] for c in calms),
                         "ferries": sum(r.get("calm", {}).get("ferries", 0) for r in pool.values())},
        "connections": {"flown": sum(len(x.connections) for t in shipped.values() for x in t.tails),
                        "faster_than_rule": sum(len(t.quick) for t in shipped.values())},
        "construction": construction(shipped),
        "over_anchor_pct": over,
        "recorded_over_anchor": dist1([r["recorded"]["total"] / r["anchor"]["objective"] for r in priced.values()
                                       if "recorded" in r]),
        "replay_over_recorded": dist1([r["airline"]["total"] / r["recorded"]["total"] for r in pool.values()
                                       if "total" in r.get("airline", {}) and r.get("recorded", {}).get("total")]),
        "anchor_cancels_per_day": dist1([r["anchor"]["cancelled"] + r["anchor"]["auto_cancelled"]
                                         for r in priced.values()]),
        "jetblue_cancels_per_day": dist1([sum(f.actual.dep is None for f in tasks[i].flights) for i in pool]),
        "mip": {k: dist1([r["anchor"][k] for r in priced.values()]) for k in ("cols", "rows", "copies")},
        "unseen_share": unseen.unseen_share([tasks[i] for i, v in split.items() if v == "train"]),
        "movements": movements([tasks[i] for i, v in split.items() if v == "train"]),
    }


def findings(tasks: Mapping[str, Task], results: Mapping[str, dict]) -> dict:
    """Over the storm dates: the record's morning against the schedule's positions (network.shortfall) and, when it
    ran, the calm day from the record's morning; the tails away from a gateway and out of service per date."""
    storm = {i: r for i, r in results.items() if i in tasks and "shortfall" in r}
    short = [r["shortfall"] for r in storm.values()]
    by: dict[str, Counter] = {"family": Counter(), "station": Counter()}
    for v in short:
        for k, n in v["short"].items():
            fam, station = k.split()
            by["family"][fam] += n
            by["station"][station] += n
    ran = [r["morning"] for r in storm.values() if "cancelled" in r.get("morning", {})]
    return {"morning": {"dates": len(short), "dates_short": sum(v["aircraft"] > 0 for v in short),
                        "aircraft": dist1([v["aircraft"] for v in short]),
                        "departures": dist1([v["departures"] for v in short]),
                        "fleet": dist1([v["fleet"] for v in short]),
                        "by_family": dict(by["family"].most_common()),
                        "by_station": dict(by["station"].most_common(15)),
                        "calm_day_from_it": {"dates": len(ran), "cancelled": sum(c["cancelled"] for c in ran),
                                             "ferries": sum(c["ferries"] for c in ran),
                                             "within_calm_caps": share([c["cancelled"] <= CALM_CANCELS
                                                                        and c["ferries"] <= CALM_FERRIES for c in ran]),
                                             "by_cause": dict(Counter(x["cause"] for c in ran for x in c["causes"]))}},
            "tails": {k: {**dist1([r[k] for r in storm.values()]), "total": sum(r[k] for r in storm.values())}
                      for k in ("gateway", "out_of_service")}}


def construction(shipped: Mapping[str, Task]) -> dict:
    """Over every task: where the starts came from, the Moves and busy blocks, the short stations of tomorrow's
    needs, the tasks whose windows fell back to the first program hour."""
    net: Counter = Counter()
    for t in shipped.values():
        net.update(t.clauses.get("network", {}))
    short = {i: t.clauses["short"] for i, t in sorted(shipped.items()) if t.clauses.get("short")}
    return {"network": dict(sorted(net.items())), "short_dates": len(short),
            "short_stations": sum(sum(v.values()) for v in short.values()), "short_by_task": short,
            "fallback": sorted(i for i, t in shipped.items() if t.clauses.get("fallback"))}


def movements(train: list[Task]) -> dict:
    """The unseen stream's share of program-hour demand on the train dates beside the share of each airport's own
    counted arrivals that BTS misses (BTS: arrivals that operated, over the cached months)."""
    out = unseen.unseen_share(train)
    days = {m: len(pack.month_days(m)) for m in candidates.MONTHS}
    for hub in HUBS:
        n = 0
        for m in candidates.MONTHS:
            x = candidates.month_frame(INPUTS, DERIVED, m)
            n += int(((x.Dest == hub) & (x.Cancelled == 0) & x.FlightDate.str.startswith(m)).sum())
        out[hub] = {**out.get(hub, {}), **unseen.missed_share(hub, n / sum(days.values()), candidates.MONTHS, days)}
    return out


def row_of(i: str, r: dict, kept: bool, fid: Mapping[str, tuple]) -> dict:
    costs = {k: r[k]["total"] for k in ("anchor", "naive", "follower", "blind", "rule", "airline", "recorded")
             if "total" in r.get(k, {})}
    calm, morning, short = r.get("calm", {}), r.get("morning", {}), r.get("shortfall", {})
    return {"task_id": i, "split": r.get("split"), "hubs": "+".join(r.get("hubs", ())), "start": r.get("start"),
            "kept": kept, "left_out": unproven(r), "flights": r.get("flights"), "tails": r.get("tails"),
            "moves": r.get("moves"), "gateway": r.get("gateway"), "out_of_service": r.get("out_of_service"),
            "connections": list((r.get("connections") or {}).values()), **costs,
            "flat_anchor": (r.get("flat") or {}).get("objective"), "naive_flat": (r.get("naive") or {}).get("flat"),
            "calm": [calm.get("cancelled"), calm.get("ferries")],
            "shortfall": [short.get("aircraft"), short.get("departures"), short.get("fleet")],
            "morning": [morning.get("cancelled"), morning.get("ferries")],
            "replay_auto": [(r.get("airline") or {}).get("auto"), (r.get("airline") or {}).get("flew")],
            "fidelity": {h: round(m / o, 3) if o > 0 else None for h, (m, o, _) in sorted(fid.items())},
            "anchor_gap_pct": (r.get("anchor") or {}).get("gap_pct"), "anchor_exact": (r.get("anchor") or {}).get(
                "exact"), "below_anchor": r.get("below_anchor"), "peak_rss_mb": r.get("peak_rss_mb"),
            "seconds": r.get("seconds", {})}


def candidate_rows(built: list, results: Mapping[str, dict], kept: set[str], drawn: Mapping[str, str],
                   split: Mapping[str, str], fid: Mapping[tuple[str, str], tuple]) -> list[dict]:
    rows = []
    for b in built:
        row = dict(b.row)
        i = row.get("task_id")
        if b.task is None:
            rows.append({**row, "built": False})
            continue
        r = results.get(i, {})
        control = b.task.split == "control"
        rows.append({**row, "built": True, "build_clauses": b.task.clauses,
                     "drawn": "control" if control else drawn.get(b.task.date),
                     "split": "control" if control else split.get(i), "kept": i in kept,
                     "left_out": unproven(r) if r else ["missing"],
                     "fidelity": {h: list(v) for (j, h), v in fid.items() if j == i},
                     "replay_auto": [(r.get("airline") or {}).get("auto"), (r.get("airline") or {}).get("flew")]})
    return rows


def timing(results: Mapping[str, dict]) -> dict:
    per = defaultdict(list)
    for r in results.values():
        for k, v in r.get("seconds", {}).items():
            per[k].append(v)
    total = [sum(r.get("seconds", {}).values()) for r in results.values()]
    solves = {"anchor_wall_s": [r["anchor"]["wall_s"] for r in results.values() if "wall_s" in r.get("anchor", {})],
              "max_watch_s": [r[k]["max_watch_s"] for r in results.values() for k in ("follower", "blind")
                              if "max_watch_s" in r.get(k, {})],
              "peak_rss_mb": [r["peak_rss_mb"] for r in results.values() if r.get("peak_rss_mb")]}
    return {"per_task": {k: dist1(v) for k, v in per.items()}, "task_total": dist1(total),
            "solves": {k: dist1(v) for k, v in solves.items()}, "sum_s": round(sum(total), 1)}


def dist1(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "min": round(v[0], 3), "p10": round(percentile(v, 10), 3),
            "median": round(statistics.median(v), 3), "p90": round(percentile(v, 90), 3), "max": round(v[-1], 3),
            "mean": round(statistics.mean(v), 3)}


def percentile(values: list[float], pct: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, math.ceil(pct / 100 * len(v)) - 1)]


def markdown(report: dict) -> str:
    g = report["gates"]
    lines = [f"# Storm-day world check: numbers ({datetime.now(UTC).date().isoformat()})", "", "## Gates", "",
             "| Gate | Value | Threshold | Pass |", "|---|---|---|---|"]
    for name, x in g.items():
        if name == "go":
            continue
        value = {k: v for k, v in x.items() if k not in ("threshold", "pass", "rule", "controls", "left_out_tasks")}
        lines.append(f"| {name} | {json.dumps(value, default=str)} | {x.get('threshold')} | {x['pass']} |")
    lines += ["", f"All gated pass: {g['go']}", "", "## Controls", "",
              f"```\n{json.dumps(g['controls']['controls'], indent=1, default=str)}\n```", "", "## Parameters", "",
              f"```\n{json.dumps(report['parameters'], indent=1, default=str)}\n```", "", "## Diagnostics", "",
              f"```\n{json.dumps(report['diagnostics'], indent=1, default=str)}\n```", "", "## Timing", "",
              f"```\n{json.dumps({'stages_s': report['seconds'], **report['timing']}, indent=1, default=str)}\n```",
              "", "## Per task", "", "| " + " | ".join(COLUMNS) + " | seconds |", "|" + "---|" * (len(COLUMNS) + 1)]
    for row in report["tasks"]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + f" | {json.dumps(row['seconds'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
