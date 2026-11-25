"""Stage 2b of AirOpsEnv v3, as pre-registered: does the passengers' premise hold in the world the env ships, crews
included, and can a scripted customer desk act on it? No LLM is called; the desks are scripts.

    python tools/customer_desk_check.py --out cache/v3/customer [--reference cache/v3/references] [--split eval]
        [--workers 24] [--only DATE,...] [--inputs cache/inputs] [--derived cache/derived] [--pack PATH] [--resume]

Per date, with its passengers booked as step 0 books them (tools/passenger_plans.py) and cost P with crews:
- 2b-1, the plan premise in the crew world: P(v2's joint best plan) / P(stage 2's reference plan), both read from the
  per-date files of tools/passenger_references.py's run under --reference (`candidates.v2_best.P`, `reference.P`);
  it holds on a date at 1.05 or more, and passes on at least 12 of the 24 eval dates;
- 2b-2, a desk can act on it: the two scripted desks of v2's stage 2b with their channel (programs.occ, mode `talk`)
  and the same with the scripted customer desk (`talk-customer`), both on the crew sim with passengers; it holds on a
  date when P(three desks) <= 0.98 x P(two desks), and passes on at least 12 of the 24 eval dates.
The customer desk's holds (flight, minute) and the flights it protected are kept per date. Without --reference, or for
a date the stage-2 run did not write, 2b-1 is left out. Every job writes dates/<date>.json as it finishes (--resume
keeps those written); guards.json is recomputed from those files alone.
"""

import argparse
import glob
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from airops_core.build import nextday
from airops_core.occ.pack import Pack
from airops_core.programs import occ, pax, sim

ROOT = Path(__file__).resolve().parents[1]
PREMISE, DESK, DATES = 1.05, 0.98, 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None, help="a tools/passenger_references.py run")
    ap.add_argument("--split", default="eval", help="eval, train, control or all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated dates")
    ap.add_argument("--inputs", type=Path, default=ROOT / "cache" / "inputs")
    ap.add_argument("--derived", type=Path, default=ROOT / "cache" / "derived")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    days = a.out / "dates"
    days.mkdir(parents=True, exist_ok=True)
    tasks = Pack(a.pack).tasks
    only = set(filter(None, a.only.split(",")))
    dates = sorted(v["date"] for v in tasks.values() if (a.split == "all" or v["split"] == a.split)
                   and (not only or v["date"] in only))
    todo = [d for d in dates if not (a.resume and (days / f"{d}.json").exists())]
    print(f"{len(todo)} jobs on {min(a.workers, max(1, len(todo)))} workers", file=sys.stderr)
    with ProcessPoolExecutor(min(a.workers, max(1, len(todo))), max_tasks_per_child=1) as ex:
        futures = {ex.submit(job, d, a): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                row = fut.result()
            except Exception:
                row = {"date": d, "error": traceback.format_exc()}
            (days / f"{d}.json").write_text(json.dumps(row, indent=1))
            print(f"{d}: {brief(row)}", file=sys.stderr)
    summary = guards(a.out)
    (a.out / "guards.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({g: summary[g]["pass"] for g in ("2b-1", "2b-2")}))


def job(date: str, a: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    pack = Pack(a.pack)
    tid = next(t for t, v in pack.tasks.items() if v["date"] == date)
    w = pack.world(tid)
    task = w.task
    passengers = pax.book(task, pax.Flows.load(date), pax.Dwell.load(), nextday.seats(task, a.inputs, a.derived))
    case = occ.Case(task, w.units, w.local, w.airports, w.trips, sim.truth(task), passengers)
    two, three = occ.run(case, "talk"), occ.run(case, "talk-customer")
    ratio = three.day.total / two.day.total
    row = {"date": date, "task_id": tid, "split": pack.tasks[tid]["split"], "flows": passengers.source,
           "2b-2": {"two": round(two.day.total, 1), "three": round(three.day.total, 1), "ratio": ratio,
                    "holds": ratio <= DESK},
           "customer": {"holds": [[j, m, task.flight(j).sched_dep] for j, m in sorted(three.holds.items())],
                        "protected": sorted(three.protected),
                        "roles": occ.roles(three, occ.MODES["talk-customer"], task), "acausal": three.acausal},
           "outcomes": {"two": {**two.day.run.pax.outcomes(), "misconnected": two.day.run.pax.misconnected()},
                        "three": {**three.day.run.pax.outcomes(), "misconnected": three.day.run.pax.misconnected()}},
           "counts": {"two": occ.counts(two), "three": occ.counts(three)},
           "seconds": {"two": two.seconds, "three": three.seconds, "total": round(time.monotonic() - t0, 1)}}
    ref = a.reference / "dates" / f"{date}.json" if a.reference else None
    if ref is not None and ref.is_file() and "error" not in (r := json.loads(ref.read_text())):
        p_v2, p_ref = r["candidates"]["v2_best"]["P"], r["reference"]["P"]
        row["2b-1"] = {"v2_best": p_v2, "reference": p_ref, "ratio": p_v2 / p_ref, "holds": p_v2 >= PREMISE * p_ref}
    return row


def brief(row: dict) -> str:
    if "error" in row:
        return "ERROR " + row["error"].strip().splitlines()[-1]
    b = row["2b-2"]
    first = f"2b-1 {row['2b-1']['ratio']:.3f}; " if "2b-1" in row else ""
    return (f"{first}2b-2 three/two {b['ratio']:.4f}; holds {len(row['customer']['holds'])}, protected "
            f"{len(row['customer']['protected'])}")


def guards(out: Path) -> dict:
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "dates" / "*.json")))]
    ok = [r for r in rows if "error" not in r]
    base = [r for r in ok if r["split"] == "eval"]
    out_ = {"errors": [r["date"] for r in rows if "error" in r]}
    for gate, rule in (("2b-1", f"P(v2 joint best) >= {PREMISE} x P(stage-2 reference) on >= {DATES} of 24 eval dates"),
                       ("2b-2", f"P(three scripted desks) <= {DESK} x P(two) on >= {DATES} of 24 eval dates")):
        have = [r for r in base if gate in r]
        passing = [r["date"] for r in have if r[gate]["holds"]]
        out_[gate] = {"rule": rule, "dates": len(have), "passing": len(passing),
                      "pass": len(have) == 24 and len(passing) >= DATES, "passing_dates": passing,
                      "ratios": {r["date"]: round(r[gate]["ratio"], 4) for r in ok if gate in r}}
    out_["customer"] = {r["date"]: {"holds": len(r["customer"]["holds"]), "protected": len(r["customer"]["protected"])}
                        for r in ok}
    return out_


if __name__ == "__main__":
    main()
