"""The scripted desks of airops-occ3, played in process through OccDay3's calls on a b6-v3 pack's dates, with the
checks that no page or reply leaks the references or the future. No model, no network: the desks are
agentenv/agents/airops-desk's scripts, each also calling every lookup of its desk every turn so that every tool's
reply is checked too.

    python tools/scripted_desks3.py --out cache/scripted3 [--pack tasks/b6-v3] [--split train] [--only TASK_ID,...]
        [--workers 2]

Per date, three desks (joint) and one agent at all three (one):
- audit and replays: graded, the audit passes, the record replays byte for byte;
- refs_moved, the references: the day played again with the pack's references moved (best, bound, B0, the gap, the
  naive control and the best plan) gives every page, every tool reply and the env's record (data/get) at every watch
  before the day ends byte for byte, and the same digest; and refs_unseen: no page or reply before the end shows the
  reference, the bound or a reward;
- future_unseen, the future: the day with every delay a desk can't know by a watch (its flight due after the watch
  plus LOOK) and every such block changed gives every reply and record up to that watch byte for byte (v2's leak
  test, tools/scripted_desks.py);
- the largest page per desk.
The crew desk's options (the standard crew desk's search, seconds a call) are asked every SOMETIMES watches.
dates/<task id>.json as each finishes, then summary.json.
"""

import argparse
import importlib.util
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from airops_core.core.known import LOOK
from airops_core.occ.day import digest
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack3, World
from airops_core.occ.replay import ROLE, replay

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("airops_desk", ROOT / "agentenv" / "agents" / "airops-desk" / "agent.py")
desk_agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(desk_agent)
LEAK_AT = 600
FIRST = re.compile(r"\bf\d+\b")
CHECKS = ("audit", "replays", "refs_moved", "refs_unseen", "future_unseen")
ATTENTION = {"ids": None, "station": None, "status": "attention", "start": None, "end": None}
LOOKUPS = {"ops": ("ops_flights", "ops_program", "ops_aircraft", "ops_check"),
           "crew": ("crew_flights", "crew_crews", "crew_reserves", "crew_program", "crew_options", "crew_check"),
           "customer": ("cust_flights", "cust_flight")}
LOOKUPS["one"] = LOOKUPS["ops"] + LOOKUPS["crew"] + LOOKUPS["customer"]
SOMETIMES = {"crew_options": 4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--split", default="train,eval,control")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    (a.out / "dates").mkdir(parents=True, exist_ok=True)
    pack = Pack3(a.pack)
    only, splits = {x for x in a.only.split(",") if x}, set(a.split.split(","))
    ids = [t for t, v in sorted(pack.tasks.items()) if v["split"] in splits and (not only or t in only)]
    rows = {}
    with ProcessPoolExecutor(min(a.workers, len(ids)), max_tasks_per_child=1) as ex:
        futures = {ex.submit(job, str(pack.path), t): t for t in ids}
        for f in as_completed(futures):
            t = futures[f]
            rows[t] = f.result()
            (a.out / "dates" / f"{t}.json").write_text(json.dumps(rows[t], indent=1, default=str))
            print(f"{t}: {'ok' if rows[t]['ok'] else 'FAILED ' + str(rows[t]['failed'])} {rows[t]['seconds']}s",
                  file=sys.stderr, flush=True)
    summary = {"dates": len(rows), "ok": all(r["ok"] for r in rows.values()),
               "passed": sorted(t for t, r in rows.items() if r["ok"]),
               "failed": {t: r["failed"] for t, r in rows.items() if not r["ok"]},
               "checks": {f"{m} {k}": sum(r.get(m, {}).get(k) is True for r in rows.values())
                          for m in ("joint", "one") for k in CHECKS},
               "pages": {d: max((r[m]["pages"].get(d, 0) for r in rows.values() for m in ("joint", "one")
                                 if m in r), default=0) for d in LOOKUPS}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def job(pack: str, task_id: str) -> dict:
    t0 = time.monotonic()
    out, failed = {"task_id": task_id}, []
    try:
        w = Pack3(pack).world(task_id)
        for mode in ("joint", "one"):
            row = out[mode] = checks(w, mode)
            failed += [f"{mode} {k}" for k in CHECKS if not row[k]]
    except Exception:
        out["error"] = traceback.format_exc()
        failed.append("error")
    out["failed"], out["ok"] = failed, not failed
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def checks(w: World, mode: str) -> dict:
    a = play(w, mode)
    moved = play(refs_moved(w), mode)
    at = next(t for t in a.times if t >= LEAK_AT)
    later = play(leaky(w, at), mode)
    g = a.grade
    again = replay(w, a.data(), OccDay3)
    return {"cost": g["cost"], "reward": g["reward"], "reward_L": g["reward_L"], "best": g["best"], "B0": g["B0"],
            "passengers": g["passengers"], "requests": g["counts"]["requests"], "audit": a.audit()["ok"],
            "replays": again.exact, "digest": a.digest(),
            "refs_moved": mid_day(a) == mid_day(moved) and a.digest() == moved.digest(),
            "refs_unseen": scan(said(a), w.refs), "future_unseen": upto(a, at) == upto(later, at), "leak_at": at,
            "calls": len(a.calls), "replies": len(a.replies),
            "tools": sorted({r[2] for r in a.replies}),
            "pages": {d: max((p["chars"] for p in a.shown if p["desk"] == d), default=0) for d in a.desks}}


def play(w: World, mode: str) -> OccDay3:
    """The day played by the scripts, each desk also calling every lookup of its desk every turn; every reply before
    the end kept on the day as `replies` (seq, desk, tool, the reply as JSON), and the env's record as each watch opens
    as `records` (watch, minute, its digest)."""
    d = OccDay3(w, mode)
    d.replies, d.records = [], []
    scripts = {k: desk_agent.Script(k) for k in d.desks}
    hub = w.task.hub if w.task.hub in w.task.programs else min(w.task.programs, default=w.task.hub)

    def call(desk: str, tool: str, args: dict) -> dict:
        out = d.call(ROLE[desk], tool, args)
        if not d.done:
            d.replies.append((d.seq, desk, tool, json.dumps(out, sort_keys=True, default=str)))
        return out
    while not d.done:
        if not d.records or d.records[-1][0] != d.k:
            d.records.append((d.k, d.t, digest(d.data())))
        desk = d.turn["desk"]
        for n in list(d.pending):
            d.notice(n, "trigger")
        page = call(desk, "wait_turn", {})
        first = "f0"
        for tool in LOOKUPS[desk]:
            if tool in SOMETIMES and d.k % SOMETIMES[tool]:
                continue
            out = call(desk, tool, lookup(tool, hub, first))
            if tool.endswith("_flights") and (m := FIRST.search(json.dumps(out.get("rows", [])))):
                first = m[0]

        async def act(name: str, args: dict, desk: str = desk) -> dict:
            return json.loads(json.dumps(call(desk, name, args)))
        done(scripts[desk].turn(page["page"], act))
        call(desk, "end_turn", {"note": None, "stand_down": False})
    for desk in d.desks:
        d.call(ROLE[desk], "resign", {"reason": "done"})
    return d


def lookup(tool: str, hub: str, first: str) -> dict:
    if tool.endswith("_flights"):
        return ATTENTION
    return {"ops_program": {"hub": hub}, "crew_program": {"hub": hub}, "ops_aircraft": {"station": None},
            "ops_check": {"changes": []}, "crew_check": {"moves": []}, "crew_reserves": {"base": None},
            "crew_crews": {"unit": None, "station": None, "base": None, "kind": None, "group": None,
                           "state": "on_duty"},
            "crew_options": {"flight": first, "kind": "pilot"}, "cust_flight": {"flight": first}}[tool]


def done(coro) -> None:
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass


def said(d: OccDay3) -> list:
    return [*d.replies, *[(p["watch"], p["desk"], p["text"]) for p in d.shown]]


def mid_day(d: OccDay3) -> list:
    return [*said(d), *d.records]


def upto(d: OccDay3, t: int) -> list:
    last = max((c["seq"] for c in d.calls if c["t"] <= t), default=0)
    return [*[r for r in d.replies if r[0] <= last], *[r for r in d.records if r[1] <= t]]


def refs_moved(w: World) -> World:
    r = w.refs
    return replace(w, refs={**r, "best": round(r["best"] * 1.37, 1), "B0": {"notice": round(r["B0"]["notice"] * 1.5,
                                                                                          1)},
                            "bound": None if r["bound"] is None else round(r["bound"] * 0.61, 1), "gap_pct": 99.9,
                            "best_from": "moved", "naive": None if r.get("naive") is None else r["naive"] * 2},
                   best_plan={})


def leaky(w: World, t: int) -> World:
    """The world with every delay a desk can't know by t (not negative, its flight due after t + LOOK) and every such
    block changed."""
    flights = tuple(replace(f, lb=f.lb + 37, block=f.block + 11) if f.sched_dep > t + LOOK and f.lb >= 0 else f
                    for f in w.task.flights)
    return replace(w, task=replace(w.task, flights=flights))


def scan(seen: list, refs: dict) -> bool:
    """No page or reply shows the reference or the bound (as the pages print numbers), nor a reward."""
    values = [v for v in (refs["best"], refs["bound"]) if v is not None]
    marks = [f(v) for v in values for f in (lambda v: f"{v:,.0f}", lambda v: f"{round(v)}", lambda v: f"{v:.1f}")]
    text = "\n".join(str(x[-1]) for x in seen)
    return not any(m in text for m in marks) and '"reward' not in text


if __name__ == "__main__":
    main()
