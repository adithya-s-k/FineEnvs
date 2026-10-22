"""Digests of a pack date's v2 outputs (the empty plan's day, B0 and the best plan's crew days, the hindsight MIP's
model), which passengers off must leave byte-identical; v2 modules only, so the same code digests v2's own tree. The
date is the fixture pack's eval date."""

import hashlib
import json
from pathlib import Path

import numpy as np

from airops_core.occ.pack import Pack
from airops_core.programs import crewsim, mip, sim

DATE = "2025-01-11"
PACK = Path(__file__).resolve().parents[1] / "fixtures" / "b6-v2"


def digests(date: str = DATE) -> dict[str, str]:
    pack = Pack(PACK)
    w = pack.world(next(t for t, v in pack.tasks.items() if v["date"] == date))
    task = w.task
    naive = sim.execute(task, {})
    out = {"naive": _sha([naive.rows(), naive.cost.to_dict(), naive.legs, naive.ends, [f.to_dict() if hasattr(
        f, "to_dict") else list(f) for f in naive.ferries], naive.wasted, sorted(naive.moves.items())])}
    for name, plan, kw in (("B0", {}, {"desk": True}), ("best", w.best_plan, {"fill": False})):
        day = crewsim.execute(task, plan, w.units, local=w.local, airports=w.airports, **kw)
        out[name] = _sha([day.run.rows(), day.run.cost.to_dict(), day.crews, sorted(day.riders.items()),
                          day.uncrewed, day.cost, day.units, sorted(day.status.items()), day.messages,
                          sorted(day.dropped.items()), day.total])
    m = mip.Network(task).m
    out["mip"] = hashlib.sha256(np.array(m.cost, dtype=float).tobytes() + np.array(m.ub, dtype=float).tobytes()
                                + np.array(m.integer).tobytes() + json.dumps(
                                    [[sorted(c.items()), lo, up] for c, lo, up in m.rows]).encode()).hexdigest()
    return out


def _sha(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()
