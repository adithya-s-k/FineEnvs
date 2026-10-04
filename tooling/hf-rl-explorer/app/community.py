"""Runner-independent filtering and honest summaries of saved public rollouts.

Visibility is checked by the API before records enter this module. Facets ignore
their own selection; headline counts always describe the filtered population.
Rewards are grouped by environment, runner, domain, metric, judge and revision.
There is no cross-environment model ranking.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

DOMAINS = {"code": "Code", "webdev": "Webdev", "cyber": "Cyber", "music": "Music", "general": "General"}


def is_public(r, hidden=()):
    return (r.get("visibility") == "public" and r.get("status") == "done" and scalar_reward(r)
            and not r.get("restricted") and r.get("id") not in hidden)


def scalar_reward(r):
    value = r.get("reward")
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def runner(r):
    value = r.get("runner") or ("mimo" if r.get("domain") in DOMAINS else "harbor")
    return {"nemo": "nemo-gym", "nemogym": "nemo-gym", "nemo_gym": "nemo-gym"}.get(value, value)


def group(r):
    return DOMAINS.get(r.get("domain")) or r.get("dataset") or "Unknown environment"


def served(r):
    return "own endpoint" if r.get("endpoint") else r.get("provider") or "auto"


def band(r):
    if r.get("status") != "done" or not scalar_reward(r):
        return "unscored"
    return "full" if r["reward"] >= .999 else "zero" if r["reward"] <= 0 else "partial"


VALUES = {"dataset": lambda r: r.get("dataset") or "", "group": group, "runner": runner,
          "model": lambda r: r.get("model") or "custom endpoint", "served": served,
          "judge": lambda r: r.get("judge") or "", "reward": band,
          "thinking": lambda r: (r.get("params") or {}).get("thinking") or "default"}


def context(r):
    return (r.get("dataset") or "Unknown environment", runner(r), r.get("domain") or "",
            r.get("reward_key") or "reward", r.get("judge") or "",
            r.get("sha") or (r.get("provenance") or {}).get("commit") or "")


def stats(runs):
    groups = defaultdict(list)
    for r in runs:
        if r.get("status") == "done" and scalar_reward(r):
            groups[(*context(r), VALUES["model"](r), served(r))].append(r["reward"])
    results = [{"dataset": k[0], "runner": k[1], "domain": k[2], "metric": k[3], "judge": k[4], "revision": k[5],
                "model": k[6], "served": k[7], "runs": len(v), "mean": math.fsum(x / len(v) for x in v), "min": min(v), "max": max(v)}
               for k, v in sorted(groups.items())]
    contexts = {context(r) for r in runs}
    return {"runs": len(runs), "scored": sum(len(v) for v in groups.values()), "datasets": len({r.get("dataset") for r in runs}),
            "tasks": len({(r.get("dataset"), r.get("path")) for r in runs}), "models": len({VALUES["model"](r) for r in runs}),
            "runners": len({runner(r) for r in runs}), "comparable": len(contexts) == 1, "results": results}


def query(everyone, filters, *, sort="new", offset=0, limit=100):
    def keep(r, skip=None):
        if filters.get("path") is not None and r.get("path") != filters["path"]:
            return False
        words = str(filters.get("q") or "").lower().split()[:16]
        haystack = " ".join(str(r.get(k) or "") for k in ("title", "task_id", "path", "dataset", "model", "runner", "adapter")).lower()
        if not all(word in haystack for word in words):
            return False
        return all(not value or name == skip or VALUES[name](r) == value
                   for name, value in filters.items() if name in VALUES)
    runs = [r for r in everyone if keep(r)]
    # Stable explicit ordering also makes pagination deterministic for tied times.
    runs.sort(key=lambda r: (-(r.get("created_at") or 0), r["id"]))
    key = {"reward_desc": lambda r: -r["reward"], "reward_asc": lambda r: r["reward"],
           "cost_asc": lambda r: (r.get("cost") or {}).get("total") or 0,
           "cost_desc": lambda r: -((r.get("cost") or {}).get("total") or 0)}.get(sort)
    if key:
        runs.sort(key=key)
    facets = {}
    for name, value in VALUES.items():
        counts = Counter(value(r) for r in everyone if keep(r, name) and value(r))
        facets[name] = [[k, n] for k, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    summary = stats(runs)
    return {"stats": summary, "facets": facets, "total": len(runs), "tasks": summary["tasks"],
            "runs": runs[max(0, offset):max(0, offset) + max(1, min(limit, 500))]}
