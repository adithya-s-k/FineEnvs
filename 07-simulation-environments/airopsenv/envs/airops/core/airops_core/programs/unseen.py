"""The unseen arrivals e: traffic BTS never sees (foreign-flag, cargo, charter and non-reporting carriers) plus UDP's
pop-up reserve, as exempt arrivals per hour in every GDP's ration-by-schedule list.

Per advisory, e is the in-sample rate that makes RBS over the demand as listed when the advisory was signed (cancelled
flights gone NOTICE before their scheduled departure, every carrier alike) reproduce its stated AVERAGE DELAY times one
scale per airport, averaged over the flights it controls (past the period the list spills on at the last hour's rate).
The scale sits on the stated delay, not on e: near the rate the delay is steep in e, so a scale on e moves a busy
program's delay far more than a light one's. An advisory the form can't match within FIT_TOLERANCE takes its
airport's median. The stated delay is known at issuance, so the truth and the forecast share the same e. The scale is
fitted on train days only to the other carriers' observed program delay: the BTS NASDelay of their controlled arrivals
that day. JetBlue's flights never enter the fit, and nothing is fitted to JetBlue's day or its replay. Fidelity is the
modelled over the observed other carrier program delay of one day-hub, the clause asking 0.67 to 1.5. Both sides are
measured alike: a diverted arrival, which BTS gives no NAS delay, is left out, and a flight's modelled arrival is held,
as the truth holds JetBlue's, by any ground stop at the hub covering its modelled departure.

The movement-count check sets the fitted stream against what the airports count: the unseen share of program-hour
arrivals (e against e plus BTS demand) beside the share of each airport's own arrivals BTS misses (JFK: the Port
Authority's Airport Traffic Report 2024, 468,570 movements; BOS: Massport's monthly inbound flights, all users).
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .advisories import GDP, PROPOSED, Advisory, Run, Stop, as_ran, held_until
from .model import Arrival, Task
from .rbs import Covers, assign, exempt, segment

E_MAX = 60
FIT_TOLERANCE = 15
GRID = tuple(round(0.025 * k, 3) for k in range(81))
FIDELITY = (0.67, 1.5)
FLOOR = 0.01
JFK_MOVEMENTS_2024 = 468_570
"""Port Authority of NY & NJ, 2024 Airport Traffic Report: JFK's plane movements (arrivals and departures), 2024."""
MASSPORT = "https://www.massport.com/sites/default/files/"
MASSPORT_ARRIVALS = {
    "2022-12": (15_066, "2023-09/1222-avstats-airport-traffic-summary.pdf"),
    "2023-06": (17_036, "2023-09/0623-avstats-airport-traffic-summary.pdf"),
    "2023-07": (17_061, "2023-09/bos-about-aviation-stats-2023-07.pdf"),
    "2023-08": (18_046, "2023-10/0823%20avstats%20airport%20traffic%20summary.pdf"),
    "2023-12": (15_211, "2024-01/avstats-airport-traffic-summary-dec23.pdf"),
    "2024-01": (13_853, "2024-02/0124-avstats-airport-traffic-summary.pdf"),
    "2024-02": (13_938, "2024-03/0224-avstats-airport-traffic-summary.pdf"),
    "2024-06": (18_395, "2024-07/0624-avstats-airport-traffic-summary.pdf"),
    "2024-07": (19_130, "2024-09/0724-avstats-airport-traffic-summary.pdf"),
    "2024-08": (19_386, "2024-09/0824-avstats-airport-traffic-summary.pdf"),
    "2025-01": (14_748, "2025-02/0125-avstats-airport-traffic-summary.pdf"),
    "2025-02": (13_761, "2025-03/0225-avstats-airport-traffic-summary.pdf"),
    "2025-07": (19_031, "2025-08/0725-avstats-airport-traffic-summary.pdf"),
    "2025-08": (19_206, "2025-09/0825-avstats-airport-traffic-summary.pdf"),
    "2025-12": (15_637, "2026-01/1225-avstats-airport-traffic-summary.pdf"),
    "2026-01": (13_553, "2026-02/JAN-2026-Air-Traffic-Report.pdf"),
    "2026-02": (13_128, "2026-04/0226-Air-Traffic-Report.pdf"),
    "2026-06": (18_473, "2026-07/0626-Air-Traffic-Report.pdf"),
    "2026-07": (19_224, "2026-08/0726AirTrafficReport.pdf"),
}
"""Massport's Monthly Airport Traffic Summary for Logan: Total Arriving (InBound) Flights, all users, with each month's
report under MASSPORT (pinned in inputs/massport/; Massport serves them only to a browser, so they were saved from one
through fetch.save and fetch.record)."""


@dataclass(frozen=True)
class Fit:
    """`curve` is the RBS average delay at issuance for e = 0..E_MAX; `e`, `modelled` and `matched` read it at the
    stated delay itself (scale 1)."""

    advisory: str
    hub: str
    e: float
    modelled: float
    stated: int
    matched: bool
    curve: tuple[float, ...]


def in_sample(adv: Advisory, demand: Sequence[Arrival], covers: Covers, stops: Sequence[Stop]) -> Fit:
    """The e in 0..E_MAX whose RBS average delay at issuance is nearest the stated AVERAGE DELAY (ties to the lower)."""
    run = Run(adv, adv.issued, adv.end + 1)
    curve = tuple(average_delay(run, demand, covers, e, stops) for e in range(E_MAX + 1))
    e = nearest(curve, adv.avg_delay)
    return Fit(adv.id, adv.hub, float(e), round(curve[e], 1), adv.avg_delay,
               abs(curve[e] - adv.avg_delay) <= FIT_TOLERANCE, curve)


def nearest(curve: Sequence[float], target: float) -> int:
    return min(range(len(curve)), key=lambda e: (abs(curve[e] - target), e))


def average_delay(run: Run, demand: Sequence[Arrival], covers: Covers, e: float, stops: Sequence[Stop]) -> float:
    got = assign(run, demand, covers, e, stops, known=run.adv.issued)
    delays = [s - demand[i].sched_arr for i, s in got.items() if not exempt(demand[i], run, covers, stops)]
    return statistics.mean(delays) if delays else 0.0


def fittable(adv: Advisory) -> bool:
    return (adv.kind in (GDP, PROPOSED) and adv.binds and bool(adv.rates) and adv.avg_delay is not None
            and adv.start is not None and adv.end is not None)


Table = tuple[dict[str, float], dict[str, float]]
"""Each advisory's e and each airport's median e at one scale per airport, as `table` gives them."""


def table(fits: Iterable[Fit], scale: Mapping[str, float] | None = None) -> Table:
    """Each advisory's e at its airport's scale (1 without one): the e whose average delay at issuance is nearest the
    scale times the stated AVERAGE DELAY, its airport's median of the matched ones where none is within
    FIT_TOLERANCE; and that median for every airport fitted or scaled."""
    at = {}
    for f in fits:
        target = f.stated * (1.0 if scale is None else scale[f.hub])
        e = nearest(f.curve, target)
        at[f.advisory] = (f.hub, float(e), abs(f.curve[e] - target) <= FIT_TOLERANCE)
    hubs = {h for h, _, _ in at.values()} | set(scale or ())
    median = {hub: statistics.median([e for h, e, ok in at.values() if h == hub and ok] or [0.0])
              for hub in sorted(hubs)}
    return {a: e if ok else median[h] for a, (h, e, ok) in at.items()}, median


def fit_scale(tasks: Iterable[Task], fits: Iterable[Fit], grid: Sequence[float] = GRID) -> dict[str, float]:
    """One scale per airport, on the given (train) tasks storming there (Task.storms): the grid value whose fidelity
    is nearest 1 in the least-squares log sense across days (ties to the smaller scale)."""
    fits = list(fits)
    hubs = {f.hub for f in fits}
    by_hub = defaultdict(list)
    for t in tasks:
        for h in t.storms:
            by_hub[h].append(t)
    tables = {s: table(fits, dict.fromkeys(hubs | set(by_hub), s)) for s in grid}
    out = {}
    for hub, ts in sorted(by_hub.items()):
        rows = {s: [program_delay(t, tables[s], hub) for t in ts] for s in grid}
        loss = {s: sum(math.log(max(m / o, FLOOR)) ** 2 for m, o, _ in r if o > 0) for s, r in rows.items()}
        out[hub] = min(grid, key=lambda s: (round(loss[s], 9), s))
    return out


def fidelity(task: Task, at: Table | None = None, hub: str | None = None) -> float | None:
    modelled, observed, _ = program_delay(task, at, hub)
    return modelled / observed if observed > 0 else None


def program_delay(task: Task, at: Table | None = None, hub: str | None = None) -> tuple[float, float, int]:
    """(modelled, observed, flights): the other carriers' arrivals at `hub` (the task's by default) scheduled on its
    date, flown to it and controlled by a GDP segment, their modelled arrival delay against their BTS NASDelay; `at`
    overrides the hub's e. The modelled arrival is the RBS slot, held like the truth's flights: a flight leaves at the
    slot less its scheduled block, not before its scheduled departure, and waits out any stop at the hub covering its
    origin."""
    hub = hub or task.hub
    p = task.programs[hub]
    e_fit, e_median = (p.e_fit, p.e_median) if at is None else (at[0], at[1][hub])
    runs = {r.adv.id: r for r in as_ran(p.advisories)[0]}
    rows = defaultdict(list)
    for i, a in enumerate(p.demand):
        if (a.controlled in runs and a.carrier != task.carrier and not a.cancelled and not a.diverted
                and 0 <= a.sched_arr < 1440):
            rows[a.controlled].append(i)
    modelled = observed = 0.0
    for adv, idx in rows.items():
        _, got = segment(runs[adv], p.demand, task.covers, e_fit.get(adv, e_median), p.stops)
        for i in idx:
            a = p.demand[i]
            block = a.sched_arr - a.sched_dep
            leaves = held_until(p.stops, task.covers, a.orig, max(a.sched_dep, got[i] - block))
            modelled += max(got[i], leaves + block) - a.sched_arr
            observed += a.nas
    return modelled, observed, sum(len(v) for v in rows.values())


def unseen_share(tasks: Iterable[Task]) -> dict[str, dict]:
    """Per hub, the unseen stream's share of program-hour demand over the truth segments of the given tasks' storm
    hubs."""
    acc = defaultdict(lambda: [0.0, 0])
    for t in tasks:
        for hub in t.storms:
            p = t.programs[hub]
            for seg in p.segments:
                hours = (seg.end + 1 - seg.start) / 60
                acc[hub][0] += seg.e * hours
                acc[hub][1] += sum(1 for a in p.demand if seg.start <= a.sched_arr <= seg.end)
    return {h: {"unseen": round(u), "bts": n, "share": round(u / (u + n), 3) if u + n else None}
            for h, (u, n) in sorted(acc.items())}


def missed_share(hub: str, bts_arrivals_per_day: float, months: Sequence[str],
                 days: Mapping[str, int]) -> dict[str, float]:
    """The share of the airport's own counted arrivals BTS does not report, from JFK's 2024 movements or Logan's
    monthly inbound flights over `months` (with `days` per month)."""
    if hub == "JFK":
        counted = JFK_MOVEMENTS_2024 / 2 / 366
    else:
        counted = sum(MASSPORT_ARRIVALS[m][0] for m in months) / sum(days[m] for m in months)
    bts = float(bts_arrivals_per_day)
    return {"counted_per_day": round(counted, 1), "bts_per_day": round(bts, 1),
            "missed_share": round(1 - bts / counted, 3)}
