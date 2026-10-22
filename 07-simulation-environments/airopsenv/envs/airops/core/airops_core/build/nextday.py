"""The following days' published schedules for passengers JetBlue can't get there the same day (programs.pax.Seat):
for each of the `days` days after the task's date, every flight BTS on-time data has the carrier scheduled that day,
cancelled ones included, in minutes from the task date's midnight, each with its seats left after its booked
passengers: its tail's seats (Form 41 B-43, else the FAA registry; the day's median where neither has it) less
seats x its T-100 route-month load factor, as the task's own flights are loaded (fleet.pax), its B-43 years the
build's (candidates.b43_years: 2026 has no published inventory). A day whose month's on-time or T-100 file the inputs
lack repeats the latest day built, shifted a day per day missing (an assumption, for dates near the end of the
published data)."""

import statistics
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from ..programs import candidates
from ..programs.model import Task
from ..programs.pax import Seat
from . import bts, fleet

DAYS = 5


def seats(task: Task, inputs: Path, derived: Path, days: int = DAYS) -> tuple[Seat, ...]:
    out: list[Seat] = []
    last: list[Seat] = []
    shift = 0
    for k in range(1, days + 1):
        day = (date.fromisoformat(task.date) + timedelta(days=k)).isoformat()
        if bts.ontime_path(inputs, day[:7]).exists() and fleet.t100_path(inputs, day[:7]).exists():
            last, shift = scheduled(task, day, inputs, derived), 0
        elif last:
            shift += 1440
        else:
            raise FileNotFoundError(f"no on-time or T-100 data for {day}, the day after {task.date}")
        out += [replace(s, flight=f"{s.flight} +{k}", dep=s.dep + shift, arr=s.arr + shift, day=k) for s in last]
    return tuple(out)


def scheduled(task: Task, day: str, inputs: Path, derived: Path) -> list[Seat]:
    month = day[:7]
    df = bts.read_ontime(bts.ontime_path(inputs, month), derived)
    df = df[(df.Reporting_Airline == task.carrier) & (df.FlightDate == day) & df.sched_dep.notna()]
    craft = candidates.typed(set(df.Tail_Number.dropna()), candidates.b43_years(month), inputs)
    median = int(statistics.median(a.seats for a in craft.values()))
    lf = fleet.load_factors(task.carrier, month, inputs)
    out = []
    for r in df.assign(num=df.Flight_Number_Reporting_Airline.astype(int)).sort_values(
            ["sched_dep", "num", "Origin"]).itertuples():
        n = craft[r.Tail_Number].seats if isinstance(r.Tail_Number, str) and r.Tail_Number in craft else median
        booked = fleet.pax(n, lf, r.Origin, r.Dest)
        out.append(Seat(f"{task.carrier} {r.num} {r.Origin}", r.Origin, r.Dest, int(r.sched_dep) - task.base,
                        int(r.sched_arr) - task.base, max(0, n - booked), 1, booked))
    return out
