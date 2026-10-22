"""The ops control center's task pack (tools/build_occ_pack.py): per date its whole-day task and its crew world frozen
at the start (the units, every airport a unit may reach, the round trips of the planned rotations), with the references
the grade reads. The env never rebuilds units, so a day starts alike on any machine.

b6-v3 (`Pack3`) adds each date's passengers and its references under cost P; its world carries the passengers and reads
those references and the best plan under P in v2's places."""

import functools
import gzip
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..core.model import Airport
from ..programs.crews import Unit
from ..programs.model import Task
from ..programs.pax import Passengers

PACK = Path(__file__).resolve().parents[3] / "tasks" / "b6-v2"
PACK3 = Path(__file__).resolve().parents[3] / "tasks" / "b6-v3"


@dataclass(frozen=True)
class World:
    task: Task
    units: tuple[Unit, ...]
    airports: dict[str, Airport]
    trips: tuple[tuple[int, int], ...]
    refs: dict
    best_plan: dict
    local: Callable[[str, int], int] = field(compare=False, repr=False)
    passengers: Passengers | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        task = Task.from_dict(d["task"])
        airports = {c: Airport.from_dict(c, a) for c, a in d["airports"].items()}
        units = tuple(Unit(**{k: tuple(v) if isinstance(v, list) else v for k, v in u.items()}) for u in d["units"])
        plan = {int(k) if k != "reports" else k: v for k, v in d["best_plan"].items()}
        return cls(task, units, airports, tuple(tuple(p) for p in d["trips"]), d["refs"], plan,
                   station_clock(task, airports))

    @classmethod
    def from_v3(cls, d: dict) -> "World":
        """A b6-v3 date: its passengers, and its references under cost P where v2's were (best the stage-2 reference,
        B0 the standard crew desk with aircraft doing nothing, the bound the capacity-aware one, None where the pack
        has none)."""
        w, p = cls.from_dict(d), d["refs_p"]
        refs = {"best": p["best"], "bound": p["bound"], "gap_pct": p["gap_pct"], "best_from": p["best_from"],
                "B0": {"notice": p["candidates"]["B0"]}, "naive": p["controls"]["naive"]}
        plan = {int(k) if k != "reports" else k: v for k, v in d["best_plan_p"].items()}
        return replace(w, refs=refs, best_plan=plan, passengers=Passengers.from_dict(d["passengers"]))

    @functools.cached_property
    def partner(self) -> dict[int, int]:
        """Each flight of a round trip of the planned rotations, to the other."""
        return {a: b for x, y in self.trips for a, b in ((x, y), (y, x))}


def station_clock(task: Task, airports: dict[str, Airport]) -> Callable[[str, int], int]:
    """The task's minute on a station's clock (network.local over the pack's time zones)."""
    hub = ZoneInfo(task.tz)

    @functools.cache
    def local(station: str, minute: int) -> int:
        at = datetime.fromtimestamp(60 * (task.base + minute), UTC)
        return minute + int((at.astimezone(ZoneInfo(airports[station].tz)).utcoffset()
                             - at.astimezone(hub).utcoffset()).total_seconds()) // 60

    return local


class Pack:
    """A pack folder: its manifest and, on demand, one date's world."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(str(path or os.environ.get("AIROPS_OCC_PACK") or PACK))
        self.manifest = json.loads((self.path / "manifest.json").read_text())

    @property
    def tasks(self) -> dict[str, dict]:
        return self.manifest["tasks"]

    def world(self, task_id: str) -> World:
        if task_id not in self.tasks:
            raise ValueError(f"unknown task id {task_id!r}")
        raw = gzip.decompress((self.path / "dates" / f"{self.tasks[task_id]['date']}.json.gz").read_bytes())
        return World.from_dict(json.loads(raw))


class Pack3(Pack):
    """A b6-v3 pack folder: v2's dates with their passengers and references under cost P."""

    def __init__(self, path: Path | str | None = None):
        super().__init__(path or os.environ.get("AIROPS_OCC3_PACK") or PACK3)

    def world(self, task_id: str) -> World:
        if task_id not in self.tasks:
            raise ValueError(f"unknown task id {task_id!r}")
        raw = gzip.decompress((self.path / "dates" / f"{self.tasks[task_id]['date']}.json.gz").read_bytes())
        return World.from_v3(json.loads(raw))
