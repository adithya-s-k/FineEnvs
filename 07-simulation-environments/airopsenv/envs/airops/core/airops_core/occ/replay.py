"""A recorded ops control center day played again: every call the env logged, in the order it received them, through
a fresh OccDay on the same pack, each result checked byte for byte against its recorded digest, the FAA advisories the
triggers delivered (or the env, late) at their places, and the rest of the day as end-day ran it."""

from dataclasses import dataclass

from .day import OccDay, digest
from .pack import World

ROLE = {"ops": "ops", "crew": "crew", "one": "both"}


@dataclass(frozen=True)
class Replayed:
    day: OccDay
    mismatches: list[dict]

    @property
    def exact(self) -> bool:
        return not self.mismatches


def replay(world: World, data: dict, make: type[OccDay] = OccDay) -> Replayed:
    """`data` is the env's data/get record of a day; `make` the day class to play it on."""
    day = make(world, data["mode"])
    events = [(c["seq"], "call", c) for c in data["calls"]] + \
        [(n["seq"], "notice", n) for n in data["notices"] if n["via"] in ("trigger", "late")]
    out = []
    for _, kind, x in sorted(events, key=lambda e: e[0]):
        if kind == "notice":
            day.notice(x["news_id"], x["via"])
            continue
        result = day.call(ROLE[x["desk"]], x["tool"], x["args"])
        if digest(result) != x["result"]:
            out.append({"seq": x["seq"], "desk": x["desk"], "tool": x["tool"], "watch": x["watch"]})
    if data["done"] and not day.done:
        day.end()
    if data.get("digest") and day.done and day.digest() != data["digest"]:
        out.append({"digest": day.digest(), "recorded": data["digest"]})
    return Replayed(day, out)
