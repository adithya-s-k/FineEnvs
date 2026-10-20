"""The pack's provenance: the pins of every input (a file by its sha256, an advisory day by its parsed records'), the
manifest, and a SOURCE.md per source, all rendered from the inputs alone so a rebuild writes the same bytes."""

import json
from dataclasses import dataclass
from pathlib import Path

import airportsdata

from . import advisories, bts, fetch, fleet, pack, weather, windows

PUBLIC_DOMAIN = ("Public domain: a work of the U.S. Government, not subject to copyright in the United States "
                 "(17 U.S.C. § 105).")
IEM_TERMS = ("Public domain. The IEM disclaimer (https://mesonet.agron.iastate.edu/disclaimer.php) says its data are "
             "\"in the public domain and may be used freely by anyone for any lawful purpose\"; attributing the Iowa "
             "Environmental Mesonet of Iowa State University is appreciated. The METARs, TAFs and forecast "
             "discussions themselves are U.S. National Weather Service products (17 U.S.C. § 105).")


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    publisher: str
    url: str
    licence: str
    use: str


SOURCES = (
    Source("bts-ontime", "Reporting Carrier On-Time Performance (1987-present)", "U.S. DOT Bureau of Transportation "
           "Statistics (BTS), TranStats", "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ",
           PUBLIC_DOMAIN + " catalog.data.gov lists the dataset under the U.S. Public Domain label "
           "(http://www.usa.gov/publicdomain/label/1.0/).",
           "Every flight of the month: schedule, actual times, cancellation codes, delay causes and tail numbers. "
           "The tasks' flights, tails, frozen legs, lb, caps, nominal and the airline's actual day."),
    Source("bts-b43", "Air Carrier Financial Reports (Form 41), Schedule B-43 Inventory", "U.S. DOT Bureau of "
           "Transportation Statistics (BTS), TranStats table GEH",
           "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEH", PUBLIC_DOMAIN,
           "Each tail's model and its airline's cabin seats (inventory as of 31 December, the nearest years first)."),
    Source("bts-t100", "T-100 Domestic Segment (U.S. Carriers)", "U.S. DOT Bureau of Transportation Statistics "
           "(BTS), TranStats table FIM", "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM",
           PUBLIC_DOMAIN, "JetBlue's passengers / seats by route and month (route-months with at least "
           f"{fleet.MIN_ROUTE_DEPARTURES} departures, else the month's): pax = seats x load factor, rounded."),
    Source("faa-registry", "Releasable Aircraft Database", "Federal Aviation Administration (FAA) Civil Aviation "
           "Registry", fleet.REGISTRY, PUBLIC_DOMAIN,
           "The type of a tail B-43 misses (MASTER, then DEREG, through ACFTREF); its seats are then the family's "
           "median in B-43."),
    Source("faa-atcscc", "ATCSCC Advisories Database", "FAA Air Traffic Control System Command Center",
           "https://www.fly.faa.gov/adv/adv_list (a UTC day's list), /adv/adv_otherdis (each advisory)", PUBLIC_DOMAIN,
           "Every advisory whose control element is the hub (ground delay programs, ground stops, their "
           "revisions and cancellations, arrival delay notices), as raw text at its signature time: news kind "
           "'advisory', speaker 'FAA Command Center'. Pinned per UTC day and hub by the sha256 of the parsed records "
           "(advisory number, header, message, signature, effective time) as sorted-key compact JSON."),
    Source("iem", "IEM ASOS archive (routine METARs) and AFOS text archive (TAFs, Area Forecast Discussions)",
           "Iowa Environmental Mesonet (IEM), Iowa State University", "https://mesonet.agron.iastate.edu/",
           IEM_TERMS, "The hub's routine METARs (news kind 'metar'), its TAFs as issued with amendments ('taf') and "
           "the .AVIATION section of the local NWS office's forecast discussion ('outlook'), each at its issue time."),
    Source("airportsdata", "airportsdata", "Mike Borsetti and contributors (PyPI package)",
           "https://github.com/mborsetti/airportsdata", "MIT licence.",
           "IATA code to IANA time zone (the UTC conversion of every BTS time), airport names and coordinates."),
)


def pins(inputs: Path, months: list[str], hubs: list[str]) -> dict[str, dict]:
    """Every input of the build: a file's sha256 and size, an advisory day and hub's records digest and count."""
    out: dict[str, dict] = {}
    for path in files(inputs, months, hubs):
        out[str(path.relative_to(inputs))] = {"sha256": fetch.sha256(path), "bytes": path.stat().st_size}
    for day in advisory_days(inputs, months):
        for hub in hubs:
            records = advisories.load_day(inputs, day, hub)
            out[f"atcscc/{day}/{hub}"] = {"sha256": advisories.day_digest(records), "count": len(records)}
    return dict(sorted(out.items()))


def advisory_days(inputs: Path, months: list[str]) -> list[str]:
    """The months' UTC days whose advisory list is cached (all of them once the inputs are fetched)."""
    return [d for m in months for d in pack.month_days(m) if (advisories.day_dir(inputs, d) / "list.html").exists()]


def files(inputs: Path, months: list[str], hubs: list[str]) -> list[Path]:
    years = sorted({y for m in months for y in fleet.b43_years(m)})
    out = [bts.ontime_path(inputs, m) for m in months] + [fleet.t100_path(inputs, m) for m in months]
    out += [fleet.b43_path(inputs, y) for y in years] + [fleet.registry_path(inputs)]
    for m in months:
        for hub in hubs:
            out += [weather.metar_path(inputs, hub, m), weather.afos_path(inputs, f"TAF{hub}", m),
                    weather.afos_path(inputs, pack.HUBS[hub].afd, m)]
    return out


def verify(current: dict[str, dict], pinned: dict[str, dict]) -> list[str]:
    """The inputs whose pin changed: a rebuild from them would not reproduce the pack."""
    return [k for k, v in current.items() if k in pinned and pinned[k]["sha256"] != v["sha256"]]


def build_manifest(name: str, tasks: list[dict], tried: list[dict], pinned: dict[str, dict], log: dict,
                   tasks_sha256: str, months: list[str], hubs: list[str]) -> dict:
    splits = {s: sum(1 for t in tasks if t["split"] == s) for s in pack.SPLITS}
    reasons: dict[str, dict[str, int]] = {}
    for r in tried:
        key = r["reason"] or "kept"
        reasons.setdefault(r["split"], {})[key] = reasons.setdefault(r["split"], {}).get(key, 0) + 1
    inputs = {}
    for key, pin in pinned.items():
        rel = f"{key.rsplit('/', 1)[0]}/list.html" if key.startswith("atcscc/") else key
        inputs[key] = {**pin, **log.get(rel, {})}
    return {
        "name": name, "carrier": pack.CARRIER, "hubs": hubs, "months": months, "tasks": len(tasks),
        "splits": splits, "sha256": tasks_sha256,
        "rules": {
            "window": f"whole local hours, {windows.MIN_HOURS} to {windows.MAX_HOURS}, starting from "
                      f"{windows.FIRST_HOUR:02d}:00 to {windows.LAST_START:02d}:00 and ending by "
                      f"{windows.LAST_HOUR:02d}:00, passing the clauses. A candidate's is the one of most leverage: "
                      "the largest (naive - hindsight) / hindsight, naive the empty plan run by core.sim.execute and "
                      f"hindsight the MIP optimum (refs.mip, proven within {windows.PROVE_SECONDS:.0f} s), ties to "
                      "the earlier start, then the shorter. A control's is its disruption window: the shortest "
                      f"holding {windows.SHARE:.0%} of the most hub disruption any window passing the clauses holds "
                      f"(disruption = {windows.CANCEL_WEIGHT} per B6 hub movement cancelled with code B or C + its "
                      "weather and NAS delay minutes); then most disrupted, busiest, earliest",
            "clauses": {"tails_known": "every B6 hub movement in the window has a tail, and every tail is typed",
                        "weather_causes": f"(B+C)/(A+B+C) of the airline's cancellations in scope >= "
                                          f"{pack.MIN_WEATHER_SHARE}, vacuous with none",
                        "size": f"{pack.MIN_TAILS} to {pack.MAX_TAILS} tails"},
            "disruption": {"candidates": f"the {pack.SHORTLIST} (day, hub) of most disruption in their disruption "
                                         f"window, at least {pack.MIN_DISRUPTION}",
                           "control": "no airline cancellation in scope"},
            "days": "a day whose previous and next days are in the loaded months; controls the first hub passing "
                    "of " + "; ".join(f"{d} ({', '.join(h)})" for d, h in pack.CONTROL_DAYS)
                    + "; eval the candidates of most leverage passing every clause of scripts/references.py screen "
                    "(naive >= 10% above the anchor, the follower <= 15% above hindsight, hindsight proven), at most "
                    "one per (date, hub): " + ", ".join(f"{d} {h}" for d, h in pack.EVAL)
                    + "; train every other candidate on a date with no eval or control task",
            "news": f"advisories signed from start - {pack.NEWS_BEFORE} min, routine METARs from start - "
                    f"{pack.METAR_BEFORE} min, the TAF and forecast discussion in force at start; all up to the "
                    "latest flight's sched_dep + dmax",
            "caps": "per constrained airport (" + ", ".join(pack.CONSTRAINED) + ") the scope uses and local hour "
                    "from start to 23: max(scope's realized movements, round(B6's share of all carriers' scheduled "
                    "movements x all carriers' realized movements) - B6's realized movements outside the scope)",
            "nominal": "the scope's scheduled movements, the same airports and hours",
            "lb": "departed: max(0, DepDelay - LateAircraftDelay); cancelled: the 75th percentile of that among B6 "
                  f"departures from the origin within {pack.LIFT_WINDOW} min (else the whole day)",
            "dmax": f"max({pack.DMAX}, start - sched_dep + 60): a flight still unflown an hour into the task can leave",
        },
        "attempts": reasons,
        "summary": [pack.summary(t) for t in tasks],
        "dropped": [r for r in tried if r["split"] != "train" and not r.get("task_id")],
        "candidates": [r for r in tried if "naive_gap" in r],
        "inputs": inputs,
        "airportsdata": {"version": airportsdata.__version__, "airports.csv": airports_csv()},
    }


def airports_csv() -> str:
    return fetch.sha256(Path(airportsdata.__file__).parent / "airports.csv")


def source_docs(manifest: dict) -> dict[str, str]:
    """Each source's SOURCE.md, from the manifest."""
    inputs = manifest["inputs"]
    groups = {"bts-ontime": "bts/ontime_", "bts-b43": "bts/b43_", "bts-t100": "bts/t100d_",
              "faa-registry": "faa/", "faa-atcscc": "atcscc/", "iem": "iem/"}
    out = {}
    for s in SOURCES:
        lines = [f"# {s.title}", "", f"- Publisher: {s.publisher}", f"- URL: {s.url}", f"- Licence: {s.licence}",
                 f"- Use in {manifest['name']}: {s.use}", ""]
        if s.name == "airportsdata":
            a = manifest["airportsdata"]
            lines += [f"Pinned as the installed package version {a['version']}; sha256 of its airports.csv: "
                      f"`{a['airports.csv']}`.", ""]
        else:
            rows = {k: v for k, v in inputs.items() if k.startswith(groups[s.name])}
            dates = sorted({v["retrieved"] for v in rows.values() if "retrieved" in v})
            lines += [f"Retrieved: {', '.join(dates)}. Files are under cache/inputs/ (not in git); "
                      "`python scripts/build_pack.py` fetches what is missing and checks every pin.", ""]
            if s.name == "faa-atcscc":
                lines += ["| UTC day / hub | advisories | sha256 of the parsed records |", "|---|---|---|"]
                lines += [f"| {k.removeprefix('atcscc/')} | {v['count']} | `{v['sha256']}` |" for k, v in rows.items()]
            else:
                lines += ["| File | Bytes | sha256 | Retrieved | URL |", "|---|---|---|---|---|"]
                lines += [f"| {k} | {v['bytes']} | `{v['sha256']}` | {v.get('retrieved', '')} | {v.get('url', '')} |"
                          for k, v in rows.items()]
            lines.append("")
        out[s.name] = "\n".join(lines)
    return out


def write(out: Path, manifest: dict) -> None:
    fetch.save(out / "manifest.json", (json.dumps(manifest, indent=1) + "\n").encode())
    for name, text in source_docs(manifest).items():
        fetch.save(out / "sources" / name / "SOURCE.md", text.encode())
