"""Replay the published runs through this code: every scored v1 run (a sweep folder with sweep.json), every v2 attempt
that recorded its day (a pilot folder with pilot.json) and every v3 one (a pilot folder named occ3-*, as the plugin's
dataset build reads them) under RUNS, the dataset's runs/ at v0.4.2 or the plugin's results/runs. A v1 run's tool calls
go through a fresh Day in order, each output compared byte for byte with its transcript, and the day the env then grades
is compared with the one its verifier read (from the dataset's raw/airops.jsonl, when --raw names it); a v2 attempt's
calls go through a fresh OccDay on b6-v2 and a v3 attempt's through a fresh OccDay3 on b6-v3, each result compared with
its recorded digest, and the whole record the env served (grade, audit, digest, v3's passengers) compared with the
recorded one. Prints one line per run, naming what differs; exit 1 on any difference.

Run from envs/airops on the fetched packs:

    uv run --project agentenv python tools/replay_runs.py RUNS [--raw RAW_JSONL] [--workers N] [--only v1,v2,v3]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from airops_core.core.model import load_pack
from airops_core.occ.day import OccDay
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack, Pack3
from airops_core.occ.replay import replay as occ_replay

from airops_agentenv.live import PACK
from airops_agentenv.replay import ReplayError
from airops_agentenv.replay import replay as day_replay

VERSIONS = {"v2": (Pack, OccDay), "v3": (Pack3, OccDay3)}
PARTS = ("digest", "grade", "audit", "passengers")


def jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def v1_runs(root: Path, graded: dict[tuple[str, str], dict]) -> list[str]:
    days = {t.task_id: t for t in load_pack(Path(str(PACK)))}
    out = []
    for sweep in sorted(p.parent for p in root.glob("*/sweep.json")):
        for line in (sweep / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["outcome"] != "scored":
                continue
            name = f"{sweep.name}/{row['task_id']} {row['model']}"
            record = json.loads((sweep / row["transcript"]).read_text())
            try:
                played = day_replay(days[row["task_id"]], record["messages"])
            except ReplayError as e:
                out.append(f"DIFF {name}: {e}")
                continue
            data = jsonable(played.day.data())
            problems = [] if data["reward"] == row["reward"] else [f"reward {data['reward']}, scored {row['reward']}"]
            verified = graded.get((row["task_id"], row["instance"].rsplit("-", 1)[-1]))
            if verified is not None and data != verified:
                problems.append("the graded day differs from the one the verifier read")
            outputs = sum(1 for s in played.steps if "error" not in s["result"])
            out.append(f"{'DIFF' if problems else 'same'} {name}: {outputs} outputs, reward {data['reward']}"
                       + (", graded day as the verifier read it" if verified is not None and not problems else "")
                       + (f"; {'; '.join(problems)}" if problems else ""))
    return out


def occ_attempt(job: tuple[str, str, str, str]) -> str:
    version, name, day_path, pack = job
    make_pack, make_day = VERSIONS[version]
    recorded = json.loads(Path(day_path).read_text())
    again = occ_replay(make_pack(pack).world(recorded["task_id"]), recorded, make_day)
    data = jsonable(again.day.data())
    same = [k for k in PARTS if k in recorded and data.get(k) == recorded[k]]
    problems = [f"{len(again.mismatches)} mismatches, first {again.mismatches[:2]}"] if again.mismatches else []
    if data != recorded:
        keys = sorted(k for k in set(data) | set(recorded) if data.get(k) != recorded.get(k))
        problems.append(f"the record differs in {', '.join(keys)}")
    grade = recorded.get("grade") or {}
    return (f"{'DIFF' if problems else 'same'} {name}: {len(recorded['calls'])} calls, reward {grade.get('reward')}, "
            f"reward_L {grade.get('reward_L')}, {' '.join(same)} equal"
            + (f"; {'; '.join(problems)}" if problems else ""))


def occ_jobs(root: Path, packs: dict[str, str], versions: set[str]) -> list[tuple[str, str, str, str]]:
    jobs = []
    for pilot in sorted(p.parent for p in root.glob("*/pilot.json")):
        version = "v3" if pilot.name.startswith("occ3-") else "v2"
        if version not in versions:
            continue
        for line in (pilot / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("day"):
                jobs.append((version, f"{pilot.name}/{row['run']}-a{row['attempt']} {row['outcome']}",
                             str(pilot / row["day"]), packs[version]))
    return jobs


def verifier_days(raw: Path | None) -> dict[tuple[str, str], dict]:
    """Each v1 run's day as its verifier read it from the env, by task id and instance suffix."""
    out = {}
    for line in (raw.read_text().splitlines() if raw else []):
        run = json.loads(line)
        episode = run["record"]["metadata"]["verifications"]["airops"]["results"][0]["episode"]
        out[episode["task_id"], run["episode_id"].rsplit("-", 1)[-1]] = episode
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path)
    parser.add_argument("--raw", type=Path, help="the dataset's raw/airops.jsonl: each v1 run as its verifier read it")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", default="v1,v2,v3", help="the versions to replay (default all three)")
    args = parser.parse_args()
    versions = set(args.only.split(","))
    lines = v1_runs(args.runs, verifier_days(args.raw)) if "v1" in versions else []
    print("\n".join(lines), flush=True)
    packs = {v: str(make().path) for v, (make, _) in VERSIONS.items() if v in versions}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        for line in pool.map(occ_attempt, occ_jobs(args.runs, packs, versions)):
            print(line, flush=True)
            lines.append(line)
    differ = sum(line.startswith("DIFF") for line in lines)
    print(f"{len(lines) - differ} of {len(lines)} runs replay the same")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
