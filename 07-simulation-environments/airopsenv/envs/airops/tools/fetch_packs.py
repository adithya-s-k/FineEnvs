"""Fetch the days the three envs play, which this repository doesn't commit: b6-v1's tasks.jsonl (4.7 MB), b6-v2's 70
dates (6.3 MB) and b6-v3's 70 dates (10.5 MB), rebuilt byte for byte from the v1_tasks, v2_tasks and v3_tasks tables of
the dataset earakely-scale/AirOpsEnv-AgentEnv at v0.4.2, the release the v3 board played; its v1 and v2 tables are
v0.3.3's byte for byte, the release the v1 and v2 runs played. Each table is checked against its sha256 before it is
read, and each file written against the committed manifest of its pack, which pins tasks.jsonl and every date.

- fetch (the default): the nine tables into --cache (a rerun reuses them), then tasks/b6-v1/tasks.jsonl in the
  manifest's order and tasks/b6-v2/dates/<date>.json.gz and tasks/b6-v3/dates/<date>.json.gz as the packs' builder
  writes them (gzip, mtime 0).
- verify: the packs on disk against their manifests; exit 1 on a mismatch.

Run from envs/airops:

    uv run -p 3.12 --with pyarrow python tools/fetch_packs.py [fetch|verify] [--cache DIR]
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "tasks"
V1, V2, V3 = TASKS / "b6-v1", TASKS / "b6-v2", TASKS / "b6-v3"
REPO = "earakely-scale/AirOpsEnv-AgentEnv"
REVISION = "781bca423300c8fb921eead316a9120291d536cb"
"""The dataset's commit tagged v0.4.2."""
TABLES = {
    "tasks/v1-eval.parquet": "a925fa7d95dc86365b72931e2d9baf8b27ae7972bf4fc1f648d5dc5f01cc8de4",
    "tasks/v1-control.parquet": "34a6b874734df8ac58f2df097752d32f4acbf713a492562b5a2b4b77fd9c6876",
    "tasks/v1-train.parquet": "f889162242d01a461aecc6a8d3e58d13c0e33cb0790dc21652570ee121f153d6",
    "tasks/v2-eval.parquet": "b24cdeead8ddaaacfd71eb47876a416f2f7254fc0e3a932e09d2c3fe3658e751",
    "tasks/v2-control.parquet": "45deb62d949276e9792dd89e826e9f671a43ac8ff8b65b362dea9aadc2c82748",
    "tasks/v2-train.parquet": "65b102b58ba878292d5804af8b8daaa709483f1d92517684e2ef56cfbeb404a2",
    "tasks/v3-eval.parquet": "ca1d217bfdc1103ae4ffa13510a219b9a22c1283853859c80ee40943f1050316",
    "tasks/v3-control.parquet": "b14b4a9da92dfccfdf9992549d3b96d5ed4f2170d528524d7a0f178a1b714272",
    "tasks/v3-train.parquet": "518d802e5611aa76abe62148fd0401c966146b3334bd7ce1cc058293eedd7726",
}
"""The v1 and v2 tables are the six v0.3.3 pinned, unchanged at v0.4.2."""
DATED = {"b6-v2": V2, "b6-v3": V3}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(path: str, columns: list[str], cache: Path) -> list[dict]:
    """A table's rows, downloaded into the cache once and checked against its pin. Only fetching reads the tables, so
    pyarrow is imported here: problems(), which rollout.py runs, needs nothing outside the standard library."""
    import pyarrow.parquet as pq

    local = cache / REVISION[:12] / path
    if not (local.is_file() and sha256(local.read_bytes()) == TABLES[path]):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        with urllib.request.urlopen(url, timeout=300) as response:
            data = response.read()
        if sha256(data) != TABLES[path]:
            sys.exit(f"{url}: sha256 {sha256(data)}, pinned {TABLES[path]}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return pq.read_table(local, columns=columns).to_pylist()


def v1_body(cache: Path) -> bytes:
    lines = {r["task_id"]: r["task"] for split in ("eval", "control", "train")
             for r in rows(f"tasks/v1-{split}.parquet", ["task_id", "task"], cache)}
    order = [s["task_id"] for s in manifest(V1)["summary"]]
    return "".join(lines[t] + "\n" for t in order).encode()


def dates(version: str, cache: Path) -> dict[str, bytes]:
    out = {}
    for split in ("eval", "control", "train"):
        for r in rows(f"tasks/{version}-{split}.parquet", ["date", "world"], cache):
            name = f"{r['date']}.json.gz"
            buffer = io.BytesIO()
            with gzip.GzipFile(filename=name, mode="wb", fileobj=buffer, mtime=0) as fh:
                fh.write(r["world"].encode())
            out[f"dates/{name}"] = buffer.getvalue()
    return out


def manifest(pack: Path) -> dict:
    return json.loads((pack / "manifest.json").read_text())


def problems() -> list[str]:
    out = []
    tasks = V1 / "tasks.jsonl"
    if not tasks.is_file():
        out.append("b6-v1/tasks.jsonl: missing")
    elif sha256(tasks.read_bytes()) != manifest(V1)["sha256"]:
        out.append("b6-v1/tasks.jsonl: differs from its manifest")
    for name, pack in DATED.items():
        pins = manifest(pack)["files"]
        missing = [r for r in pins if not (pack / r).is_file()]
        if missing:
            out.append(f"{name}: {len(missing)} of {len(pins)} dates missing")
        out += [f"{name}/{r}: differs from its manifest" for r, pin in pins.items()
                if r not in missing and sha256((pack / r).read_bytes()) != pin]
    return out


def fetch(args) -> int:
    body = v1_body(args.cache)
    if sha256(body) != manifest(V1)["sha256"]:
        sys.exit("b6-v1/tasks.jsonl: the dataset's tables don't rebuild the pinned file")
    built = {}
    for name, pack in DATED.items():
        built[pack] = dates(name.removeprefix("b6-"), args.cache)
        pins = manifest(pack)["files"]
        if wrong := sorted(set(pins) ^ set(built[pack])) + sorted(r for r in built[pack]
                                                                    if sha256(built[pack][r]) != pins.get(r)):
            sys.exit(f"{name}: the dataset's tables don't rebuild {', '.join(wrong)}")
    (V1 / "tasks.jsonl").write_bytes(body)
    for pack, files in built.items():
        (pack / "dates").mkdir(exist_ok=True)
        for rel, data in files.items():
            (pack / rel).write_bytes(data)
    print(f"b6-v1: {len(manifest(V1)['summary'])} days in tasks.jsonl; b6-v2: {len(built[V2])} dates; b6-v3: "
          f"{len(built[V3])} dates; all match their manifests")
    return verify(args)


def verify(args) -> int:
    found = problems()
    print("\n".join(found + ["fetch them with: uv run -p 3.12 --with pyarrow python tools/fetch_packs.py"] if found
                    else ["b6-v1, b6-v2 and b6-v3 match their manifests"]))
    return 1 if found else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch", "verify"])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "airopsenv",
                        help="where the dataset's tables are kept (default ~/.cache/airopsenv)")
    args = parser.parse_args()
    raise SystemExit(fetch(args) if args.command == "fetch" else verify(args))


if __name__ == "__main__":
    main()
