"""Fetch the days the two envs play, which this repository doesn't commit: b6-v1's tasks.jsonl (4.5 MB) and b6-v2's 70
dates (6.2 MB), rebuilt byte for byte from the v1_tasks and v2_tasks tables of the dataset
earakely-scale/AirOpsEnv-AgentEnv at v0.3.3, the release the published runs played. Each table is checked against its
sha256 before it is read, and each file written against the committed manifest of its pack, which pins tasks.jsonl and
every date.

- fetch (the default): the six tables into --cache (a rerun reuses them), then tasks/b6-v1/tasks.jsonl in the
  manifest's order and tasks/b6-v2/dates/<date>.json.gz as the pack's builder writes them (gzip, mtime 0).
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

import pyarrow.parquet as pq

TASKS = Path(__file__).resolve().parents[1] / "tasks"
V1, V2 = TASKS / "b6-v1", TASKS / "b6-v2"
REPO = "earakely-scale/AirOpsEnv-AgentEnv"
REVISION = "5e455a44e10d11712464e6fdf76d6b49078d4d5c"
"""The dataset's commit tagged v0.3.3."""
TABLES = {
    "tasks/v1-eval.parquet": "a925fa7d95dc86365b72931e2d9baf8b27ae7972bf4fc1f648d5dc5f01cc8de4",
    "tasks/v1-control.parquet": "34a6b874734df8ac58f2df097752d32f4acbf713a492562b5a2b4b77fd9c6876",
    "tasks/v1-train.parquet": "f889162242d01a461aecc6a8d3e58d13c0e33cb0790dc21652570ee121f153d6",
    "tasks/v2-eval.parquet": "b24cdeead8ddaaacfd71eb47876a416f2f7254fc0e3a932e09d2c3fe3658e751",
    "tasks/v2-control.parquet": "45deb62d949276e9792dd89e826e9f671a43ac8ff8b65b362dea9aadc2c82748",
    "tasks/v2-train.parquet": "65b102b58ba878292d5804af8b8daaa709483f1d92517684e2ef56cfbeb404a2",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(path: str, columns: list[str], cache: Path) -> list[dict]:
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


def v2_dates(cache: Path) -> dict[str, bytes]:
    out = {}
    for split in ("eval", "control", "train"):
        for r in rows(f"tasks/v2-{split}.parquet", ["date", "world"], cache):
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
    pins = manifest(V2)["files"]
    missing = [r for r in pins if not (V2 / r).is_file()]
    if missing:
        out.append(f"b6-v2: {len(missing)} of {len(pins)} dates missing")
    out += [f"b6-v2/{r}: differs from its manifest" for r, pin in pins.items()
            if r not in missing and sha256((V2 / r).read_bytes()) != pin]
    return out


def fetch(args) -> int:
    body = v1_body(args.cache)
    if sha256(body) != manifest(V1)["sha256"]:
        sys.exit("b6-v1/tasks.jsonl: the dataset's tables don't rebuild the pinned file")
    dates = v2_dates(args.cache)
    pins = manifest(V2)["files"]
    if wrong := sorted(set(pins) ^ set(dates)) + sorted(r for r in dates if sha256(dates[r]) != pins.get(r)):
        sys.exit(f"b6-v2: the dataset's tables don't rebuild {', '.join(wrong)}")
    (V1 / "tasks.jsonl").write_bytes(body)
    (V2 / "dates").mkdir(exist_ok=True)
    for rel, data in dates.items():
        (V2 / rel).write_bytes(data)
    print(f"b6-v1: {len(manifest(V1)['summary'])} days in tasks.jsonl; b6-v2: {len(dates)} dates; all match their "
          "manifests")
    return verify(args)


def verify(args) -> int:
    found = problems()
    print("\n".join(found + ["fetch them with: uv run -p 3.12 --with pyarrow python tools/fetch_packs.py"] if found
                    else ["b6-v1 and b6-v2 match their manifests"]))
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
