"""The pack builder's assembly: a build's dates and worlds become one file per date with its task and split, under a
manifest with every file's sha256; the dates it failed on and the date outside the pool by rule are listed as
skipped."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from occ_helpers import storm

from airops_core.occ.pack import Pack

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("build_occ_pack", ROOT / "tools" / "build_occ_pack.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_a_build_assembles_into_a_pack_the_env_loads(tmp_path):
    w = storm()
    build, tasks = tmp_path / "build", tmp_path / "tasks"
    for sub in ("dates", "worlds"):
        (build / sub).mkdir(parents=True)
    tasks.mkdir()
    (tasks / "tasks.jsonl").write_text(w.task.to_json() + "\n")
    world = {"date": w.task.date, "task_id": w.task.task_id, "units": [u.to_dict() for u in w.units],
             "airports": {c: a.to_dict() for c, a in w.airports.items()}, "trips": [[0, 1]], "refs": w.refs,
             "best_plan": {"0": {"cancel": True, "decided": 0}, "reports": {}}}
    (build / "worlds" / f"{w.task.date}.json.gz").write_bytes(gzip.compress(json.dumps(world).encode()))
    refs = {**w.refs, "best_from": "run"}
    (build / "dates" / f"{w.task.date}.json").write_text(json.dumps({"date": w.task.date, "task_id": w.task.task_id,
                                                                     "split": "eval", "refs": refs}))
    (build / "dates" / "2023-01-02.json").write_text(json.dumps({"date": "2023-01-02", "error": "Traceback"}))
    (build / "summary.json").write_text(json.dumps({"commit": "abc", "inputs": {"tasks": str(tasks)}}))
    out = builder.assemble(build, tmp_path / "pack")
    assert out["dates"] == 1 and out["skipped"] == {"2023-01-02": "error in the build", "2023-06-02": (
        "its joint search crashed in the crew run; outside the pool by rule")}
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    path = tmp_path / "pack" / "dates" / f"{w.task.date}.json.gz"
    assert manifest["files"] == {f"dates/{w.task.date}.json.gz": hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = Pack(tmp_path / "pack").world(w.task.task_id)
    assert loaded.task.split == "eval" and loaded.units == w.units and loaded.best_plan == {0: {"cancel": True,
                                                                                               "decided": 0},
                                                                                           "reports": {}}
    assert builder.names({1: {"crew": {"pilot": "P1"}, "deadhead": ["C2"]}, "reports": {"R#2": 600}}) == {
        "P1", "C2", "R"}
