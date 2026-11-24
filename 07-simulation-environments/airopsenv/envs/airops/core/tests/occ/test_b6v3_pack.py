"""b6-v3 is written from b6-v2, a reference run under P and a bound run; b6-v2 stays byte-identical: its committed
manifest is the one the plugin shipped at v0.3.3, b6-v3's names it as its base, and a v3 build leaves the base pack's
files as they were. The shipped b6-v3 has a capacity-aware bound on every date, at or under every candidate on the
fixture dates, so every day it plays gets reward_L; its split is b6-v2's, with the two controls. On the committed
fixture packs, whose dates are the packs' own."""

import gzip
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from airops_core.occ.pack import Pack, Pack3
from airops_core.programs import pax

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("build_occ_pack", ROOT / "tools" / "build_occ_pack.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
TASKS = ROOT / "tasks"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
B6_V2, B6_V3 = FIXTURES / "b6-v2", FIXTURES / "b6-v3"
MANIFEST_SHA256 = "5576545931b62dd6bd9274a966016648a51f014b54aa452367983ad401bf13b8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intact(pack: Path) -> bool:
    manifest = json.loads((pack / "manifest.json").read_text())
    return all(sha256(pack / f) == h for f, h in manifest["files"].items())


def test_b6_v2_is_byte_identical_and_a_v3_build_leaves_it_so(tmp_path):
    assert sha256(TASKS / "b6-v2" / "manifest.json") == MANIFEST_SHA256
    assert json.loads((TASKS / "b6-v3" / "manifest.json").read_text())["base_manifest_sha256"] == MANIFEST_SHA256
    shipped = json.loads((TASKS / "b6-v2" / "manifest.json").read_text())["files"]
    assert intact(B6_V2) and all(shipped[f] == h for f, h in json.loads(
        (B6_V2 / "manifest.json").read_text())["files"].items())
    before = sha256(B6_V2 / "manifest.json")
    base = Pack(B6_V2)
    tid, t = sorted(base.tasks.items())[0]
    w = base.world(tid)
    passengers = pax.book(w.task, pax.Flows.load(t["date"]), pax.Dwell.load())
    run = tmp_path / "run" / "dates"
    run.mkdir(parents=True)
    plan = {"0": {"not_before": 300}}
    (run / f"{t['date']}.json").write_text(json.dumps({
        "date": t["date"], "passengers": passengers.to_dict(),
        "reference": {"P": 100.0, "bound": 90.0, "gap_pct": 10.0, "from": "B0", "mode": "desk", "plan": plan},
        "candidates": {"B0": {"P": 100.0}, "v2_best": {"P": 120.0}}, "controls": {"naive": 130.0},
        "airline": {"P": 140.0}}))
    (tmp_path / "bound" / "dates").mkdir(parents=True)
    (tmp_path / "bound" / "dates" / f"{t['date']}.json").write_text(json.dumps({"bound": 80.0, "valid": True}))
    out = builder.assemble_v3(tmp_path / "run", tmp_path / "b6-v3", B6_V2, tmp_path / "bound")
    assert out["dates"] == 1 and len(out["skipped"]) == len(base.manifest["skipped"]) + len(base.tasks) - 1
    d = json.loads(gzip.decompress((tmp_path / "b6-v3" / "dates" / f"{t['date']}.json.gz").read_bytes()))
    v2 = json.loads(gzip.decompress((B6_V2 / "dates" / f"{t['date']}.json.gz").read_bytes()))
    assert {k: d[k] for k in v2} == v2 and d["best_plan_p"] == plan and d["refs_p"]["best_from"] == "B0"
    assert pax.Passengers.from_dict(d["passengers"]) == passengers
    assert {k: d["refs_p"][k] for k in ("bound", "gap_pct", "bound_from")} == {"bound": 80.0, "gap_pct": 20.0,
                                                                                "bound_from": "capacity-aware"}
    manifest = json.loads((tmp_path / "b6-v3" / "manifest.json").read_text())
    assert manifest["base_manifest_sha256"] == before and intact(tmp_path / "b6-v3")
    assert manifest["bound_run"] == {"name": "bound", "dates": 1}
    assert manifest["pax_sha256"] == json.loads((TASKS / "b6-v3" / "manifest.json").read_text())["pax_sha256"]
    assert Pack(tmp_path / "b6-v3").world(tid).task == w.task
    assert sha256(B6_V2 / "manifest.json") == before and intact(B6_V2)
    shutil.rmtree(tmp_path / "b6-v3")


def test_the_shipped_b6_v3_has_a_bound_under_every_candidate_on_every_date():
    shipped = json.loads((TASKS / "b6-v3" / "manifest.json").read_text())
    assert len(shipped["tasks"]) == shipped["bound_run"]["dates"] == len(shipped["files"]) == 70
    pack = Pack3(B6_V3)
    assert pack.manifest["base_manifest_sha256"] == MANIFEST_SHA256 and intact(pack.path)
    assert all(shipped["files"][f] == h for f, h in pack.manifest["files"].items()) and len(pack.tasks) == 2
    for tid, t in sorted(pack.tasks.items()):
        r = json.loads(gzip.decompress((pack.path / "dates" / f"{t['date']}.json.gz").read_bytes()))["refs_p"]
        assert r["bound_from"] == "capacity-aware" and r["bound"] <= min(r["candidates"].values()) <= r["best"] + 0.5
        assert r["gap_pct"] == round(100 * (r["best"] - r["bound"]) / r["best"], 4)
        assert pack.world(tid).refs["bound"] == r["bound"]


def test_b6_v3s_split_is_b6_v2s_with_the_controls():
    split = json.loads((TASKS / "b6-v3-split.json").read_text())
    v2 = json.loads((TASKS / "b6-v2-split.json").read_text())
    tasks = json.loads((TASKS / "b6-v3" / "manifest.json").read_text())["tasks"].values()
    assert (split["eval"], split["train"]) == (v2["eval"], v2["train"])
    assert {k: sorted(t["date"] for t in tasks if t["split"] == k) for k in ("eval", "train", "control")} == {
        "eval": sorted(split["eval"]), "train": sorted(split["train"]), "control": split["control"]}
