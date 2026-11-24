import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from airops_core.refs import mip

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "references.py"
KEYS = ["floor", "hindsight", "anchor", "follower", "blind", "naive", "airline", "gap_pct", "qualifies", "clauses",
        "hindsight_plan", "follower_plan"]


@pytest.fixture(scope="module")
def refs():
    spec = importlib.util.spec_from_file_location("references_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_fixture_carries_a_full_reference_block(pack):
    for task in pack:
        ref = task.reference
        assert list(ref) == KEYS
        assert ref["anchor"] == ref["hindsight"] and ref["floor"] <= ref["hindsight"]
        assert ref["qualifies"] == all(c["ok"] for c in ref["clauses"].values())
        assert set(ref["hindsight_plan"]) == set(ref["follower_plan"]) == {str(f.id) for f in task.flights}


def test_the_clauses(refs, by_id):
    task = by_id["b6-jfk-20230626-1400"]
    ref = task.reference
    proven = mip.Solution("Optimal", ref["hindsight"], ref["hindsight"], 0.0, 1.0, {}, 0, 0)
    checks = refs.clauses(task, ref, proven)
    assert checks == ref["clauses"] and all(c["ok"] for c in checks.values())
    assert checks["tails"]["value"] == 50 and checks["weather_atc"]["value"] >= 0.8
    assert checks["naive_over_anchor"]["value"] == round(ref["naive"] / ref["anchor"] - 1, 4)
    slow = replace(proven, wall_s=61.0)
    assert not refs.clauses(task, ref, slow)["hindsight_proven"]["ok"]
    loose = replace(proven, status="Time limit reached", gap_pct=0.5)
    assert not refs.clauses(task, ref, loose)["hindsight_proven"]["ok"]
    worse = dict(ref, follower=round(ref["hindsight"] * 1.2))
    assert not refs.clauses(task, worse, proven)["follower_over_hindsight"]["ok"]
    normal = by_id["b6-jfk-20230610-1400"].reference
    assert not normal["qualifies"] and not normal["clauses"]["naive_over_anchor"]["ok"]
    small = by_id["b6-jfk-20230626-1400-1700"].reference["clauses"]["tails"]
    assert small == {"value": 28, "ok": False}
