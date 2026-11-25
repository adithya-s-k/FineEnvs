"""Frozen evaluation sets: deterministic selection, tamper detection, and serving as a split."""

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURE_RELEASE
from retroenv.benchmark import Benchmark
from retroenv.evalsets import EvalsetStore, load_evalset, make_evalset, select_tiered
from retroenv.store import TaskStore

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {"easy": 2, "medium": 2, "hard": 3}


def fixture_candidates():
    store = TaskStore(FIXTURE_RELEASE / "tasks-private", FIXTURE_RELEASE / "stocks")
    return store, [t for split in ("test_id", "test_hard") for t in store.tasks(split) if t.variant == "standard"]


def test_selection_meets_quotas_and_is_deterministic():
    _, candidates = fixture_candidates()
    chosen = select_tiered(candidates, QUOTAS, "seed-a")
    assert [t.difficulty["tier"] for t in chosen] == ["easy"] * 2 + ["medium"] * 2 + ["hard"] * 3
    assert len({t.task_id for t in chosen}) == len(chosen)
    assert [t.task_id for t in select_tiered(list(reversed(candidates)), QUOTAS, "seed-a")] == [
        t.task_id for t in chosen
    ]
    assert all(t.variant == "standard" and t.split in {"test_id", "test_hard"} for t in chosen)


def test_a_short_tier_is_an_error():
    _, candidates = fixture_candidates()
    with pytest.raises(ValueError, match="fewer than its quota"):
        select_tiered(candidates, {"hard": 50}, "seed")


def test_an_edited_evalset_is_refused(tmp_path):
    _, candidates = fixture_candidates()
    record = make_evalset("final_eval", select_tiered(candidates, QUOTAS, "s"), seed="s", design={})
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(record))
    assert load_evalset(path)["evalset_id"] == record["evalset_id"]
    record["tasks"][0]["task_id"] = record["tasks"][1]["task_id"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="evalset_id"):
        load_evalset(path)


def test_an_evalset_is_served_as_its_own_split(tmp_path):
    store, candidates = fixture_candidates()
    chosen = select_tiered(candidates, QUOTAS, "s")
    served = EvalsetStore(store, [make_evalset("final_eval", chosen, seed="s", design={})])
    assert "final_eval" in served.splits() and set(store.splits()) < set(served.splits())
    assert [t.task_id for t in served.tasks("final_eval")] == [t.task_id for t in chosen]
    assert served.position("final_eval", chosen[-1].task_id) == len(chosen) - 1
    assert served.task("final_eval", 0).to_dict() == chosen[0].to_dict()
    assert sorted(t.task_id for t in served.iter_all()) == sorted(t.task_id for t in store.iter_all())
    with pytest.raises(ValueError, match="collides"):
        EvalsetStore(store, [make_evalset("dev", chosen, seed="s", design={})])


def test_benchmark_serves_evalsets_beside_the_release(tmp_path):
    _, candidates = fixture_candidates()
    root = tmp_path / "prepared"
    shutil.copytree(FIXTURE_RELEASE, root)
    (root / "evalsets").mkdir()
    record = make_evalset("final_eval", select_tiered(candidates, QUOTAS, "s"), seed="s", design={})
    (root / "evalsets" / "final_eval.json").write_text(json.dumps(record))
    benchmark = Benchmark.load(root)
    assert len(benchmark.store.tasks("final_eval")) == sum(QUOTAS.values())


def test_the_committed_final_eval_is_intact():
    record = load_evalset(ROOT / "data" / "eval-final_eval.json")
    assert record["name"] == "final_eval" and record["size"] == 50
    tiers = [t["tier"] for t in record["tasks"]]
    assert (tiers.count("easy"), tiers.count("medium"), tiers.count("hard")) == (17, 17, 16)
    assert {t["source_split"] for t in record["tasks"]} == {"test_id", "test_hard"}


def test_the_committed_core30_is_intact_and_nested_in_final_eval():
    core = load_evalset(ROOT / "data" / "eval-core30.json")
    final = {t["task_id"] for t in load_evalset(ROOT / "data" / "eval-final_eval.json")["tasks"]}
    tiers = [t["tier"] for t in core["tasks"]]
    assert core["size"] == 30 and (tiers.count("easy"), tiers.count("medium"), tiers.count("hard")) == (10, 10, 10)
    assert {t["task_id"] for t in core["tasks"]} <= final


RL_SETS = {
    "train12k": (13200, ("train",)),
    "val150": (150, ("dev",)),
    "core100": (100, ("test_id", "test_hard")),
    "core150": (150, ("test_id", "test_hard")),
    "heldout300": (300, ("test_id", "test_hard")),
}
# Each board keeps the previous one's tasks, in order, and appends the new ones.
NESTED_BOARDS = ("final_eval", "core100", "core150")
# core100 is a prefix of core150, so the largest board stands for both.
DISJOINT = ("train12k", "val150", "core150", "heldout300")


@pytest.mark.parametrize("name", sorted(RL_SETS))
def test_the_committed_rl_sets_are_intact(name):
    size, sources = RL_SETS[name]
    record = load_evalset(ROOT / "data" / f"eval-{name}.json")
    assert record["name"] == name and record["size"] == size
    assert {t["source_split"] for t in record["tasks"]} <= set(sources)


@pytest.mark.parametrize("smaller,bigger", list(zip(NESTED_BOARDS, NESTED_BOARDS[1:])))
def test_each_board_extends_the_previous_one_without_reordering_it(smaller, bigger):
    held = [t["task_id"] for t in load_evalset(ROOT / "data" / f"eval-{smaller}.json")["tasks"]]
    board = [t["task_id"] for t in load_evalset(ROOT / "data" / f"eval-{bigger}.json")["tasks"]]
    assert board[: len(held)] == held


@pytest.mark.parametrize("name,quotas", [("core100", (34, 33, 33)), ("core150", (50, 50, 50))])
def test_the_boards_are_evenly_tiered(name, quotas):
    tiers = [t["tier"] for t in load_evalset(ROOT / "data" / f"eval-{name}.json")["tasks"]]
    assert (tiers.count("easy"), tiers.count("medium"), tiers.count("hard")) == quotas


def test_the_rl_sets_never_share_a_task():
    ids = {
        name: {t["task_id"] for t in load_evalset(ROOT / "data" / f"eval-{name}.json")["tasks"]} for name in DISJOINT
    }
    for left, right in itertools.combinations(sorted(ids), 2):
        assert not ids[left] & ids[right], f"{left} and {right} share tasks"
