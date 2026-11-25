"""The serving index must hold exactly what the release-backed path computes at startup."""

from __future__ import annotations

import json
import shutil
import sqlite3

import pytest
from conftest import FIXTURE_RELEASE
from retroenv.benchmark import Benchmark
from retroenv.leakage import LeakageRules
from retroenv.reactions import iter_reactions
from retroenv.retrieval import PrecedentIndex, StockIndex
from retroenv.serving import INDEX_FILE, ServingIndex, build_index
from retroenv.store import TaskStore, load_stock
from retroenv.verifier import RouteVerifier, known_routes_submission

STOCK_ID = "paroutes-archive-leaves"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """The fixture release plus its serving index, laid out as prepare.py leaves it."""
    root = tmp_path_factory.mktemp("prepared")
    shutil.copytree(FIXTURE_RELEASE, root, dirs_exist_ok=True)
    described = build_index(FIXTURE_RELEASE, root / "serving")
    return root, described


@pytest.fixture(scope="module")
def index(prepared):
    root, _ = prepared
    return ServingIndex(root / "serving" / INDEX_FILE)


def test_index_describes_every_split(prepared):
    _, described = prepared
    json_store = TaskStore(FIXTURE_RELEASE / "tasks-private", FIXTURE_RELEASE / "stocks")
    assert set(described["splits"]) == set(json_store.splits())
    for split, info in described["splits"].items():
        assert info["tasks"] == len(json_store.tasks(split))
    assert described["stocks"] == {STOCK_ID: len(load_stock(FIXTURE_RELEASE / "stocks" / f"{STOCK_ID}.smi"))}
    assert len(described["sha256"]) == 64


def test_tasks_match_the_release_rows(index):
    json_store = TaskStore(FIXTURE_RELEASE / "tasks-private", FIXTURE_RELEASE / "stocks")
    store = index.task_store()
    assert store.splits() == json_store.splits()
    for split in json_store.splits():
        expected, served = json_store.tasks(split), store.tasks(split)
        assert len(served) == len(expected)
        assert [t.to_dict() for t in served] == [t.to_dict() for t in expected]
        assert [served[i].to_dict() for i in range(len(served))] == [t.to_dict() for t in expected]
        assert served[-1].task_id == expected[-1].task_id
        last = expected[-1]
        assert store.position(split, last.task_id) == len(expected) - 1
        assert store.find(last.task_id).to_dict() == last.to_dict()
    with pytest.raises(IndexError):
        store.tasks(json_store.splits()[0])[10_000]
    with pytest.raises(KeyError):
        store.tasks("no-such-split")


def test_stock_matches_a_fresh_index(index):
    fresh = StockIndex(load_stock(FIXTURE_RELEASE / "stocks" / f"{STOCK_ID}.smi"))
    served = index.stock_index(STOCK_ID)
    assert served.smiles == fresh.smiles
    queries = [
        ("exact", fresh.smiles[0]),
        ("exact", "C"),
        ("similarity", fresh.smiles[3]),
        ("class", "amine"),
        ("substructure", "c1ccccc1"),
        ("inchikey", fresh.inchikey(fresh.smiles[5])),
    ]
    for mode, query in queries:
        assert served.retrieve(query, mode=mode, limit=20) == fresh.retrieve(query, mode=mode, limit=20), mode


def test_precedents_match_a_fresh_index(index):
    manifest = json.loads((FIXTURE_RELEASE / "manifest.json").read_text())
    rules = LeakageRules.from_manifest(manifest)
    fresh = PrecedentIndex(iter_reactions(FIXTURE_RELEASE / "library" / "reactions.jsonl.gz"), rules)
    served = index.precedents(rules)
    assert served.records == fresh.records
    store = TaskStore(FIXTURE_RELEASE / "tasks-private", FIXTURE_RELEASE / "stocks")
    train = store.tasks("train")
    for task in [None, *train[:3]]:
        for product in [r.product for r in fresh.records[:6]]:
            for reaction_class in ("", fresh.records[0].reaction_class):
                expected = fresh.search(task=task, product_smiles=product, reaction_class=reaction_class, limit=20)
                assert (
                    served.search(task=task, product_smiles=product, reaction_class=reaction_class, limit=20)
                    == expected
                )


def test_benchmark_scores_identically_from_the_index(prepared):
    root, _ = prepared
    from_release = Benchmark.load(FIXTURE_RELEASE)
    from_index = Benchmark.load(root)
    assert from_index.store.splits() == from_release.store.splits()
    verifier = RouteVerifier(from_release.library)
    for split in from_release.store.splits():
        for task in from_release.store.tasks(split):
            served = from_index.store.tasks(split)[from_index.store.position(split, task.task_id)]
            stock = from_index.store.stock(task.stock_id)
            assert stock == from_release.store.stock(task.stock_id)
            submission = known_routes_submission(task, stock)
            assert (
                RouteVerifier(from_index.library).score_submission(served, submission, stock).to_dict()
                == verifier.score_submission(task, submission, from_release.store.stock(task.stock_id)).to_dict()
            )


def test_another_rdkit_rebuilds_instead_of_trusting_the_index(prepared, tmp_path):
    root, _ = prepared
    copy = tmp_path / INDEX_FILE
    shutil.copy2(root / "serving" / INDEX_FILE, copy)
    with sqlite3.connect(copy) as db:
        db.execute("UPDATE meta SET value = '0.0.0' WHERE key = 'rdkit_version'")
    index = ServingIndex(copy)
    assert not index.rdkit_matches
    rules = LeakageRules.from_manifest(json.loads((FIXTURE_RELEASE / "manifest.json").read_text()))
    reactions = FIXTURE_RELEASE / "library" / "reactions.jsonl.gz"
    assert index.precedents(rules, reactions).records == PrecedentIndex(iter_reactions(reactions), rules).records
    assert (
        index.stock_index(STOCK_ID).smiles
        == StockIndex(load_stock(FIXTURE_RELEASE / "stocks" / f"{STOCK_ID}.smi")).smiles
    )
    with pytest.raises(ValueError, match="RDKit differs"):
        index.precedents(rules)


def test_unknown_index_version_is_refused(prepared, tmp_path):
    root, _ = prepared
    copy = tmp_path / INDEX_FILE
    shutil.copy2(root / "serving" / INDEX_FILE, copy)
    with sqlite3.connect(copy) as db:
        db.execute("UPDATE meta SET value = '99' WHERE key = 'index_version'")
    with pytest.raises(ValueError, match="index version"):
        ServingIndex(copy)
