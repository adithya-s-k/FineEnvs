import json
from collections import Counter

import pytest
from nayana_ocr.data.evalset import (
    EVALSET_VERSION,
    allocate,
    build,
    load,
    save,
    select,
    summarize,
)
from nayana_ocr.data.schema import FAMILIES, LANGUAGES

# The transport fixture supplies one MCQ per page and no descriptive VQA.
FIXTURE_FAMILIES = [f for f in FAMILIES if f != "descriptive_vqa"]
FIXTURE_LANGUAGES = ["ar", "en"]
FIXTURE_SIZE = len(FIXTURE_LANGUAGES) * len(FIXTURE_FAMILIES)


def test_allocation_totals_exactly_and_keeps_families_equal():
    counts = allocate(list(LANGUAGES), list(FAMILIES), 500)
    assert sum(counts.values()) == 500
    per_family = Counter()
    per_language = Counter()
    for (language, family), count in counts.items():
        per_family[family] += count
        per_language[language] += count
    assert set(per_family.values()) == {100}
    # No language may be starved relative to another.
    assert max(per_language.values()) - min(per_language.values()) <= 1
    assert min(counts.values()) >= 1


def test_allocation_rejects_sizes_that_cannot_cover_every_group():
    with pytest.raises(ValueError):
        allocate(list(LANGUAGES), list(FAMILIES), len(LANGUAGES) * len(FAMILIES) - 1)


def test_selection_is_deterministic_balanced_and_block_local(corpus):
    catalog = corpus[0]
    size = FIXTURE_SIZE
    kwargs = dict(
        languages=FIXTURE_LANGUAGES, families=FIXTURE_FAMILIES, size=size, seed=42
    )
    first, blocks = select(catalog, **kwargs)
    again, _ = select(catalog, **kwargs)
    assert [e["task_id"] for e in first] == [e["task_id"] for e in again]
    assert len(first) == size
    assert len({e["task_id"] for e in first}) == size
    assert Counter(e["family"] for e in first) == {
        family: 2 for family in FIXTURE_FAMILIES
    }
    # Locality is the point: a language is served by far fewer blocks than tasks.
    per_language = Counter(b["block_id"].split(":")[0] for b in blocks)
    assert all(count <= size for count in per_language.values())
    assert {e["language"] for e in first} == {"ar", "en"}


def test_build_pins_served_reference_and_round_trips(corpus, tmp_path):
    catalog = corpus[0]
    record = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    assert record["evalset_version"] == EVALSET_VERSION
    assert record["failures"] == []
    assert record["snapshot_id"] == catalog.snapshot_id
    assert all(len(e["reference_sha256"]) == 64 for e in record["tasks"])
    assert all(e.get("asset_sha256") for e in record["tasks"])

    path = save(record, tmp_path / "eval.json")
    restored = load(path, catalog.snapshot_id)
    assert restored["evalset_id"] == record["evalset_id"]
    assert summarize(restored)["size"] == record["size"]


def test_load_refuses_a_different_snapshot(corpus, tmp_path):
    catalog = corpus[0]
    path = save(
        build(
            catalog,
            languages=FIXTURE_LANGUAGES,
            families=FIXTURE_FAMILIES,
            size=FIXTURE_SIZE,
            seed=42,
        ),
        tmp_path / "e.json",
    )
    with pytest.raises(ValueError, match="different corpus snapshot"):
        load(path, "0" * 64)


def test_load_refuses_an_edited_task_list(corpus, tmp_path):
    catalog = corpus[0]
    path = save(
        build(
            catalog,
            languages=FIXTURE_LANGUAGES,
            families=FIXTURE_FAMILIES,
            size=FIXTURE_SIZE,
            seed=42,
        ),
        tmp_path / "e.json",
    )
    record = json.loads(path.read_text())
    record["tasks"] = record["tasks"][::-1]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity does not match"):
        load(path, catalog.snapshot_id)


def test_a_page_that_cannot_be_loaded_is_replaced_and_recorded(
    corpus, tmp_path, monkeypatch
):
    """A page too large to render must cost one task, not the whole set.

    Dimensions are not in the index, so selection cannot avoid such a page; the walk
    draws spares and the next candidate takes its place, deterministically.
    """
    catalog = corpus[0]
    record = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    selected = [e["task_id"] for e in record["tasks"]]
    assert len(selected) == FIXTURE_SIZE
    assert not record["failures"] and not record["excluded"]

    original = type(catalog).materialize
    broken = selected[0]

    def fail_one(self, task):
        if task["task_id"] == broken:
            raise ValueError("synthetic load failure")
        return original(self, task)

    monkeypatch.setattr(type(catalog), "materialize", fail_one)
    repaired = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    ids = [e["task_id"] for e in repaired["tasks"]]
    # Still a full set, the bad page gone, and what it cost is on the record.
    assert len(ids) == FIXTURE_SIZE
    assert broken not in ids
    assert [f["task_id"] for f in repaired["excluded"]] == [broken]
    assert not repaired["failures"]
    # Everything that was fine before is still here: only the bad page moved.
    assert set(selected) - {broken} <= set(ids)
    # And the result still loads, because a replaced page is not a broken set.
    assert load(save(repaired, tmp_path / "repaired.json"), catalog.snapshot_id)

    # Rebuilding picks the same replacement rather than a fresh draw.
    again = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    assert [e["task_id"] for e in again["tasks"]] == ids


def test_a_group_the_spares_cannot_fill_is_a_failure(corpus, tmp_path, monkeypatch):
    """Replacement has a limit: past it the set is short and must say so."""
    catalog = corpus[0]

    def fail_all(self, task):
        raise ValueError("synthetic load failure")

    monkeypatch.setattr(type(catalog), "materialize", fail_all)
    record = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    assert record["failures"], "a set nothing loaded for must record failures"
    assert not record["tasks"]
    path = save(record, tmp_path / "broken.json")
    with pytest.raises(ValueError, match="do not load"):
        load(path, catalog.snapshot_id)


def test_exported_pack_serves_offline_through_the_ordinary_catalog(corpus, tmp_path):
    from nayana_ocr.data.catalog import Catalog
    from nayana_ocr.data.evalset import export_pack

    catalog = corpus[0]
    record = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    pack = tmp_path / "pack"
    manifest = export_pack(catalog, record, pack)
    assert manifest["status"] == "ready"
    assert manifest["corpus_snapshot_id"] == catalog.snapshot_id
    assert sum(c["tasks"] for c in manifest["counts"]) == FIXTURE_SIZE

    # The pack must serve without any bucket, index, or network access.
    offline = Catalog(pack)
    assert offline.count("test") == FIXTURE_SIZE
    for position, entry in enumerate(record["tasks"]):
        task = offline.at("test", position)
        assert task["task_id"] == entry["task_id"]
        assert task["reference"]
        path, mime = offline.asset(task["asset_sha256"])
        assert path.exists() and mime
        # Discovery still withholds the reference.
        assert "reference" not in offline.public(task)

    # The re-keyed copy inside the pack loads against the pack's own snapshot.
    packed = load(pack / "evalset.json", manifest["snapshot_id"])
    assert [e["task_id"] for e in packed["tasks"]] == [
        e["task_id"] for e in record["tasks"]
    ]


def test_export_refuses_a_set_with_recorded_failures(corpus, tmp_path):
    from nayana_ocr.data.evalset import export_pack

    catalog = corpus[0]
    record = build(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=FIXTURE_SIZE,
        seed=42,
    )
    record["failures"] = [{"task_id": record["tasks"][0]["task_id"], "error": "x"}]
    with pytest.raises(ValueError, match="recorded load failures"):
        export_pack(catalog, record, tmp_path / "pack")


def test_selection_spreads_across_blocks_not_one_dense_block(corpus):
    """A language must not draw its whole allocation from a single source block.

    One block is ~100 consecutive pages and in practice one document, so collapsing a
    language onto the densest block would score that language on a single document.
    """
    catalog = corpus[0]
    # Two tasks per (language, family): enough for the walk to need a second block.
    selection, blocks = select(
        catalog,
        languages=FIXTURE_LANGUAGES,
        families=FIXTURE_FAMILIES,
        size=2 * FIXTURE_SIZE,
        seed=42,
    )
    assert len(selection) == 2 * FIXTURE_SIZE

    per_language = Counter(
        block_id.split(":")[0] for block_id in {b["block_id"] for b in blocks}
    )
    assert min(per_language.values()) >= 2, per_language
    # The cap is the mechanism: no block may supply two tasks of the same family.
    per_block_family = Counter(
        (entry["block_id"], entry["family"]) for entry in selection
    )
    assert max(per_block_family.values()) == 1


def test_per_block_cap_must_be_positive(corpus):
    catalog = corpus[0]
    with pytest.raises(ValueError, match="at least one task per family"):
        select(
            catalog,
            languages=FIXTURE_LANGUAGES,
            families=FIXTURE_FAMILIES,
            size=FIXTURE_SIZE,
            per_block_per_family=0,
        )


def test_runtime_bucket_id_overrides_without_touching_recorded_provenance(monkeypatch):
    """A rename must be addressable at runtime; the hashed identity must not move."""
    from nayana_ocr.data.storage import runtime_bucket_id

    manifest = {"bucket_id": "OldOrg/corpus_bucket"}
    assert runtime_bucket_id(manifest) == "OldOrg/corpus_bucket"

    monkeypatch.setenv("NAYANA_BUCKET_ID", "NewOrg/corpus_bucket")
    assert runtime_bucket_id(manifest) == "NewOrg/corpus_bucket"
    # The recorded value feeds inventory_id -> snapshot_id and must be untouched.
    assert manifest["bucket_id"] == "OldOrg/corpus_bucket"


def test_block_order_ignores_density():
    """Density order made all 22 languages converge on the same parallel documents.

    The cap means a block only has to hold the families still needed, so ordering must
    depend on the block id alone -- never on how many tasks it happens to carry.
    """
    from nayana_ocr.data.evalset import _block_candidates

    class FakeCatalog:
        snapshot_id = "s" * 64

        def __init__(self, blocks):
            self._blocks = blocks

        def blocks(self, split, languages, families):
            return list(self._blocks)

    ids = [f"en:{i}" for i in range(12)]
    sparse = [{"block_id": b, "tasks": 1, "image_bytes": 1} for b in ids]
    # Same blocks, wildly different densities, deliberately reversed input order.
    dense = [
        {"block_id": b, "tasks": 1000 - i * 10, "image_bytes": 1}
        for i, b in enumerate(reversed(ids))
    ]
    order_sparse = [
        b["block_id"]
        for b in _block_candidates(FakeCatalog(sparse), "test", "en", FAMILIES, 42)
    ]
    order_dense = [
        b["block_id"]
        for b in _block_candidates(FakeCatalog(dense), "test", "en", FAMILIES, 42)
    ]
    assert order_sparse == order_dense
    # And the order must not simply be the densest first.
    assert order_dense != [
        b["block_id"] for b in sorted(dense, key=lambda b: -b["tasks"])
    ]


def test_judge_retry_covers_transient_failures_only():
    """A GRPO step must survive a retryable grading failure and stop on a real one."""
    from nayana_ocr.training import step_with_judge_retry

    class Client:
        def __init__(self, errors):
            self.errors = list(errors)
            self.calls = 0

        def step(self, action):
            self.calls += 1
            if self.errors:
                raise RuntimeError(self.errors.pop(0))
            return "graded"

    slept = []
    transient = "Server error: Judge request or verdict failed (ReadTimeout)"
    client = Client([transient, "Judge busy: 16 in flight"])
    assert step_with_judge_retry(client, "answer", sleep=slept.append) == "graded"
    assert client.calls == 3 and slept == [2.0, 4.0]

    # A non-judge failure is a real one: surface it immediately, do not mask it by retrying.
    hard = Client(["Invalid terminal reward"] * 5)
    with pytest.raises(RuntimeError, match="Invalid terminal reward"):
        step_with_judge_retry(hard, "answer", sleep=slept.append)
    assert hard.calls == 1

    # Exhausting the budget re-raises rather than returning a fabricated score.
    forever = Client([transient] * 10)
    with pytest.raises(RuntimeError, match="Judge request or verdict failed"):
        step_with_judge_retry(forever, "answer", attempts=3, sleep=lambda s: None)
    assert forever.calls == 3


def test_retry_signals_are_shared_between_server_and_trainer():
    """The server's wording and the trainer's matcher must stay one definition."""
    import inspect

    from nayana_ocr.models import JUDGE_BUSY, JUDGE_FAILED
    from nayana_ocr.server import judge as judge_module
    from nayana_ocr.training import transient_judge_failure

    # The judge builds its messages from the shared constants rather than literals.
    source = inspect.getsource(judge_module)
    assert "JUDGE_BUSY}" in source and "JUDGE_FAILED}" in source

    # Both survive the client's wrapping of a server-side error.
    for signal in (JUDGE_FAILED, JUDGE_BUSY):
        wrapped = RuntimeError(
            f"Server error: {signal} (x); no reward (code: EXECUTION_ERROR)"
        )
        assert transient_judge_failure(wrapped)

    # A genuine failure must not be retried into silence.
    assert not transient_judge_failure(RuntimeError("Invalid terminal reward"))
    assert not transient_judge_failure(
        RuntimeError("Reward was routed to the wrong rollout task")
    )


def test_subsample_is_balanced_deterministic_and_never_silently_whole():
    """A smoke slice must be reproducible and must not masquerade as the full set."""
    from nayana_ocr.data.evalset import subsample

    record = {
        "snapshot_id": "a" * 64,
        "tasks": [
            {
                "task_id": f"t{i}",
                "language": f"l{i % 7}",
                "family": FAMILIES[i % len(FAMILIES)],
            }
            for i in range(200)
        ],
    }
    first = subsample(record, 10, seed=42)
    assert [e["task_id"] for e in first] == [
        e["task_id"] for e in subsample(record, 10, seed=42)
    ]
    assert len(first) == 10
    assert Counter(e["family"] for e in first) == {f: 2 for f in FAMILIES}
    # A different seed must actually choose differently, or the knob is decorative.
    assert [e["task_id"] for e in subsample(record, 10, seed=7)] != [
        e["task_id"] for e in first
    ]
    # An uneven limit still totals exactly, spreading the remainder over families.
    assert len(subsample(record, 13, seed=42)) == 13
    # Asking for at least everything returns everything, not a truncation.
    assert len(subsample(record, 500, seed=42)) == 200
    with pytest.raises(ValueError, match="positive evaluation limit"):
        subsample(record, 0)
