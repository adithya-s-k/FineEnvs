"""Sarvam Indic OCR Bench as evaluation splits served exactly like the corpus."""

import hashlib
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nayana_ocr.data import indic_ocr_bench as bench
from nayana_ocr.data.catalog import PUBLIC_FIELDS
from nayana_ocr.data.composite import CompositeCatalog
from nayana_ocr.models import NayanaAction, NayanaObservation
from nayana_ocr.server import indic_ocr_bench_metrics as official
from nayana_ocr.server.bench_rewards import official_metrics
from nayana_ocr.server.environment import NayanaEnvironment
from nayana_ocr.server.rewards import score
from PIL import Image

SPLIT = "indic_ocr_bench_test"
ROWS = [
    ("indic_ocr_bench_test_kan_1", "ಕನ್ನಡ ಪಠ್ಯ", "Kannada", (40, 12), "PNG"),
    ("indic_ocr_bench_test_brx_2", "बड़ो लिरनाय", "Bodo", (32, 10), "JPEG"),
    ("indic_ocr_bench_test_kan_3", "ಮತ್ತೊಂದು ಸಾಲು", "Kannada", (48, 14), "PNG"),
]


def encoded(size, shade, fmt):
    buffer = io.BytesIO()
    Image.new("L", size, shade).save(buffer, format=fmt)
    return buffer.getvalue()


def write_parquet(path, rows):
    pq.write_table(
        pa.table(
            {
                "image": [
                    None if size is None
                    else {"bytes": encoded(size, 40 * i + 10, fmt), "path": f"{name}.img"}
                    for i, (name, _, _, size, fmt) in enumerate(rows)
                ],
                "image_name": [r[0] for r in rows],
                "gt": [r[1] for r in rows],
                "language": [r[2] for r in rows],
            }
        ),
        path,
    )
    return str(path)


@pytest.fixture
def splits(monkeypatch):
    # Fake rows under the real split name, so pin the fake split's (empty) exclusions
    # too - otherwise the guard against an upstream change fires, correctly, on them.
    monkeypatch.setattr(bench, "SPLITS", {SPLIT: ("test", len(ROWS))})
    monkeypatch.setattr(bench, "EXCLUDED", {SPLIT: ()})


@pytest.fixture
def published(tmp_path, splits):
    """A bucket as publishing leaves it, read in place the way a Space reads its mount."""
    parquet = write_parquet(tmp_path / "test.parquet", ROWS)
    root = tmp_path / "bucket"
    bench.build_split(SPLIT, root, fetch=lambda d: [parquet])
    return bench.BenchCatalog(tmp_path / "cache", root=root, fallback=False)


def test_crops_are_served_as_ordinary_section_ocr_tasks(published):
    task = published.at(SPLIT, 0)
    public = published.public(task)
    assert task["family"] == "section_ocr"
    assert set(PUBLIC_FIELDS) <= set(public)
    # The reference never leaves the server.
    assert "reference" not in public and ROWS[0][1] not in str(public)
    assert public["language"] == "kn"
    assert public["prompt"] == bench.prompt("kn")
    assert (public["width"], public["height"]) == (40, 12)
    assert public["asset_path"].startswith(f"/assets/{task['asset_sha256']}?task_id=")


def test_image_formats_are_read_from_the_bytes_not_the_card(published):
    assert published.at(SPLIT, 0)["mime"] == "image/png"
    assert published.at(SPLIT, 1)["mime"] == "image/jpeg"


def test_task_ids_round_trip_and_reject_other_revisions(published):
    task = published.at(SPLIT, 1)
    assert published.get(task["task_id"]) is task
    assert task["language"] == "brx"  # not a corpus language
    with pytest.raises(KeyError):
        published.get(task["task_id"].replace(bench.VERSION, "0" * 12))


def test_assets_are_served_only_for_their_own_task(published):
    first, second = published.at(SPLIT, 0), published.at(SPLIT, 1)
    raw, mime = published.asset_bytes(first["asset_sha256"], first["task_id"])
    assert hashlib.sha256(raw).hexdigest() == first["asset_sha256"]
    assert mime == "image/png"
    with pytest.raises(KeyError):
        published.asset_bytes(first["asset_sha256"], second["task_id"])
    with pytest.raises(KeyError):
        published.asset_bytes("not-a-hash", first["task_id"])


def test_a_mount_without_the_revision_is_an_error_not_a_silent_fallback(tmp_path, splits):
    catalog = bench.BenchCatalog(tmp_path / "cache", root=tmp_path / "empty-mount")
    with pytest.raises(FileNotFoundError, match="publish"):
        catalog.at(SPLIT, 0)


def test_without_a_bucket_it_builds_once_and_then_reuses_the_cache(
    tmp_path, splits, monkeypatch
):
    parquet = write_parquet(tmp_path / "test.parquet", ROWS)
    fetched = []

    def unreachable(self, split):
        raise ConnectionError("bucket unreachable")

    monkeypatch.setattr(bench.BenchCatalog, "_pull", unreachable)
    first = bench.BenchCatalog(
        tmp_path / "cache", fetch=lambda d: fetched.append(d) or [parquet]
    )
    assert first.count(SPLIT) == 3 and first.at(SPLIT, 2)["unit"] == ROWS[2][0]

    def refuse(directory):
        raise AssertionError("the local cache should have been reused")

    second = bench.BenchCatalog(tmp_path / "cache", fetch=refuse)
    assert second.at(SPLIT, 0)["unit"] == ROWS[0][0]
    assert fetched == ["test"]


def test_a_split_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "SPLITS", {SPLIT: ("test", 99)})
    parquet = write_parquet(tmp_path / "t.parquet", ROWS)
    with pytest.raises(ValueError, match="expected 99"):
        bench.build_split(SPLIT, tmp_path / "bucket", fetch=lambda d: [parquet])


def test_an_unknown_language_is_an_error_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "SPLITS", {SPLIT: ("test", 1)})
    parquet = write_parquet(tmp_path / "t.parquet", [("x_1", "t", "Klingon", (8, 8), "PNG")])
    with pytest.raises(ValueError, match="unknown language"):
        bench.build_split(SPLIT, tmp_path / "bucket", fetch=lambda d: [parquet])


def test_every_listed_language_has_a_distinct_code():
    assert len(bench.LANGUAGES) == 23
    assert len(set(bench.LANGUAGES.values())) == 23


@pytest.mark.parametrize(
    "gt,pred",
    [
        ("ಕನ್ನಡ ಪಠ್ಯ", "ಕನ್ನಡ ಪಠ್ಯ"),
        ("ಕನ್ನಡ ಪಠ್ಯ", "ಕನ್ನಡ ಪಠ"),
        ("“quoted” – text", '"quoted" - text'),
        ("दो पंक्तियाँ\nयहाँ", "दो पंक्तियाँ यहाँ"),
    ],
)
def test_reported_metrics_are_the_benchmarks_own(gt, pred):
    ours = official_metrics(pred, gt)
    _, results = official.compute_metrics(
        [{"image_name": "x", "gt": gt, "pred": pred}], normalize=True, replace_n=False
    )
    assert ours["official_cer"] == results[0]["metrics"]["cer"]
    assert ours["official_wer"] == results[0]["metrics"]["wer"]


def test_empty_and_runaway_answers_carry_the_benchmarks_flags():
    assert official_metrics("  ", "ಕನ್ನಡ")["missing_prediction"] is True
    assert official_metrics("ಕನ್ನಡ " * 40, "ಕನ್ನಡ")["loop_or_catastrophic"] is True


class Primary:
    """Stands in for the corpus catalog: it owns everything a benchmark does not."""

    manifest = {"snapshot_id": "corpus"}
    eval_splits = {"eval_22_validation": [1, 2]}
    eval_ids = {"eval_22_validation": "abc"}

    def __init__(self):
        self.prefetched = []

    def splits(self):
        return ["train", "validation", "test", "eval_22_validation"]

    def count(self, split):
        return 7

    def blocks(self, *args, **kwargs):
        return ["corpus-block"]

    def prefetch(self, *, task_ids=(), block_ids=()):
        self.prefetched.extend(task_ids)


def test_composite_routes_benchmark_calls_and_leaves_the_corpus_alone(published):
    primary = Primary()
    composite = CompositeCatalog(primary, [published])
    assert composite.splits()[-1] == SPLIT
    assert composite.count("validation") == 7 and composite.count(SPLIT) == 3
    assert composite.blocks() == ["corpus-block"]  # training samplers see the corpus only
    assert composite.manifest["snapshot_id"] == "corpus"
    assert len(composite.eval_splits[SPLIT]) == 3
    task = composite.at(SPLIT, 0)
    assert composite.get(task["task_id"]) is task
    # The playground prefetches neighbours; a benchmark ID must never reach the corpus.
    composite.prefetch(task_ids=["nayana-c1.x.kn.train.1", task["task_id"]])
    assert primary.prefetched == ["nayana-c1.x.kn.train.1"]


def test_reward_and_grading_are_identical_to_corpus_section_ocr(published):
    env = NayanaEnvironment(catalog=CompositeCatalog(Primary(), [published]))
    task = published.at(SPLIT, 2)
    observation = env.reset(task_id=task["task_id"])
    assert observation.family == "section_ocr" and observation.language == "kn"
    answer = ROWS[2][1][:-2]
    result = env.step(NayanaAction(answer=answer))
    expected_reward, expected_metrics = score("section_ocr", answer, ROWS[2][1])
    assert result.reward == pytest.approx(expected_reward)
    assert result.grading_policy_id == ""  # exactly what a corpus section task reports
    for key, value in expected_metrics.items():
        assert result.metrics[key] == value
    # The benchmark's own scores ride along, reported and never rewarded.
    assert result.metrics["official_cer"] == official_metrics(answer, ROWS[2][1])["official_cer"]


def test_the_tool_surface_is_unchanged():
    assert sorted(NayanaAction.model_fields) == ["answer", "metadata"]
    assert sorted(NayanaObservation.model_fields) == [
        "annotation_masked", "asset_path", "asset_sha256", "done", "family",
        "grading_policy_id", "height", "language", "metadata", "metrics", "mime",
        "prompt", "reading_order", "reading_order_policy", "reward", "snapshot_id",
        "split", "task_id", "width",
    ]


def test_the_harness_reports_what_the_benchmarks_scorer_reports():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "train" / "eval_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_vllm_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = [
        ("kn", "ಕನ್ನಡ ಪಠ್ಯ", "ಕನ್ನಡ ಪಠ್ಯ"),
        ("kn", "ಮತ್ತೊಂದು ಸಾಲು ಇಲ್ಲಿ", "ಮತ್ತೊಂದು ಸಾಲ"),
        ("brx", "बड़ो लिरनाय", ""),  # empty: dropped from the official means
        ("brx", "बड़ो", "बड़ो " * 40),  # runaway: kept, but outside valid_*
    ]
    samples = [
        {"task_id": f"{bench.SOURCE}.{bench.VERSION}.{SPLIT}.x{i}", "language": lang,
         "family": "section_ocr", **official_metrics(pred, gt)}
        for i, (lang, gt, pred) in enumerate(pairs)
    ]
    ours = module.bench_report(samples)
    theirs, _ = official.compute_metrics(
        [{"image_name": str(i), "gt": gt, "pred": pred, "language": lang}
         for i, (lang, gt, pred) in enumerate(pairs)],
        normalize=True,
        replace_n=False,
    )
    for key in ("cer", "wer"):
        assert ours["avg_metrics"][key] == pytest.approx(theirs["avg_metrics"][key])
    for key in ("scored_sample_count", "valid_sample_count", "missing_prediction_count",
                "loop_failure_count", "benchmark_sample_count"):
        assert ours[key] == theirs[key], key
    assert ours["word_accuracy"] == pytest.approx(theirs["word_accuracy"])
    # A corpus task is not benchmark data, even though it shares the family.
    assert module.bench_report([{"task_id": "nayana-c1.x", "family": "section_ocr"}]) is None


def test_rows_without_an_image_are_excluded_recorded_and_pinned(tmp_path, monkeypatch):
    """The pinned test split ships rows with no image; they are not answerable."""
    rows = ROWS + [("indic_ocr_bench_test_eng_5", "no picture", "English", None, None)]
    monkeypatch.setattr(bench, "SPLITS", {SPLIT: ("test", len(rows))})
    parquet = write_parquet(tmp_path / "t.parquet", rows)
    fetch = lambda d: [parquet]  # noqa: E731

    # Unpinned, the build refuses: a source change must be noticed, not absorbed.
    monkeypatch.setattr(bench, "EXCLUDED", {SPLIT: ()})
    with pytest.raises(ValueError, match="re-pin"):
        bench.build_split(SPLIT, tmp_path / "a", fetch=fetch)
    # Discovery reports it without serving it.
    bench.build_split(SPLIT, tmp_path / "b", fetch=fetch, discover=True)

    monkeypatch.setattr(bench, "EXCLUDED", {SPLIT: ("indic_ocr_bench_test_eng_5",)})
    served = bench.build_split(SPLIT, tmp_path / "c", fetch=fetch)
    assert [t["unit"] for t in served] == [r[0] for r in ROWS]
    import json

    index = json.loads(bench.index_path(tmp_path / "c", SPLIT).read_text())
    assert index["total_rows"] == 4
    assert index["excluded"] == [{
        "image_name": "indic_ocr_bench_test_eng_5", "language": "en",
        "reason": "no image in the source row",
    }]
    catalog = bench.BenchCatalog(tmp_path / "cache", root=tmp_path / "c", fallback=False)
    assert catalog.count(SPLIT) == 3
    assert catalog.at(SPLIT, 2)["unit"] == ROWS[2][0]


def test_the_real_pins_describe_the_real_revision():
    # The one imageless row at the pinned revision, found by --discover.
    assert bench.EXCLUDED["indic_ocr_bench_test"] == ("indic_ocr_bench_test_eng_5",)
    assert bench.served_count("indic_ocr_bench_test") == 6908
    assert bench.served_count("indic_ocr_bench_small") == 1173
