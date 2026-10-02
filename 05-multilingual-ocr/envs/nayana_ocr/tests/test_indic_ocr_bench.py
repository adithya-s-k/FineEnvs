"""Sarvam Indic OCR Bench as an evaluation-only source beside the corpus."""

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
from nayana_ocr.server.bench_rewards import POLICY, score_bench
from nayana_ocr.server.environment import NayanaEnvironment
from PIL import Image

ROWS = [
    ("indic_ocr_bench_test_kan_1", "ಕನ್ನಡ ಪಠ್ಯ", "Kannada", (40, 12)),
    ("indic_ocr_bench_test_brx_2", "बड़ो लिरनाय", "Bodo", (32, 10)),
    ("indic_ocr_bench_test_kan_3", "ಮತ್ತೊಂದು ಸಾಲು", "Kannada", (48, 14)),
]


def png(size, shade):
    buffer = io.BytesIO()
    Image.new("L", size, shade).save(buffer, format="PNG")
    return buffer.getvalue()


def write_parquet(path, rows):
    table = pa.table(
        {
            "image": [
                {"bytes": png(size, 40 * i + 10), "path": f"{name}.png"}
                for i, (name, _, _, size) in enumerate(rows)
            ],
            "image_name": [r[0] for r in rows],
            "gt": [r[1] for r in rows],
            "language": [r[2] for r in rows],
        }
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench, "SPLITS", {"indic_ocr_bench_test": ("test", len(ROWS))}
    )
    parquet = write_parquet(tmp_path / "test.parquet", ROWS)
    calls = []

    def fetch(directory):
        calls.append(directory)
        return [str(parquet)]

    return bench.BenchCatalog(tmp_path / "cache", fetch=fetch), calls


def test_tasks_present_exactly_as_section_ocr_tasks_do(source):
    catalog, _ = source
    assert catalog.count("indic_ocr_bench_test") == 3
    task = catalog.at("indic_ocr_bench_test", 0)
    public = catalog.public(task)
    # The shared public fields and nothing else - in particular never the reference.
    assert set(PUBLIC_FIELDS) <= set(public)
    assert "reference" not in public and "ಕನ್ನಡ" not in str(public)
    assert public["family"] == bench.FAMILY
    assert public["language"] == "kn"
    assert public["prompt"] == bench.prompt("kn")
    assert public["asset_path"].startswith(f"/assets/{task['asset_sha256']}?task_id=")
    assert (public["width"], public["height"]) == (40, 12)


def test_task_ids_round_trip_and_reject_other_revisions(source):
    catalog, _ = source
    task = catalog.at("indic_ocr_bench_test", 1)
    assert catalog.get(task["task_id"]) is task
    assert task["language"] == "brx"  # a language the Nayana corpus does not have
    stale = task["task_id"].replace(bench.REVISION[:12], "0" * 12)
    with pytest.raises(KeyError):
        catalog.get(stale)


def test_assets_are_served_only_for_the_task_they_belong_to(source):
    catalog, _ = source
    first, second = (catalog.at("indic_ocr_bench_test", i) for i in (0, 1))
    raw, mime = catalog.asset_bytes(first["asset_sha256"], first["task_id"])
    assert hashlib.sha256(raw).hexdigest() == first["asset_sha256"]
    assert mime == "image/png"
    with pytest.raises(KeyError):
        catalog.asset_bytes(first["asset_sha256"], second["task_id"])
    with pytest.raises(KeyError):
        catalog.asset_bytes("not-a-hash", first["task_id"])


def test_extraction_happens_once_and_is_reused(source, tmp_path):
    catalog, calls = source
    catalog.at("indic_ocr_bench_test", 0)

    def refuse(directory):
        raise AssertionError("the cached index should have been reused")

    again = bench.BenchCatalog(tmp_path / "cache", fetch=refuse)
    assert again.count("indic_ocr_bench_test") == 3
    assert again.at("indic_ocr_bench_test", 2)["unit"] == ROWS[2][0]
    assert calls == ["test"]


def test_a_split_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "SPLITS", {"indic_ocr_bench_test": ("test", 99)})
    parquet = write_parquet(tmp_path / "t.parquet", ROWS)
    catalog = bench.BenchCatalog(tmp_path / "cache", fetch=lambda d: [str(parquet)])
    with pytest.raises(ValueError, match="expected 99"):
        catalog.at("indic_ocr_bench_test", 0)


def test_an_unknown_language_is_an_error_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "SPLITS", {"indic_ocr_bench_test": ("test", 1)})
    parquet = write_parquet(tmp_path / "t.parquet", [("x_1", "text", "Klingon", (8, 8))])
    catalog = bench.BenchCatalog(tmp_path / "cache", fetch=lambda d: [str(parquet)])
    with pytest.raises(ValueError, match="unknown language"):
        catalog.at("indic_ocr_bench_test", 0)


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
def test_reward_uses_the_benchmarks_own_scores(gt, pred):
    reward, metrics = score_bench(pred, gt)
    summary, results = official.compute_metrics(
        [{"image_name": "x", "gt": gt, "pred": pred}], normalize=True, replace_n=False
    )
    assert metrics["char_error_rate"] == results[0]["metrics"]["cer"]
    assert metrics["word_error_rate"] == results[0]["metrics"]["wer"]
    assert reward == pytest.approx(1.0 - summary["avg_metrics"]["cer"])


def test_an_empty_answer_scores_zero_and_is_flagged():
    reward, metrics = score_bench("   ", "ಕನ್ನಡ")
    assert reward == 0.0
    assert metrics["missing_prediction"] is True


def test_runaway_repetition_is_flagged_as_the_benchmark_flags_it():
    _, metrics = score_bench("ಕನ್ನಡ " * 40, "ಕನ್ನಡ")
    assert metrics["loop_or_catastrophic"] is True


class Primary:
    """Stands in for the corpus catalog: it owns everything a benchmark does not."""

    manifest = {"snapshot_id": "corpus"}
    eval_splits = {"eval_22_validation": [1, 2]}
    eval_ids = {"eval_22_validation": "abc"}

    def splits(self):
        return ["train", "validation", "test", "eval_22_validation"]

    def count(self, split):
        return 7

    def blocks(self, *args, **kwargs):
        return ["corpus-block"]


def test_composite_routes_benchmark_calls_and_leaves_the_corpus_alone(source):
    catalog, _ = source
    composite = CompositeCatalog(Primary(), [catalog])
    assert composite.splits()[-1] == "indic_ocr_bench_test"
    assert composite.count("validation") == 7
    assert composite.count("indic_ocr_bench_test") == 3
    assert composite.blocks() == ["corpus-block"]
    assert composite.manifest["snapshot_id"] == "corpus"
    assert len(composite.eval_splits["indic_ocr_bench_test"]) == 3
    assert "eval_22_validation" in composite.eval_ids
    task = composite.at("indic_ocr_bench_test", 0)
    assert composite.get(task["task_id"]) is task


def test_reset_and_step_through_the_unchanged_environment(source):
    catalog, _ = source
    env = NayanaEnvironment(catalog=CompositeCatalog(Primary(), [catalog]))
    task = catalog.at("indic_ocr_bench_test", 0)
    observation = env.reset(task_id=task["task_id"])
    assert observation.family == bench.FAMILY and observation.language == "kn"
    result = env.step(NayanaAction(answer=ROWS[0][1]))
    assert result.done and result.reward == pytest.approx(1.0)
    assert result.grading_policy_id == POLICY
    assert result.metrics["char_error_rate"] == 0.0


def test_the_tool_surface_is_unchanged():
    # Adding a benchmark must not change what an agent sends or receives.
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
    samples = []
    for i, (lang, gt, pred) in enumerate(pairs):
        _, metrics = score_bench(pred, gt)
        samples.append({"task_id": str(i), "language": lang, "family": bench.FAMILY, **metrics})
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
    assert module.bench_report([{"family": "section_ocr"}]) is None
