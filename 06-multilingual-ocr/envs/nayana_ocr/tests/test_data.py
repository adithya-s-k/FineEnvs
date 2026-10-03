import copy
import json
from dataclasses import replace

import pytest
from nayana_ocr.data.catalog import SPLITS, Catalog
from nayana_ocr.data.prepare import PrepareConfig, prepare
from nayana_ocr.data.schema import REVISION, document_id, split_for_page
from nayana_ocr.data.tasks import derive_tasks
from nayana_ocr.fixtures import fixture_rows, fixture_stream, make_fixture


def test_all_language_copies_and_pages_share_document_split():
    for document in range(100):
        assert split_for_page(f"document_{document}_page_0") == split_for_page(
            f"document_{document}_page_999"
        )
    with pytest.raises(ValueError, match="partition safely"):
        document_id("00000000")


def test_derivation_uses_language_text_original_coordinates_and_mcq_index():
    row = next(fixture_rows("kn"))
    row["regions.json"][0]["translated_text"] = "ಕನ್ನಡ ಭಾಷೆ"
    tasks, skipped = derive_tasks(row, "kn", REVISION)
    ocr, vqa, page, layout = tasks
    assert ocr["reference"] == "ಕನ್ನಡ ಭಾಷೆ" and ocr["unit"] == "17"
    assert (ocr["width"], ocr["height"]) == (245, 75)
    assert ocr["mime"] == "image/png" and vqa["media"] == row["jpg"]["bytes"]
    assert vqa["reference"] == "B" and not skipped
    assert page["family"] == "page_ocr" and page["reference"] == ocr["reference"]
    assert page["mime"] == "image/png" and page["reading_order"] == [17]
    assert page["annotation_masked"] is True
    other, _ = derive_tasks(row, "en", REVISION)
    assert ocr["task_id"] != other[0]["task_id"] and ocr["split"] == other[0]["split"]


def test_invalid_regions_and_ambiguous_questions_are_audited():
    row = next(fixture_rows("en"))
    row["regions.json"][0]["bbox"]["xmax"] = 321
    row["vqa.json"]["questions"][0]["options"] = ["invoice", "invoice"]
    tasks, skipped = derive_tasks(row, "en", REVISION)
    assert tasks == []
    assert skipped == {
        "invalid_bbox": 1,
        "layout_incomplete_or_invalid_annotations": 1,
        "ambiguous_mcq_answer": 1,
        "page_ocr_incomplete_annotations": 1,
    }
    with pytest.raises(ValueError, match="max_pixels"):
        derive_tasks(row, "en", REVISION, max_pixels=10)


def test_discovery_never_reads_images_and_does_not_expose_references(
    tmp_path, monkeypatch
):
    make_fixture(tmp_path)
    catalog = Catalog(tmp_path)
    monkeypatch.setattr(
        "PIL.Image.open", lambda *a, **k: pytest.fail("Image decoded during discovery")
    )
    assert catalog.count("train") == 32
    task = catalog.task_range("train", 0, 1)[0]
    assert "reference" not in task and "media" not in task
    assert "invoice" not in json.dumps({k: v for k, v in task.items() if k != "prompt"})
    with pytest.raises(IndexError):
        catalog.at("train", -1)
    with pytest.raises(KeyError):
        catalog.asset("../../catalog.sqlite")


def test_interrupted_stream_resumes_to_identical_task_window(tmp_path):
    config = PrepareConfig(
        languages=("en",), pages_per_language=6, source="synthetic-fixture"
    )

    class Interrupt:
        def __init__(self, stream):
            self.stream = stream

        def __iter__(self):
            for i, row in enumerate(self.stream):
                if i == 2:
                    raise RuntimeError("simulated interruption")
                yield row

        def state_dict(self):
            return self.stream.state_dict()

    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated"):
        prepare(
            interrupted,
            config,
            source_factory=lambda c, lang: Interrupt(fixture_stream(c, lang)),
        )
    assert not (interrupted / "manifest.json").exists()
    resumed = prepare(interrupted, config, source_factory=fixture_stream)
    clean = prepare(tmp_path / "clean", config, source_factory=fixture_stream)
    assert resumed["snapshot_id"] == clean["snapshot_id"]
    assert resumed["pages"] == {"en": 6}
    for split in SPLITS:
        assert Catalog(interrupted).task_range(split) == Catalog(
            tmp_path / "clean"
        ).task_range(split)
    with pytest.raises(ValueError, match="Resume settings differ"):
        prepare(
            interrupted, replace(config, split_seed=7), source_factory=fixture_stream
        )


def test_byte_budget_never_publishes_partial_snapshot(tmp_path):
    config = PrepareConfig(
        languages=("en",),
        pages_per_language=1,
        max_media_bytes=1,
        source="synthetic-fixture",
    )
    with pytest.raises(ValueError, match="byte limit"):
        prepare(tmp_path, config, source_factory=fixture_stream)
    assert not (tmp_path / "manifest.json").exists()
    assert list((tmp_path / "assets").iterdir()) == []


def test_english_missing_translation_is_valid_non_english_is_not():
    row = copy.deepcopy(next(fixture_rows("en")))
    del row["regions.json"][0]["translated_text"]
    english, _ = derive_tasks(row, "en", REVISION)
    kannada, skipped = derive_tasks(row, "kn", REVISION)
    assert len(english) == 4 and len(kannada) == 2
    assert skipped["empty_region_text"] == 1
    assert skipped["page_ocr_incomplete_annotations"] == 1


def test_parquet_checkpoint_replays_exact_next_page(tmp_path):
    from datasets import Dataset, Image, load_dataset

    rows = list(fixture_rows("en"))
    path = tmp_path / "source.parquet"
    Dataset.from_list(rows).cast_column("jpg", Image()).to_parquet(path)

    def stream():
        return load_dataset(
            "parquet", data_files=str(path), split="train", streaming=True, batch_size=1
        ).cast_column("jpg", Image(decode=False))

    original = stream()
    iterator = iter(original)
    next(iterator)
    next(iterator)
    checkpoint = json.loads(json.dumps(original.state_dict()))
    remaining = [r["image_id.txt"] for r in iterator]
    resumed = stream()
    resumed.load_state_dict(checkpoint)
    assert [r["image_id.txt"] for r in resumed] == remaining
    assert remaining == [r["image_id.txt"] for r in rows[2:]]
