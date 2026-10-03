import copy
import io

import pytest
from nayana_ocr.data.catalog import Catalog
from nayana_ocr.data.reading_order import ordered_regions
from nayana_ocr.data.schema import REVISION
from nayana_ocr.data.tasks import derive_tasks
from nayana_ocr.fixtures import fixture_rows, make_fixture
from nayana_ocr.server.gradio_ui import Playground
from nayana_ocr.server.rewards import score
from PIL import Image, ImageDraw


def test_full_width_headings_and_columns_have_explicit_reading_order():
    regions = [
        {"region_id": 7, "bbox": [0, 0, 100, 10]},
        {"region_id": 2, "bbox": [0, 20, 40, 40]},
        {"region_id": 5, "bbox": [0, 50, 40, 70]},
        {"region_id": 8, "bbox": [60, 20, 100, 40]},
        {"region_id": 1, "bbox": [60, 50, 100, 70]},
        {"region_id": 9, "bbox": [0, 80, 100, 90]},
    ]
    assert [r["region_id"] for r in ordered_regions(list(reversed(regions)))] == [
        7,
        2,
        5,
        8,
        1,
        9,
    ]
    assert [r["region_id"] for r in ordered_regions(regions, rtl=True)] == [
        7,
        8,
        1,
        2,
        5,
        9,
    ]


def test_page_reference_uses_geometry_not_source_list_order():
    row = next(fixture_rows("en"))
    first = row["regions.json"][0]
    first["english_text"] = "top"
    second = copy.deepcopy(first)
    second.update(
        region_id=3,
        english_text="bottom",
        bbox={"xmin": 5, "ymin": 90, "xmax": 250, "ymax": 150},
    )
    row["regions.json"] = [second, first]
    tasks, _ = derive_tasks(row, "en", REVISION)
    page = next(t for t in tasks if t["family"] == "page_ocr")
    assert page["reference"] == "top\n\nbottom"
    assert page["reading_order"] == [17, 3]
    assert score("page_ocr", "top\n\nbottom", page["reference"])[0] == 1.0
    assert score("page_ocr", "bottom top", page["reference"])[0] < 1.0


def test_overlapping_annotations_do_not_create_misleading_full_page_reference():
    row = next(fixture_rows("en"))
    duplicate = copy.deepcopy(row["regions.json"][0])
    duplicate["region_id"] = 3
    row["regions.json"].append(duplicate)
    tasks, skipped = derive_tasks(row, "en", REVISION)
    assert not any(t["family"] == "page_ocr" for t in tasks)
    assert skipped["page_ocr_overlapping_regions"] == 1


def test_full_page_preserves_canvas_and_masks_unannotated_text():
    row = next(fixture_rows("en"))
    with Image.open(io.BytesIO(row["jpg"]["bytes"])) as image:
        ImageDraw.Draw(image).rectangle([270, 5, 300, 40], fill="red")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
    row["jpg"]["bytes"] = buffer.getvalue()
    tasks, _ = derive_tasks(row, "en", REVISION)
    page = next(t for t in tasks if t["family"] == "page_ocr")
    with (
        Image.open(io.BytesIO(page["media"])) as masked,
        Image.open(io.BytesIO(row["jpg"]["bytes"])) as source,
    ):
        assert masked.size == source.size
        assert masked.getpixel((280, 20)) == (255, 255, 255)
        assert source.getpixel((280, 20)) != (255, 255, 255)
        assert (
            masked.crop((5, 5, 250, 80)).tobytes()
            == source.crop((5, 5, 250, 80)).tobytes()
        )


def test_playground_navigation_resets_reference_and_scores_selected_task(tmp_path):
    make_fixture(tmp_path)
    playground = Playground(Catalog(tmp_path))
    first = playground.choose("train", "ar", "page_ocr")
    assert first[6]["value"] == "" and first[6]["rtl"] is True
    reference = playground.catalog.get(first[0])["reference"]
    summary, revealed, metrics = playground.submit(first[0], reference)
    assert (
        revealed == reference and metrics["reward"] == 1.0 and "Exact match" in summary
    )
    second = playground.choose("train", "ar", "page_ocr", first[0], 1)
    assert first[0] != second[0] and second[6]["value"] == ""
    previous = playground.choose("train", "ar", "page_ocr", second[0], -1)
    assert previous[0] == first[0]
    for result in (first, second, previous):
        result[1].close()
    with pytest.raises(Exception, match="Load a task"):
        playground.submit("", "answer")
