"""Synthetic transport fixtures only; never a training or OCR quality benchmark."""

import io

from PIL import Image, ImageDraw

from .data.prepare import PrepareConfig, prepare
from .data.schema import split_for_page


def fixture_rows(language):
    found = {split: 0 for split in ("train", "validation", "test")}
    for document in range(1000):
        page_id = f"document_{document}_page_0"
        split = split_for_page(page_id)
        if found[split] == 2:
            continue
        found[split] += 1
        text = f"invoice {document}"
        with Image.new("RGB", (320, 160), "white") as image:
            ImageDraw.Draw(image).text((10, 10), text, fill="black")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
        yield {
            "jpg": {"bytes": buffer.getvalue(), "path": "00000000.jpg"},
            "image_id.txt": page_id,
            "regions.json": [
                {
                    "region_id": 17,
                    "layout_type": "text",
                    "english_text": text,
                    "translated_text": text,
                    "bbox": {"xmin": 5, "ymin": 5, "xmax": 250, "ymax": 80},
                }
            ],
            "vqa.json": {
                "questions": [
                    {
                        "type": "mcq",
                        "question": "What type of document is shown?",
                        "options": ["letter", "invoice"],
                        "answer": "invoice",
                    }
                ]
            },
        }
        if all(value == 2 for value in found.values()):
            break


def fixture_stream(config, language):
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        fixture_rows, gen_kwargs={"language": language}
    )


def make_fixture(directory):
    return prepare(
        directory,
        PrepareConfig(
            languages=("en", "kn", "hi", "ar"),
            pages_per_language=6,
            source="synthetic-fixture",
        ),
        source_factory=fixture_stream,
    )
