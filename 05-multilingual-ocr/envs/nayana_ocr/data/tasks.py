"""Derive task records; keep references out of every public representation."""

import io
import math
from collections import Counter

from PIL import Image

from .schema import FAMILIES, document_id, normalize_text, split_for_page, task_id


def derive_tasks(row, language, revision, split_seed=42, max_pixels=50_000_000):
    page_id = row["image_id.txt"]
    doc_id = document_id(page_id)
    raw = row["jpg"]["bytes"]
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Expected embedded image bytes with Image(decode=False)")
    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    if width * height > max_pixels:
        raise ValueError(f"{page_id}: {width}x{height} exceeds max_pixels={max_pixels}")
    if image.format != "JPEG":
        raise ValueError(f"{page_id}: expected native JPEG, got {image.format}")
    tasks, skipped = [], Counter()

    def add(family, unit, prompt, reference, media, mime, size, bbox=None):
        tasks.append(
            {
                "task_id": task_id(revision, language, page_id, family, unit),
                "language": language,
                "page_id": page_id,
                "document_id": doc_id,
                "family": family,
                "unit": str(unit),
                "split": split_for_page(page_id, split_seed),
                "prompt": prompt,
                "reference": reference,
                "media": media,
                "mime": mime,
                "width": size[0],
                "height": size[1],
                "bbox": bbox,
                "page_width": width,
                "page_height": height,
            }
        )

    try:
        regions = row["regions.json"]
        ids = [region.get("region_id") for region in regions]
        for region in regions:
            unit = region.get("region_id")
            reference = region.get(
                "english_text" if language == "en" else "translated_text"
            )
            if unit is None or ids.count(unit) != 1:
                skipped["duplicate_or_missing_region_id"] += 1
                continue
            if not isinstance(reference, str) or not normalize_text(reference):
                skipped["empty_region_text"] += 1
                continue
            box = region.get("bbox", {})
            coords = [box.get(key) for key in ("xmin", "ymin", "xmax", "ymax")]
            if not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in coords
            ):
                skipped["invalid_bbox"] += 1
                continue
            x0, y0, x1, y1 = coords
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                skipped["invalid_bbox"] += 1
                continue
            bbox = [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]
            with image.crop(bbox) as crop:
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")  # lossless crop, no page downscaling
                add(
                    FAMILIES[0],
                    unit,
                    f"Transcribe the text in this cropped document region (language: {language}). "
                    "Return only the text, preserving its language and punctuation.",
                    reference,
                    buffer.getvalue(),
                    "image/png",
                    crop.size,
                    bbox,
                )

        for index, question in enumerate(row["vqa.json"].get("questions", [])):
            if question.get("type") != "mcq":
                skipped["descriptive_vqa_deferred"] += 1
                continue
            options = question.get("options")
            answer = question.get("answer")
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 26
                or not all(
                    isinstance(option, str) and normalize_text(option)
                    for option in options
                )
                or not isinstance(answer, str)
                or not question.get("question")
            ):
                skipped["invalid_mcq"] += 1
                continue
            normalized = [normalize_text(option) for option in options]
            target = normalize_text(answer)
            if len(set(normalized)) != len(options) or normalized.count(target) != 1:
                skipped["ambiguous_mcq_answer"] += 1
                continue
            label = chr(65 + normalized.index(target))
            prompt = (
                question["question"]
                + "\n\n"
                + "\n".join(
                    f"{chr(65 + i)}. {option}" for i, option in enumerate(options)
                )
            )
            prompt += (
                "\n\nReturn only the single uppercase letter of the correct option."
            )
            add(FAMILIES[1], index, prompt, label, raw, "image/jpeg", image.size)
    finally:
        image.close()
    return tasks, skipped
