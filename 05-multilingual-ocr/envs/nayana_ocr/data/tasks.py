"""Derive task records; keep references out of every public representation."""

import io
import json
import math
from collections import Counter

from PIL import Image

from .reading_order import ordered_regions, overlapping_regions
from .schema import (
    FAMILIES,
    LAYOUT_LABELS,
    READING_ORDER,
    document_id,
    normalize_text,
    split_for_page,
    task_id,
)


def derive_tasks(
    row,
    language,
    revision,
    split_seed=42,
    max_pixels=50_000_000,
    *,
    metadata_only=False,
    selection=None,
):
    page_id = row["image_id.txt"]
    doc_id = document_id(page_id)
    raw, image, width, height = None, None, None, None
    if not metadata_only:
        raw = row["jpg"]["bytes"]
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("Expected embedded image bytes with Image(decode=False)")
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width * height > max_pixels:
            image.close()
            raise ValueError(
                f"{page_id}: {width}x{height} exceeds max_pixels={max_pixels}"
            )
        if image.format != "JPEG":
            image.close()
            raise ValueError(f"{page_id}: expected native JPEG, got {image.format}")

    def render(family, unit):
        return not metadata_only and (
            selection is None or selection == (family, str(unit))
        )

    tasks, skipped, page_regions = [], Counter(), []

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
            if not (0 <= x0 < x1 and 0 <= y0 < y1) or (
                width is not None and (x1 > width or y1 > height)
            ):
                skipped["invalid_bbox"] += 1
                continue
            bbox = [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]
            page_regions.append({"region_id": unit, "bbox": bbox, "text": reference})
            media = None
            if render("section_ocr", unit):
                with image.crop(bbox) as crop:
                    buffer = io.BytesIO()
                    crop.save(buffer, format="PNG")
                    media = buffer.getvalue()
            add(
                FAMILIES[0],
                unit,
                f"Transcribe the text in this cropped document region (language: {language}). "
                "Return only the text, preserving its language and punctuation.",
                reference,
                media,
                "image/png",
                (bbox[2] - bbox[0], bbox[3] - bbox[1]),
                bbox,
            )

        for index, question in enumerate(row["vqa.json"].get("questions", [])):
            if question.get("type") == "descriptive":
                answer, query = question.get("answer"), question.get("question")
                if not all(
                    isinstance(v, str) and normalize_text(v) for v in (answer, query)
                ):
                    skipped["invalid_descriptive_vqa"] += 1
                    continue
                # Bound the judge input without truncating the authoritative answer.
                if len(answer) > 8192 or len(query) > 8192:
                    skipped["descriptive_vqa_context_limit"] += 1
                    continue
                add(
                    "descriptive_vqa",
                    index,
                    query
                    + f"\n\nAnswer in {language}. Be concise and complete; do not add unsupported claims.",
                    answer,
                    raw,
                    "image/jpeg",
                    (width, height),
                )
                continue
            if question.get("type") != "mcq":
                skipped["unsupported_vqa_type"] += 1
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
            add(FAMILIES[1], index, prompt, label, raw, "image/jpeg", (width, height))
        # Full-page supervision must include every supplied text annotation. Do
        # not turn a partially valid page into an apparently complete target.
        if not regions or len(page_regions) != len(regions):
            skipped["page_ocr_incomplete_annotations"] += 1
        elif overlapping_regions(page_regions):
            skipped["page_ocr_overlapping_regions"] += 1
        else:
            rtl = language == "ar"
            ordered = ordered_regions(page_regions, rtl=rtl)
            direction = "right to left" if rtl else "left to right"
            # Nayana can retain visible source headers outside its text annotations.
            # Preserve the full canvas while masking unsupervised areas so a correct
            # transcription is not penalized for text absent from the reference.
            media = None
            if render("page_ocr", "page"):
                with Image.new("RGB", image.size, "white") as masked:
                    for region in page_regions:
                        with image.crop(region["bbox"]) as crop:
                            masked.paste(crop, tuple(region["bbox"][:2]))
                    buffer = io.BytesIO()
                    masked.save(buffer, format="PNG")
                    media = buffer.getvalue()
            add(
                "page_ocr",
                "page",
                f"Transcribe all visible text on this full document page (language: {language}). "
                "Unannotated areas have been masked while preserving the page layout. "
                f"Read separate columns {direction}, and read text within each column from top to bottom. "
                "Read spanning headings before the body and spanning footers after it. "
                "Preserve language and punctuation. Separate regions with a blank line. "
                "Return only the transcription; do not describe images or reconstruct table formatting.",
                "\n\n".join(region["text"] for region in ordered),
                media,
                "image/png",
                (width, height),
            )
            tasks[-1]["reading_order_policy"] = READING_ORDER
            tasks[-1]["reading_order"] = [region["region_id"] for region in ordered]
            tasks[-1]["annotation_masked"] = True

        # Layout supervision includes non-text regions, independently of OCR eligibility.
        # Require a complete valid annotation set; overlapping regions are allowed.
        layout = []
        for region in regions:
            box = region.get("bbox", {})
            coords = [box.get(k) for k in ("xmin", "ymin", "xmax", "ymax")]
            if region.get("layout_type") not in LAYOUT_LABELS or not all(
                type(v) in (int, float) and math.isfinite(v) for v in coords
            ):
                break
            x0, y0, x1, y1 = coords
            if not (0 <= x0 < x1 and 0 <= y0 < y1) or (
                width is not None and (x1 > width or y1 > height)
            ):
                break
            layout.append({"label": region["layout_type"], "bbox": coords})
        if regions and len(layout) == len(regions) and len(layout) <= 512:
            add(
                "layout_detection",
                "page",
                "Detect the annotated document layout regions. Return only a JSON array of "
                '{"label":"text","bbox":[xmin,ymin,xmax,ymax]} objects. '
                "Use absolute pixel coordinates in the supplied image, with origin at its top left. "
                f"Allowed labels: {', '.join(LAYOUT_LABELS)}. Include every region once; no extra keys or prose. "
                "These labels describe document regions, not arbitrary objects."
                + (
                    f" Coordinate canvas: width {width} pixels, height {height} pixels."
                    if width is not None
                    else ""
                ),
                json.dumps(layout, ensure_ascii=False, separators=(",", ":")),
                raw,
                "image/jpeg",
                (width, height),
            )
        else:
            skipped["layout_incomplete_or_invalid_annotations"] += 1
    finally:
        if image is not None:
            image.close()
    return tasks, skipped
