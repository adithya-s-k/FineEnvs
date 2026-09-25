"""Dataset identity, document-level partitioning, and multilingual scoring policy."""

import hashlib
import json
import re
import unicodedata

REPO_ID = "Cognitive-Lab/NayanaOCR_Corpus_2025"
REVISION = "b220b074a8c82bb90427051e856e4c4edc79885b"
LANGUAGES = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "gu",
    "hi",
    "it",
    "ja",
    "kn",
    "ko",
    "ml",
    "mr",
    "or",
    "pa",
    "ru",
    "sa",
    "ta",
    "te",
    "th",
    "zh",
)
FAMILIES = ("section_ocr", "mcq_vqa", "page_ocr", "layout_detection", "descriptive_vqa")
SCHEMA_VERSION = 3
LAYOUT_LABELS = ("text", "title", "caption", "table", "image", "formula")
READING_ORDER = "whitespace-columns-v1"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_text(text):
    # Preserve case, punctuation, vowel signs, ZWJ/ZWNJ, and RTL text order.
    return " ".join(unicodedata.normalize("NFC", text).split())


def document_id(page_id):
    match = re.fullmatch(r"(document_[0-9]+)_page_[0-9]+", page_id)
    if not match:
        raise ValueError(f"Unrecognized page ID: {page_id!r}; cannot partition safely")
    return match[1]


def split_for_page(page_id, seed=42):
    # Same document, all its pages, languages, regions, and questions stay together.
    bucket = (
        int(digest(["document-split-v1", seed, document_id(page_id)])[:16], 16) % 100
    )
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def task_id(revision, language, page_id, family, unit):
    return "nayana-" + digest(
        [SCHEMA_VERSION, revision, language, page_id, family, unit]
    )
