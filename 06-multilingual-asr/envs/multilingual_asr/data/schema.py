"""Dataset identity, language scoring policy, and task derivation for FLEURS ASR."""

import hashlib
import json
import re
import unicodedata

REPO_ID = "google/fleurs"
BUCKET_ID = "FineEnvs/fleurs-bucket"
SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
FAMILIES = ("transcription", "verbatim_transcription", "language_id")

# FLEURS ships its own train/validation/test splits per language. They are used as
# published: a re-split would silently break comparison with every reported FLEURS score.
NATIVE_SPLITS = True

# Scripts that do not delimit words with spaces. Word error rate is meaningless for them -
# a reference is one "word" and any error scores 1.0 - so they are scored per character.
# This is the standard FLEURS/Whisper treatment, not a convenience.
CHARACTER_SCORED = frozenset(
    {
        "cmn_hans_cn",  # Mandarin
        "yue_hant_hk",  # Cantonese
        "ja_jp",  # Japanese
        "th_th",  # Thai
        "lo_la",  # Lao
        "my_mm",  # Burmese
        "km_kh",  # Khmer
    }
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def character_scored(language):
    return language in CHARACTER_SCORED


def normalize_for_scoring(text, family):
    """Normalize a transcript before scoring.

    Applied identically to the prediction and the reference, so it can only remove a
    difference both sides agree is not an error.

    `transcription` targets FLEURS' normalized field, which is already lowercased and
    stripped of punctuation, so scoring removes punctuation and case: a model should not
    lose reward for writing a comma the reference never had. `verbatim_transcription`
    targets `raw_transcription`, where punctuation and case *are* the task, so only
    Unicode form and whitespace are normalized.
    """
    text = unicodedata.normalize("NFKC", text)
    if family == "transcription":
        text = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in text
        )
        text = text.casefold()
    return " ".join(text.split())


def score_tokens(text, language):
    """Split a normalized transcript into the units its script is scored in."""
    if character_scored(language):
        # Spaces are not word boundaries in these scripts; compare the characters.
        return [c for c in text if not c.isspace()]
    return text.split()


def language_code(language):
    if not re.fullmatch(r"[a-z]{2,4}(_[A-Za-z]+)+", language):
        raise ValueError(f"Unrecognized FLEURS language code: {language!r}")
    return language


def task_id(revision, language, split, sample_id, family):
    return "fleurs-" + digest(
        [SCHEMA_VERSION, revision, language, split, int(sample_id), family]
    )
