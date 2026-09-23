"""Dataset identity, language scoring policy, and task derivation for FLEURS ASR."""

import hashlib
import json
import re
import unicodedata

REPO_ID = "google/fleurs"
BUCKET_ID = "FineEnvs/fleurs-bucket"
# The bucket holds a pinned copy, so the effective source identity is its content.
# This names the revision recorded in a snapshot and a frozen evaluation set; both
# must use the same value or the task IDs they derive cannot match.
DEFAULT_REVISION = "main"
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
    if not re.fullmatch(r"[a-z]{2,4}(_[A-Za-z0-9]+)+", language):
        raise ValueError(f"Unrecognized FLEURS language code: {language!r}")
    return language


def task_id(revision, language, split, sample_id, family):
    """A content-derived id that still says where to look it up.

    The digest pins the schema, revision, utterance, and family, so an id means the same
    task in any snapshot. The language and split are kept in plain text because a bare
    hash would have to be searched for across all 102 language indexes to be resolved.
    """
    return f"fleurs-{language}.{split}." + digest(
        [SCHEMA_VERSION, revision, language, split, int(sample_id), family]
    )


def parse_task_id(value):
    """Return (language, split) for a task id, without needing an index."""
    match = re.fullmatch(
        # A region segment is not always alphabetic: es_419 is Latin American Spanish.
        r"fleurs-([a-z]{2,4}(?:_[A-Za-z0-9]+)+)\.([a-z]+)\.[0-9a-f]{64}",
        value or "",
    )
    if not match:
        raise KeyError(f"Unrecognized task id: {value!r}")
    language, split = match[1], match[2]
    if split not in SPLITS:
        raise KeyError(f"Unknown split in task id: {value!r}")
    return language, split
