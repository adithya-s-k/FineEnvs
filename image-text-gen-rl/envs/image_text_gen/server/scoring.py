"""Deterministic text-rendering reward from blind verifier transcriptions.

Each verifier reads the image without seeing the target and marks every malformed or
illegible glyph with MALFORMED. For each reading we find the span of the transcription
closest to the target (semi-global edit distance). Readings are then combined the way
the environment's contract promises:

* text accuracy comes from the best reading, so one verifier misreading a correct image
  does not cost the policy reward;
* malformed glyphs come from the worst reading, so one verifier repairing a broken
  glyph in its head does not hide it;
* spurious text outside the target comes from the most lenient reading.

reward = clip(text_accuracy * MALFORMED_DECAY**malformed - extra_penalty, 0, 1)
         * (1 - CASE_WEIGHT * (not case_match))
"""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

MALFORMED = "\ufffd"
MALFORMED_DECAY = 0.8
EXTRA_TOLERANCE = 2
EXTRA_WEIGHT = 0.3
CASE_WEIGHT = 0.1
MAX_TRANSCRIPT_CHARS = 1200
POLICY = {
    "name": "blind-transcription-v1",
    "malformed_marker": MALFORMED,
    "malformed_decay": MALFORMED_DECAY,
    "extra_tolerance": EXTRA_TOLERANCE,
    "extra_weight": EXTRA_WEIGHT,
    "case_weight": CASE_WEIGHT,
    "combine": {"text": "best", "malformed": "worst", "extra": "min"},
    "normalisation": "NFKC, typographic quotes/dashes to ASCII, whitespace collapsed",
}

_TYPOGRAPHIC = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2015": "-", "\u2212": "-",
    }
)
_REPLACEMENT_LOOKALIKES = re.compile("[\ufffc\u25a1\u25af\u2610]")  # object/box glyphs


def normalise(text):
    text = unicodedata.normalize("NFKC", text or "").translate(_TYPOGRAPHIC)
    text = _REPLACEMENT_LOOKALIKES.sub(MALFORMED, text)
    return re.sub(r"\s+", " ", text).strip()


def _lower(text):
    # Length-preserving lowercase so indices map between the two forms.
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def align(target, text):
    """Minimum edit distance from `target` to any substring of `text`.

    Returns (distance, start, end). Starting and ending anywhere in `text` is free, so
    unrelated text around the target does not count as an error here.
    """
    m = len(target)
    if not text:
        return m, 0, 0
    previous = [(0, j) for j in range(len(text) + 1)]  # (cost, span start)
    for i in range(1, m + 1):
        current = [(i, 0)]
        t = target[i - 1]
        for j in range(1, len(text) + 1):
            substitute = previous[j - 1][0] + (t != text[j - 1])
            delete = previous[j][0] + 1
            insert = current[j - 1][0] + 1
            best = min(substitute, delete, insert)
            if best == substitute:
                current.append((best, previous[j - 1][1]))
            elif best == delete:
                current.append((best, previous[j][1]))
            else:
                current.append((best, current[j - 1][1]))
        previous = current
    end = min(range(len(text) + 1), key=lambda j: (previous[j][0], -(j - previous[j][1])))
    return previous[end][0], previous[end][1], end


def score_reading(target, transcription):
    """Score one verifier's transcription against the target."""
    target_n = normalise(target)
    text_n = normalise(transcription)
    truncated = len(text_n) > MAX_TRANSCRIPT_CHARS
    text_n = text_n[:MAX_TRANSCRIPT_CHARS]
    distance, start, end = align(_lower(target_n), _lower(text_n))
    # Letters glued to the matched span belong to the rendered word: "Julyy" or "JJuly"
    # must not match "July" exactly with the stray letter excused as extra text.
    # Surrounding punctuation (quotes the prompt used) is left outside the span.
    while start > 0 and (text_n[start - 1].isalnum() or text_n[start - 1] == MALFORMED):
        start -= 1
    while end < len(text_n) and (text_n[end].isalnum() or text_n[end] == MALFORMED):
        end += 1
    span = text_n[start:end]
    distance = Levenshtein.distance(_lower(target_n), _lower(span))
    outside = text_n[:start] + text_n[end:]
    length = max(1, len(target_n))
    return {
        "text_accuracy": max(0.0, 1.0 - distance / length),
        "edit_distance": distance,
        "malformed_glyphs": span.count(MALFORMED),
        "extra_chars": sum(1 for c in outside if not c.isspace()),
        # Case errors only: edits that disappear when both sides are lowercased.
        "case_match": Levenshtein.distance(target_n, span)
        == Levenshtein.distance(_lower(target_n), _lower(span)),
        "exact_match": _lower(span) == _lower(target_n) and MALFORMED not in span,
        "span": span,
        "truncated": truncated,
    }


def combine(target, transcriptions):
    """Combine readings from several verifiers into one reward and flat metrics."""
    if not transcriptions:
        raise ValueError("At least one transcription is required")
    readings = [score_reading(target, text) for text in transcriptions]
    best_index = max(
        range(len(readings)),
        key=lambda i: (readings[i]["text_accuracy"], -readings[i]["extra_chars"]),
    )
    best = readings[best_index]
    malformed = max(r["malformed_glyphs"] for r in readings)
    extra = min(r["extra_chars"] for r in readings)
    length = max(1, len(normalise(target)))
    extra_penalty = EXTRA_WEIGHT * min(1.0, max(0, extra - EXTRA_TOLERANCE) / max(length, 10))
    core = best["text_accuracy"] * MALFORMED_DECAY**malformed - extra_penalty
    reward = max(0.0, min(1.0, core)) * (1.0 - CASE_WEIGHT * (not best["case_match"]))
    accuracies = [r["text_accuracy"] for r in readings]
    metrics = {
        "text_accuracy": round(best["text_accuracy"], 6),
        "char_error_rate": round(best["edit_distance"] / length, 6),
        "malformed_glyphs": malformed,
        "extra_chars": extra,
        "extra_penalty": round(extra_penalty, 6),
        "case_match": best["case_match"],
        "exact_match": best["exact_match"] and malformed == 0,
        "best_verifier": best_index,
        "verifier_spread": round(max(accuracies) - min(accuracies), 6),
        "no_text_read": all(not normalise(t) for t in transcriptions),
    }
    return round(reward, 6), metrics, readings
