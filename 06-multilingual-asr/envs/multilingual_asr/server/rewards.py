"""ASR scoring: script-aware error rate, reported alongside the rate it was computed in."""

from rapidfuzz.distance import Levenshtein

from ..data.schema import (
    character_scored,
    normalize_for_scoring,
    score_tokens,
)

# Matches the OCR environment's shape so a multi-environment run reads consistently:
# most of the reward is continuous in the error rate, a slice rewards getting it exactly.
ERROR_WEIGHT = 0.8
EXACT_WEIGHT = 0.2
GRADING_POLICY = "fleurs-asr-error-rate-v1"


def error_rate(prediction, reference, language, family):
    """Edit distance over score units, divided by the reference length.

    Returns the rate and the unit it was measured in, because a word rate and a
    character rate are not comparable and a score that does not say which it is cannot
    be read. An empty reference cannot define a rate, so it is rejected rather than
    silently scored as perfect or as total failure.
    """
    unit = "cer" if character_scored(language) else "wer"
    target = score_tokens(normalize_for_scoring(reference, family), language)
    if not target:
        raise ValueError("Reference is empty after normalization; cannot score")
    guess = score_tokens(normalize_for_scoring(prediction, family), language)
    # Unbounded above: inserting freely can push the rate past 1, which the reward clamps.
    return Levenshtein.distance(guess, target) / len(target), unit


def score(task, prediction):
    """Grade one answer, returning the reward and the metrics behind it."""
    family, language = task["family"], task["language"]
    reference = task["reference"]
    if family == "language_id":
        # A code, not a transcript: normalization would only mask a wrong answer.
        correct = prediction.strip().casefold() == reference.strip().casefold()
        return float(correct), {"exact_match": correct}

    rate, unit = error_rate(prediction, reference, language, family)
    exact = normalize_for_scoring(prediction, family) == normalize_for_scoring(
        reference, family
    )
    reward = ERROR_WEIGHT * max(0.0, 1.0 - rate) + EXACT_WEIGHT * float(exact)
    # Metrics stay numeric so a trainer can average them; the key itself ('wer' or
    # 'cer') names the unit, and the observation carries it as a typed field.
    return reward, {
        unit: rate,
        "exact_match": exact,
        "reference_units": len(
            score_tokens(normalize_for_scoring(reference, family), language)
        ),
        "predicted_units": len(
            score_tokens(normalize_for_scoring(prediction, family), language)
        ),
    }
