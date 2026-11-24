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

# The default rewards each script in its natural unit: words where words are spaced,
# characters where they are not. A word rate is coarse on an agglutinative language like
# Kannada, though. One Kannada word is a long, inflected string, so fixing three letters
# of it scores the same as leaving it wrong. A GRPO run on Kannada cut character errors
# by 14% while word errors moved 2%, and the reward hardly registered the improvement.
# The "cer" policy rewards characters for every language; the word rate stays in the
# metrics either way, so the two runs are still comparable.
REWARD_UNITS = ("script", "cer")
POLICIES = {"script": GRADING_POLICY, "cer": "fleurs-asr-cer-v1"}


def reward_unit(value=None):
    """The configured reward unit, from ASR_REWARD_UNIT unless one is given."""
    import os

    unit = value or os.environ.get("ASR_REWARD_UNIT") or "script"
    if unit not in REWARD_UNITS:
        raise ValueError(f"Unknown reward unit {unit!r}; use one of {REWARD_UNITS}")
    return unit


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


def character_error_rate(prediction, reference, language, family):
    """Character edit distance over the reference length, for any script.

    Unspaced scripts already score characters without whitespace. Spaced scripts keep
    their single spaces: a rate that ignored them would pay full marks for a transcript
    with its word boundaries removed.
    """
    if character_scored(language):
        return error_rate(prediction, reference, language, family)[0]
    target = " ".join(normalize_for_scoring(reference, family).split())
    if not target:
        raise ValueError("Reference is empty after normalization; cannot score")
    guess = " ".join(normalize_for_scoring(prediction, family).split())
    return Levenshtein.distance(guess, target) / len(target)


def score(task, prediction, unit="script"):
    """Grade one answer, returning the reward and the metrics behind it."""
    family, language = task["family"], task["language"]
    reference = task["reference"]
    if family == "language_id":
        # A code, not a transcript: normalization would only mask a wrong answer.
        correct = prediction.strip().casefold() == reference.strip().casefold()
        return float(correct), {"exact_match": correct}

    rate, natural = error_rate(prediction, reference, language, family)
    # Spaced scripts report both rates, whichever one the reward is computed from.
    cer = rate if natural == "cer" else character_error_rate(
        prediction, reference, language, family
    )
    exact = normalize_for_scoring(prediction, family) == normalize_for_scoring(
        reference, family
    )
    rewarded = cer if reward_unit(unit) == "cer" else rate
    reward = ERROR_WEIGHT * max(0.0, 1.0 - rewarded) + EXACT_WEIGHT * float(exact)
    # Metrics stay numeric so a trainer can average them; each key ('wer', 'cer') names
    # the unit it was measured in, and the observation names the one rewarded.
    return reward, {
        natural: rate,
        "cer": cer,
        "exact_match": exact,
        "reference_units": len(
            score_tokens(normalize_for_scoring(reference, family), language)
        ),
        "predicted_units": len(
            score_tokens(normalize_for_scoring(prediction, family), language)
        ),
    }
