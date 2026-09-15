"""Verifiable per-family rewards. CER counts Unicode code points after NFC."""

from rapidfuzz.distance import Levenshtein

from ..data.schema import normalize_text


def score(family, prediction, reference):
    # Check raw length first, so whitespace padding cannot bypass the guard.
    overlong = len(prediction) > max(1024, 4 * len(reference))
    if family == "mcq_vqa":
        exact = prediction.strip() == reference and not overlong
        return float(exact), {"exact_match": exact, "overlong": overlong}
    if family not in {"section_ocr", "page_ocr"}:
        raise ValueError(f"Unsupported task family: {family}")
    predicted, target = normalize_text(prediction), normalize_text(reference)
    if not target:
        raise ValueError("OCR reference must not be empty")
    cer = Levenshtein.distance(predicted, target) / len(target)
    exact = predicted == target and not overlong
    reward = 0.0 if overlong else 0.8 * max(0.0, 1.0 - cer) + 0.2 * exact
    return reward, {
        "exact_match": exact,
        "char_error_rate": cer,
        "overlong": overlong,
    }
