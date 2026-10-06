"""Sarvam Indic OCR Bench's own scores, reported beside the standard OCR reward.

Benchmark crops are graded exactly like every other section-OCR task: the reward is the
corpus OCR reward, so numbers are comparable across sources and a policy is never
trained or judged on two definitions of "correct". What the benchmark adds is its own
CER and WER, computed by its vendored metrics.py with content normalization, so results
can also be read against published ones. These are reported, never rewarded.

The official report drops empty predictions from its means and keeps runaway outputs
out of its valid-sample figures; the two flags here let a summary do the same.
"""

from ..data.indic_ocr_bench import VERSION
from . import indic_ocr_bench_metrics as official

POLICY = f"sarvam-indic-ocr-bench-metrics@{VERSION}"


def official_metrics(prediction, reference):
    if official.is_missing_prediction(prediction):
        return {
            "official_cer": 1.0,
            "official_wer": 1.0,
            "missing_prediction": True,
            "loop_or_catastrophic": False,
        }
    gt, pred = official.preprocess(
        reference, prediction, normalize=True, replace_n=False
    )
    rates = official.calculate_ocr_metrics(gt, pred)
    looped, _ = official.is_loop_or_catastrophic(reference, prediction)
    return {
        "official_cer": rates["cer"],
        "official_wer": rates["wer"],
        "missing_prediction": False,
        "loop_or_catastrophic": looped,
    }
