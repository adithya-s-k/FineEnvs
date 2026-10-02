"""Reward for Sarvam Indic OCR Bench tasks, scored by the benchmark's own scorer.

A benchmark number is only comparable with published ones when it is computed the way
the benchmark defines it, so CER and WER come from the vendored metrics.py with its
content normalization (--normalize), unchanged. The reward is 1 - CER.

One deliberate departure: the official report drops empty predictions from its means.
That is reasonable for a leaderboard and wrong for a reward - it would make answering
nothing free - so an empty answer scores 0 here and carries missing_prediction, which
lets a summary exclude it again and reproduce the official numbers exactly.
"""

from ..data.indic_ocr_bench import REVISION
from . import indic_ocr_bench_metrics as official

POLICY = f"sarvam-indic-ocr-bench-metrics@{REVISION[:12]}"


def score_bench(prediction, reference):
    if official.is_missing_prediction(prediction):
        return 0.0, {
            "char_error_rate": 1.0,
            "word_error_rate": 1.0,
            "missing_prediction": True,
            "loop_or_catastrophic": False,
        }
    gt, pred = official.preprocess(
        reference, prediction, normalize=True, replace_n=False
    )
    rates = official.calculate_ocr_metrics(gt, pred)
    looped, _ = official.is_loop_or_catastrophic(reference, prediction)
    return 1.0 - rates["cer"], {
        "char_error_rate": rates["cer"],
        "word_error_rate": rates["wer"],
        "missing_prediction": False,
        "loop_or_catastrophic": looped,
    }
