# Vendored without modification from Sarvam AI's Sarvam Indic OCR Bench:
#   https://huggingface.co/datasets/sarvamai/indic-ocr-bench/blob/84ce7ce447456a92bcbf25f3c0a55d6a5a44a24b/metrics.py
# Copyright Sarvam AI. Licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). The benchmark defines its scores with
# this file, so it is kept byte-for-byte below this header; diff it against the pinned
# revision before changing anything. Our reward wrapper lives in bench_rewards.py.
#!/usr/bin/env python3
"""
Standalone OCR metrics (CER/WER) for Sarvam Indic OCR Bench predictions.

Self-contained: no repo-local imports, no third-party packages (stdlib only).

With ``--normalize`` the pipeline applies the base normalization plus the
"content" normalization folds merged from ``benchmark_normalization.py`` (HTML,
quotes, asterisks, bullets/list markers, ZWJ/ZWNJ, filler rules, and cosmetic
spacing are stripped so CER/WER reflect word/character OCR errors).

With ``--remove-br`` any ``<br>`` / ``<br/>`` / ``<br />`` tags are replaced
with a space in both gt and pred before the rest of the pipeline.

Input format (list of objects):

    [
      {
        "image_name": "sample_id",
        "gt": "ground truth text",
        "pred": "model prediction"
      },
      ...
    ]

Usage:
    python metrics.py --input results/gemini-3.5-flash-thinkingoff.gt_pred.json --normalize --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from statistics import mean

# ---------------------------------------------------------------------------
# Benchmark normalization
# ---------------------------------------------------------------------------

ERROR_PRED_RE = re.compile(r"^\[ERROR:")
BULLET_CHARS = "•●◦▪▫◆◇○●□■➢➤➔→▸▹-*"

PUNCTUATIONS = (
    r"\u0021-\u002F"
    r"\u003A-\u0040"
    r"\u005B-\u0060"
    r"\u007B-\u007E"
    r"\u0964\u0965"
    r"\u0E4F"
    r"\u2013-\u2015"
    r"\u2018-\u201F"
    r"\u2026"
    r"\u2030-\u203E"
    r"\u20A0-\u20CF"
    r"\u2100-\u214F"
    r"\u2190-\u21FF"
    r"\u2300-\u23FF"
    r"\u25A0-\u25FF"
    r"\u2600-\u26FF"
    r"\u2700-\u27BF"
)
PUNCT_SPACE_RE = re.compile(rf"\s+([{PUNCTUATIONS}])")


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+|\\n", text or "") if w.strip()])


def apply_replace_n(text: str) -> str:
    return (text or "").replace("\n", " ").replace("\\n", " ")


def fold_quotes(text: str) -> str:
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def fold_dashes(text: str) -> str:
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2015", "-")


def normalize_indic_punctuation(text: str) -> str:
    text = text.replace("||", "॥")
    text = re.sub(
        r"(?<=[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
        r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF"
        r"\u0E00-\u0E7Fa-zA-Z0-9])\|(?=[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F"
        r"\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
        r"\u0D00-\u0D7F\u0D80-\u0DFF\u0E00-\u0E7Fa-zA-Z0-9])",
        "।",
        text,
    )
    text = re.sub(r"\s*॥\s*", " ॥ ", text)
    text = re.sub(r"\s*।\s*", " । ", text)
    return text


def normalize_period_comma_spacing(text: str) -> str:
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\.\s+", ". ", text)
    text = re.sub(r",\s+", ", ", text)
    text = re.sub(r"\s+;", "; ", text)
    text = re.sub(r"\s+!", "! ", text)
    text = re.sub(r"\s+\?", "? ", text)
    return text


def normalize_bullets(text: str) -> str:
    text = re.sub(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(rf"(?<=\n)\s*[{re.escape(BULLET_CHARS)}]\s*", "- ", text)
    text = re.sub(r"(\d+)\)\s+", r"\1) ", text)
    text = re.sub(r"(\d+)\.\s+", r"\1. ", text)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_metrics(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = apply_replace_n(text)
    text = fold_quotes(text)
    text = fold_dashes(text)
    text = normalize_indic_punctuation(text)
    text = normalize_period_comma_spacing(text)
    text = normalize_bullets(text)
    return collapse_whitespace(text)


def normalize_for_scoring(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = unicodedata.normalize("NFC", text)
    text = PUNCT_SPACE_RE.sub(r"\1", text)
    return collapse_whitespace(text)


# ---------------------------------------------------------------------------
# Content normalization (merged from benchmark_normalization.py, stdlib-only)
#
# The "content" folds strip markup/formatting noise (HTML, quotes, asterisks,
# bullets, list markers, ZWJ/ZWNJ, filler rules, cosmetic spacing) so that
# CER/WER reflect word/character OCR errors rather than formatting differences.
# ---------------------------------------------------------------------------

_FILLER_LINE_CHARS = r"\-\._=~─━┄┅┈┉╌╍═"

_BULLET_CHARS = (
    "•▪▫●○◦·‣⁃∙◆♦▸►▹➢➣➤→*"
    "■□▣❑❖✦★☐⮚⇒◈◎⇨☞✽▢◼◾◆◇○◎"
    "\u2022\u2023\u2043\u2219\u25aa\u25ab\u25b8\u25cf\u25e6\u27a2\u27a4"
    "\u25a0\u25a1\u25a3\u2751\u2756\u2726\u2605\u2610\u2b9a"
    "\u21d2\u25c8\u25ce\u21e8\u261e\u273d"
)


def nfkc_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKC", text))


def strip_markdown_bold(text: str) -> str:
    return text.replace("**", "")


def collapse_long_char_runs(text: str, *, min_run: int = 5, target_len: int = 3) -> str:
    if min_run <= target_len:
        return text

    def _repl(match: re.Match[str]) -> str:
        return match.group(1) * target_len

    return re.sub(rf"(.)\1{{{min_run - 1},}}", _repl, text)


def strip_filler_char_runs(text: str, *, min_run: int = 2) -> str:
    text = text or ""
    if min_run < 2:
        return text
    return re.sub(rf"([{_FILLER_LINE_CHARS}])\1{{{min_run - 1},}}", " ", text)


def _content_base_normalize(text: str) -> str:
    """Base normalization used by the content pipeline (danda/quote/dash folds)."""
    text = nfkc_nfc(text)
    text = strip_markdown_bold(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[\u2018\u2019\u0060\u00B4]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"\s+([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~\u0964\u0965])", r"\1", text)
    text = collapse_long_char_runs(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_bullet_points(text: str) -> str:
    text = text or ""
    bullet_class = re.escape(_BULLET_CHARS)
    text = re.sub(rf"[{bullet_class}]+", " ", text)
    text = re.sub(r"(?:^|\s)\d+[\.\):](?=\s)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_html_tags(text: str) -> str:
    text = text or ""
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9]*\b[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_zwj_zwnj(text: str) -> str:
    return (text or "").replace("\u200c", "").replace("\u200d", "")


def collapse_paren_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def collapse_bracket_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    return text


def collapse_quote_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r'(["\u201c\u201d])\s+', r"\1", text)
    text = re.sub(r'\s+(["\u201c\u201d])', r"\1", text)
    text = re.sub(r"(['\u2018\u2019])\s+", r"\1", text)
    text = re.sub(r"\s+(['\u2018\u2019])", r"\1", text)
    return text


def strip_quotes(text: str) -> str:
    return re.sub(r"[\"'“”‘’`´]", "", text or "")


def strip_asterisk_variants(text: str) -> str:
    return re.sub(r"[*∗✱⁎⋆＊]", "", text or "")


def fold_literal_escapes(text: str) -> str:
    text = text or ""
    return text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")


def unify_danda_pipe(text: str) -> str:
    text = text or ""
    text = text.replace("||", "॥").replace("|", "।")
    return text


def collapse_space_around_punct_ops(text: str) -> str:
    text = text or ""
    text = re.sub(r"([,.;:!?।॥/%+\-=])\s+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?।॥/%+\-=])", r"\1", text)
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s+-", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_spaced_single_chars(text: str) -> str:
    text = text or ""

    def _repl(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return re.sub(
        r"(?<!\S)([A-Za-z0-9])(?:\s+[A-Za-z0-9]){2,}(?!\S)",
        _repl,
        text,
    )


def is_word_split_or_merge(gt: str, pred: str) -> bool:
    gt_c = re.sub(r"\s+", "", gt or "")
    pr_c = re.sub(r"\s+", "", pred or "")
    if gt_c != pr_c or not gt_c:
        return False

    def _has_letter_or_digit(tok: str) -> bool:
        for ch in tok:
            if ch.isalnum():
                return True
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("N"):
                return True
        return False

    gw = [w for w in (gt or "").split() if _has_letter_or_digit(w)]
    pw = [w for w in (pred or "").split() if _has_letter_or_digit(w)]
    if gw == pw:
        return False
    if abs(len(gw) - len(pw)) != 1:
        return False
    return "".join(gw) == "".join(pw)


def equalize_space_only_issues(gt: str, pred: str) -> tuple[str, str]:
    gt = gt or ""
    pred = pred or ""
    if gt == pred:
        return gt, pred
    if re.sub(r"\s+", "", gt) != re.sub(r"\s+", "", pred):
        return gt, pred

    def _fold(t: str) -> str:
        t = collapse_space_around_punct_ops(t)
        t = collapse_spaced_single_chars(t)
        return re.sub(r"\s+", " ", t).strip()

    gt2, pred2 = _fold(gt), _fold(pred)
    if gt2 == pred2:
        return gt2, pred2
    if re.sub(r"\s+", "", gt2) != re.sub(r"\s+", "", pred2):
        return gt, pred
    if is_word_split_or_merge(gt2, pred2):
        return gt2, pred2
    return gt2, gt2


def normalize_for_wer_content_extras(text: str) -> str:
    text = text or ""
    text = strip_zwj_zwnj(text)
    text = strip_quotes(text)
    text = strip_asterisk_variants(text)
    text = strip_bullet_points(text)
    text = strip_filler_char_runs(text, min_run=2)
    text = collapse_paren_whitespace(text)
    text = collapse_bracket_whitespace(text)
    text = collapse_quote_whitespace(text)
    text = unify_danda_pipe(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_content(text: str) -> str:
    """Single-side content normalization (wer_content / cer_content folds)."""
    text = strip_html_tags(text or "")
    text = fold_literal_escapes(text)
    text = strip_quotes(text)
    text = strip_asterisk_variants(text)
    base = normalize_for_scoring(_content_base_normalize(text))
    return normalize_for_wer_content_extras(base)


def strip_list_dash_markers(text: str) -> str:
    """Drop list-dash markers ("- ") left when the base step rewrites bullets.

    The base ``normalize_bullets`` turns bullet glyphs into ``"- "`` while the
    content folds keep ASCII hyphens (real mid-word hyphens are content). Remove
    only dash markers at a token boundary ("^- " or " - "); keeps "cap-ture".
    """
    text = re.sub(r"(?:^|\s)-\s+", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def remove_br_tags(text: str) -> str:
    """Replace ``<br>`` / ``<br/>`` / ``<br />`` (any case) with a space."""
    return re.sub(r"<br\s*/?>", " ", text or "", flags=re.IGNORECASE)


def preprocess(
    gt: str,
    pred: str,
    *,
    normalize: bool,
    replace_n: bool,
    remove_br: bool = False,
) -> tuple[str, str]:
    if remove_br:
        gt = remove_br_tags(gt)
        pred = remove_br_tags(pred)
    if normalize:
        # Existing normalization, then the merged content-normalization folds.
        gt = normalize_for_metrics(gt)
        pred = normalize_for_metrics(pred)
        gt = normalize_for_content(gt)
        pred = normalize_for_content(pred)
        gt = strip_list_dash_markers(gt)
        pred = strip_list_dash_markers(pred)
        # Pairwise: fold cosmetic space-only differences (keeps true split/merge).
        gt, pred = equalize_space_only_issues(gt, pred)
    elif replace_n:
        gt = apply_replace_n(gt)
        pred = apply_replace_n(pred)
    return normalize_for_scoring(gt), normalize_for_scoring(pred)


# ---------------------------------------------------------------------------
# Loop / catastrophic detection
# ---------------------------------------------------------------------------


def detect_tail_word_loop(text: str, min_repeats: int = 4) -> bool:
    words = (text or "").split()
    for n in (1, 2, 3, 5):
        if len(words) < n * min_repeats:
            continue
        tail = words[-n:]
        reps = 0
        i = len(words)
        while i >= n:
            if words[i - n : i] == tail:
                reps += 1
                i -= n
            else:
                break
        if reps >= min_repeats:
            return True
    return False


def detect_tail_char_loop(text: str, min_repeats: int = 8) -> bool:
    text = text or ""
    if len(text) < min_repeats * 2:
        return False
    for plen in range(1, min(40, len(text) // min_repeats)):
        pat = text[-plen:]
        if len(set(pat.strip())) > 4:
            continue
        count = 0
        pos = len(text)
        while pos >= plen and text[pos - plen : pos] == pat:
            count += 1
            pos -= plen
        if count >= min_repeats:
            return True
    return False


def detect_tail_block_loop(text: str, min_block: int = 12, min_repeats: int = 3,
                           max_block: int = 500) -> bool:
    """Detect a block (>= min_block chars) repeated >= min_repeats times at the end.

    Non-regex replacement for the old ``(.{12,}?)(?:\\s*\\1){2,}$`` pattern, which
    caused catastrophic backtracking on very long repetitive loop predictions.
    """
    text = text or ""
    n = len(text)
    if n < min_block * min_repeats:
        return False
    upper = min(max_block, n // min_repeats)
    for L in range(min_block, upper + 1):
        block = text[-L:]
        if text[-L * min_repeats:] == block * min_repeats:
            return True
    return False


def detect_tail_loop(pred: str) -> bool:
    if ERROR_PRED_RE.match(pred or ""):
        return True
    if detect_tail_word_loop(pred):
        return True
    if detect_tail_char_loop(pred):
        return True
    # Old (kept for reference): backreference regex caused catastrophic
    # backtracking / hangs on very long repetitive loop predictions.
    # if re.search(r"(.{12,}?)(?:\s*\1){2,}$", pred or ""):
    #     return True
    if detect_tail_block_loop(pred):
        return True
    return False


def detect_length_explosion(gt: str, pred: str) -> bool:
    gt_w, pred_w = word_count(gt), word_count(pred)
    if gt_w > 0 and pred_w >= 2 * gt_w and pred_w >= 20:
        return True
    if len(pred) >= 8000 and len(gt) < 4000:
        return True
    return False


def is_loop_or_catastrophic(gt: str, pred: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if ERROR_PRED_RE.match(pred or ""):
        reasons.append("pipeline_error")
    if detect_tail_loop(pred):
        reasons.append("tail_loop")
    if detect_length_explosion(gt, pred):
        reasons.append("length_explosion")
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------


try:  # C-backed Levenshtein (identical result, ~100x faster on long pages)
    import editdistance as _editdistance
except Exception:  # pragma: no cover
    _editdistance = None


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    if _editdistance is not None:
        return int(_editdistance.eval(a, b))

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def tokenize_words(text: str) -> list[str]:
    text = normalize_for_scoring(text)
    return [tok for tok in text.split() if tok]


def _bound_rate(value: float) -> float:
    """Cap an error rate at 1.0 (100%) so runaway samples can't dominate means."""
    return min(1.0, float(value))


def calculate_cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_for_scoring(reference))
    hyp = list(normalize_for_scoring(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _bound_rate(edit_distance(ref, hyp) / len(ref))


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _bound_rate(edit_distance(ref, hyp) / len(ref))


def calculate_ocr_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    return {
        "wer": calculate_wer(reference, hypothesis),
        "cer": calculate_cer(reference, hypothesis),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_gt_pred_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    rows: list[dict] = []
    for index, item in enumerate(data):
        image_name = item.get("image_name") or item.get("image")
        gt = item.get("gt")
        if gt is None:
            gt = item.get("ground_truth")
        pred = item.get("pred")
        if pred is None:
            pred = item.get("prediction")
        # Nested {"data": [{"ground_truth", "prediction"}], ...} shape.
        if (gt is None or pred is None) and isinstance(item.get("data"), list) and item["data"]:
            inner = item["data"][0]
            if gt is None:
                gt = inner.get("ground_truth") or inner.get("gt")
            if pred is None:
                pred = inner.get("prediction") or inner.get("pred")
        if pred is None:
            pred = ""
        language = item.get("language") or item.get("lang")

        if not image_name or gt is None:
            print(f"Warning: skipping row {index} (need image_name and gt)", file=sys.stderr)
            continue

        rows.append(
            {
                "image_name": str(image_name),
                "gt": str(gt),
                "pred": str(pred) if pred is not None else "",
                "language": str(language) if language is not None else "unknown",
            }
        )

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    return rows


def is_missing_prediction(pred: str) -> bool:
    return not (pred or "").strip()


def default_output_path(
    input_path: str,
    *,
    normalize: bool,
    replace_n: bool,
    remove_br: bool = False,
) -> str:
    stem = input_path.replace(".json", "")
    if normalize and remove_br:
        return f"{stem}.metrics.normalized.remove_br.json"
    if normalize:
        return f"{stem}.metrics.normalized.json"
    if replace_n and remove_br:
        return f"{stem}.metrics.replace_n.remove_br.json"
    if replace_n:
        return f"{stem}.metrics.replace_n.json"
    if remove_br:
        return f"{stem}.metrics.remove_br.json"
    return f"{stem}.metrics.json"


def compute_lang_wise_scores(results: list[dict]) -> dict[str, dict]:
    """Per-language CER/WER over valid, scored rows (excludes loops/missing)."""
    grouped: dict[str, list[dict]] = {}
    for row in results:
        if row["invalid"] or row["loop_or_catastrophic"] or row["metrics"] is None:
            continue
        grouped.setdefault(row["language"], []).append(row)

    return {
        language: {
            "cer": float(mean(r["metrics"]["cer"] for r in group)),
            "wer": float(mean(r["metrics"]["wer"] for r in group)),
            "sample_count": len(group),
        }
        for language, group in sorted(grouped.items())
    }


def compute_metrics(
    rows: list[dict],
    *,
    normalize: bool,
    replace_n: bool,
    remove_br: bool = False,
) -> tuple[dict, list[dict]]:
    results: list[dict] = []
    loop_count = 0
    missing_prediction_count = 0

    for row in rows:
        image_name = row["image_name"]
        gt_raw = row["gt"]
        pred_raw = row["pred"]
        language = row.get("language", "unknown")

        if is_missing_prediction(pred_raw):
            missing_prediction_count += 1
            results.append(
                {
                    "image_name": image_name,
                    "language": language,
                    "gt": gt_raw,
                    "pred": "",
                    "metrics": None,
                    "invalid": True,
                    "loop_or_catastrophic": False,
                    "failure_reasons": ["missing_prediction"],
                }
            )
            continue

        gt, pred = preprocess(
            gt_raw,
            pred_raw,
            normalize=normalize,
            replace_n=replace_n,
            remove_br=remove_br,
        )

        is_bad, failure_reasons = is_loop_or_catastrophic(gt_raw, pred_raw)
        if is_bad:
            loop_count += 1

        all_metrics = calculate_ocr_metrics(gt, pred)
        results.append(
            {
                "image_name": image_name,
                "language": language,
                "gt": gt,
                "pred": pred,
                "metrics": {
                    "cer": all_metrics["cer"],
                    "wer": all_metrics["wer"],
                },
                "invalid": False,
                "loop_or_catastrophic": is_bad,
                "failure_reasons": failure_reasons,
            }
        )

    scored = [r for r in results if not r["invalid"]]
    if not scored:
        raise ValueError("No samples with predictions to score")

    valid = [r for r in scored if not r["loop_or_catastrophic"]]

    summary: dict = {
        "avg_metrics": {
            "cer": float(mean(r["metrics"]["cer"] for r in scored)),
            "wer": float(mean(r["metrics"]["wer"] for r in scored)),
        },
        "word_accuracy": 100.0 * (1.0 - float(mean(r["metrics"]["wer"] for r in scored))),
        "benchmark_sample_count": len(rows),
        "scored_sample_count": len(scored),
        "valid_sample_count": len(valid),
        "missing_prediction_count": missing_prediction_count,
        "loop_failure_count": loop_count,
        "total_sample_count": len(results),
    }

    if valid:
        summary["valid_samples_cer"] = float(mean(r["metrics"]["cer"] for r in valid))
        summary["valid_samples_wer"] = float(mean(r["metrics"]["wer"] for r in valid))
        summary["valid_word_accuracy"] = 100.0 * (1.0 - summary["valid_samples_wer"])
    else:
        summary["valid_samples_cer"] = None
        summary["valid_samples_wer"] = None
        summary["valid_word_accuracy"] = None

    summary["lang_wise_scores"] = compute_lang_wise_scores(results)

    return summary, results


def default_model_name(input_path: str) -> str:
    """Derive a model label from the input path (parent dir, else filename stem)."""
    p = os.path.abspath(input_path)
    stem = os.path.basename(p)
    for suffix in (".gt_pred.json", ".gt_pred", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Predictions often live at <model>/expert_annotations.json -> use the dir.
    if stem in ("expert_annotations", "annotations", "predictions"):
        return os.path.basename(os.path.dirname(p)) or stem
    return stem


# Fixed language column order for the summary TSV.
LANG_ORDER = [
    "Bengali",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Tamil",
    "Telugu",
]


def write_summary_tsv(
    summary: dict,
    *,
    model_name: str,
    tsv_path: str,
    lang_order: list[str] | None = None,
) -> None:
    """Write a per-model TSV with two rows (WER, CER) across lang columns.

    Columns: Model, Metric, <lang_order...>, Overall.
    """
    import csv

    order = lang_order or LANG_ORDER
    header = ["Model", "Metric", *order, "Overall"]
    lang_scores = summary.get("lang_wise_scores") or {}

    def make_row(metric_key: str, overall: float) -> list[str]:
        cells = [
            f"{lang_scores[lang][metric_key]:.4f}" if lang in lang_scores else ""
            for lang in order
        ]
        return [model_name, metric_key.upper(), *cells, f"{overall:.4f}"]

    overall_wer = summary.get("valid_samples_wer", summary["avg_metrics"]["wer"])
    overall_cer = summary.get("valid_samples_cer", summary["avg_metrics"]["cer"])

    os.makedirs(os.path.dirname(os.path.abspath(tsv_path)), exist_ok=True)
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerow(make_row("wer", overall_wer))
        writer.writerow(make_row("cer", overall_cer))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CER/WER from a single gt+pred JSON file (standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON with image_name, gt, and pred per sample",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output metrics JSON (default: <input>.metrics.<mode>.json)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply base + content normalization (merged from benchmark_normalization.py) before scoring",
    )
    parser.add_argument(
        "--replace-n",
        action="store_true",
        help="Only flatten newlines before scoring",
    )
    parser.add_argument(
        "--remove-br",
        action="store_true",
        help="Replace <br>/<br/>/<br /> tags with a space in both gt and pred before scoring",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model label for the TSV (default: derived from input path)",
    )
    parser.add_argument(
        "--tsv",
        default=None,
        help="Path for the per-language summary TSV (default: <output>.summary.tsv)",
    )
    parser.add_argument(
        "--lang-order",
        default=None,
        help="Comma-separated language column order for the summary TSV "
        "(default: top-10 Indic LANG_ORDER)",
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path(
        args.input,
        normalize=args.normalize,
        replace_n=args.replace_n,
        remove_br=args.remove_br,
    )
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output exists: {output_path}. Use --overwrite to replace.")
        sys.exit(1)

    rows = load_gt_pred_rows(args.input)
    summary, results = compute_metrics(
        rows,
        normalize=args.normalize,
        replace_n=args.replace_n,
        remove_br=args.remove_br,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({**summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"benchmark_sample_count:   {summary['benchmark_sample_count']}")
    print(f"scored_sample_count:      {summary['scored_sample_count']}")
    print(f"missing_prediction_count: {summary['missing_prediction_count']}")
    print(f"cer:                      {summary['avg_metrics']['cer']:.4f}")
    print(f"wer:                      {summary['avg_metrics']['wer']:.4f}")
    print(f"word_accuracy:            {summary['word_accuracy']:.2f}%")
    print(f"loop_failure_count:       {summary['loop_failure_count']}")
    print(f"valid_sample_count:       {summary['valid_sample_count']}")
    if summary["valid_samples_cer"] is not None:
        print(f"valid_samples_cer:        {summary['valid_samples_cer']:.4f}")
        print(f"valid_samples_wer:        {summary['valid_samples_wer']:.4f}")
        print(f"valid_word_accuracy:      {summary['valid_word_accuracy']:.2f}%")

    lang_scores = summary.get("lang_wise_scores") or {}
    if lang_scores:
        print("\nper-language (valid samples):")
        print(f"  {'language':<16}{'n':>7}{'cer':>10}{'wer':>10}")
        for language, s in lang_scores.items():
            print(f"  {language:<16}{s['sample_count']:>7}{s['cer']:>10.4f}{s['wer']:>10.4f}")

    print(f"\nSaved to {output_path}")

    model_name = args.model_name or default_model_name(args.input)
    tsv_path = args.tsv or f"{os.path.splitext(output_path)[0]}.summary.tsv"
    lang_order = (
        [x.strip() for x in args.lang_order.split(",") if x.strip()]
        if args.lang_order
        else None
    )
    write_summary_tsv(
        summary,
        model_name=model_name,
        tsv_path=tsv_path,
        lang_order=lang_order,
    )
    print(f"Saved summary TSV to {tsv_path}")


if __name__ == "__main__":
    main()
