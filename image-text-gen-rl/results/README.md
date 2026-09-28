# Results

| File | What it is |
|---|---|
| `ocr-baseline-comparison.json` | Conventional OCR (PaddleOCR 2.9.1, EasyOCR 1.7.2) + edit distance vs the environment on the same 280 renders (`train/compare_ocr_baseline.py`). |
| `verifier-calibration.json` | Summary of the 280-image verifier sweep, re-scored under `blind-transcription-v2` behind the default verifier pair: per-model literal accuracy, typo/glyph repair rates, reward error, latency and cost, and the best pairs under the current scoring. Full readings: `artifacts/calibration-40.json` (ignored; copied to the bucket). |

## Verifier calibration (2026-09-27)

40 test-split targets (stratified by length) × 7 variants — clean, swapped, missing and doubled
letters, wrong case, one broken glyph, added gibberish — rendered with Pillow, 280 images, each
read blind by 11 hosted VLMs through the environment's own prompt and client.

| Model (provider) | Literal exact | Typo repaired | Broken glyph marked | Broken glyph repaired | Reward MAE | p50 s | $/1k images |
|---|---:|---:|---:|---:|---:|---:|---:|
| **google/gemma-4-31B-it** (deepinfra) | **0.950** | **0.000** | **0.900** | **0.000** | **0.005** | 2.5 | **0.06** |
| zai-org/GLM-4.6V-Flash (novita) | 0.875 | 0.008 | 0.575 | 0.125 | 0.017 | 3.3 | 0.62 |
| **Qwen/Qwen3.6-35B-A3B** (deepinfra) | 0.843 | 0.025 | 0.475 | 0.250 | 0.020 | 1.8 | 0.12 |
| meta-llama/Llama-4-Maverick-17B-FP8 (novita) | 0.832 | 0.008 | 0.450 | 0.400 | 0.021 | 1.7 | 0.68 |
| Qwen/Qwen3.5-397B-A17B (deepinfra) | 0.829 | 0.025 | 0.275 | 0.425 | 0.024 | 1.7 | 0.53 |
| Qwen/Qwen3-VL-235B-A22B-Instruct (deepinfra) | 0.839 | 0.017 | 0.350 | 0.425 | 0.025 | 2.6 | 0.33 |
| Qwen/Qwen3.5-27B (deepinfra) | 0.771 | 0.050 | 0.050 | 0.550 | 0.033 | 1.9 | 0.31 |
| Qwen/Qwen3.6-27B (deepinfra) | 0.771 | 0.058 | 0.050 | 0.475 | 0.033 | 2.1 | 0.55 |
| Qwen/Qwen3-VL-30B-A3B-Instruct (deepinfra) | 0.746 | 0.042 | 0.025 | 0.575 | 0.034 | 1.6 | 0.25 |
| deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (deepinfra) | 0.761 | 0.133 | 0.400 | 0.450 | 0.038 | 2.6 | 0.40 |
| Qwen/Qwen3.8-27B (deepinfra) | 0.714 | 0.175 | 0.050 | 0.500 | 0.039 | 1.9 | 0.35 |

Kimi-K2.5 and MiniMax-M3 were dropped after a 28-image pilot (empty readings on 36% of images;
8.8 s median latency). No model ever marked a clean render as malformed.

Pairs, combined with the environment's rule (text from the best reading, broken glyphs from
the worst, extra text from the most lenient):

| Verifier pair | Reward MAE | Clean ranked above its defect | Defects given full reward |
|---|---:|---:|---:|
| Gemma 4 31B + GLM-4.6V-Flash | 0.0041 | 0.971 | 4 / 240 |
| **Gemma 4 31B + Qwen3.6-35B-A3B (default)** | **0.0065** | **0.975** | **6 / 240** |
| Gemma 4 31B alone | 0.0046 | 0.967 | 1 / 240 |
| best all-Qwen: Qwen3.6-35B-A3B + Qwen3.5-397B | 0.0182 | 0.938 | 15 / 240 |

Mean reward of the default pair by variant: clean 1.000 · missing letter 0.924 · doubled letter
0.929 · wrong case 0.900 · swapped letters 0.853 · extra gibberish 0.782 · broken glyph 0.758.
Under v1, gibberish scored ~0.95 because the extra-text penalty scaled with target length; v2
charges a fixed 0.05 per unrequested character beyond 2 (details in `JUDGE.md`).

The default keeps one Qwen reader at ~$0.18 per 1k images; GLM is 4x the cost, slower and served
by one provider. **Caveat:** these are clean typographic renders. Agreement on real diffusion
output (busy scenes, stylised lettering, pseudo-text) is not yet measured.

## Conventional OCR as the reward (2026-09-28)

Same 280 renders, read by PaddleOCR 2.9.1 and EasyOCR 1.7.2 on HF Jobs (CPU) and scored with the
simple reward, 1 − edit distance / length (case- and space-insensitive):

| Reader + reward | Clean ranked above its defect | Defects given full reward | Error vs literal truth |
|---|---:|---:|---:|
| PaddleOCR + edit distance | 0.808 | 28 / 240 | 0.101 |
| EasyOCR + edit distance | 0.708 | 25 / 240 | 0.144 |
| PaddleOCR + environment scoring | 0.917 | 1 / 240 | 0.048 |
| **Gemma 4 31B + Qwen3.6-35B (environment)** | **0.975** | **6 / 240** | **0.007** |

OCR rarely auto-corrects (a typo came back as the correct word 0 / 120 times). It fails by misreading
correct renders (PaddleOCR read 77.5% of clean renders exactly, EasyOCR 65%),
by having no symbol for a broken glyph (which then costs about one edit, like a typo), and, with the usual
reward, by ignoring case. Clean Pillow renders only; diffusion outputs not yet measured.
