# Results

| File | What it is |
|---|---|
| `verifier-calibration.json` | Summary of the 280-image verifier sweep behind the default verifier pair: per-model literal accuracy, typo/glyph repair rates, reward error, latency and cost, and the best pairs under the current scoring. Full readings: `artifacts/calibration-40.json` (ignored; copied to the bucket). |

## Verifier calibration (2026-09-27)

40 test-split targets (stratified by length) × 7 variants — clean, swapped, missing and doubled
letters, wrong case, one broken glyph, added gibberish — rendered with Pillow, 280 images, each
read blind by 11 hosted VLMs through the environment's own prompt and client.

| Model (provider) | Literal exact | Typo repaired | Broken glyph marked | Broken glyph repaired | Reward MAE | p50 s | $/1k images |
|---|---:|---:|---:|---:|---:|---:|---:|
| **google/gemma-4-31B-it** (deepinfra) | **0.950** | **0.000** | **0.900** | **0.000** | **0.005** | 2.5 | **0.06** |
| zai-org/GLM-4.6V-Flash (novita) | 0.875 | 0.008 | 0.575 | 0.125 | 0.016 | 3.3 | 0.62 |
| **Qwen/Qwen3.6-35B-A3B** (deepinfra) | 0.843 | 0.025 | 0.475 | 0.250 | 0.020 | 1.8 | 0.12 |
| meta-llama/Llama-4-Maverick-17B-FP8 (novita) | 0.832 | 0.008 | 0.450 | 0.400 | 0.020 | 1.7 | 0.68 |
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
| Gemma 4 31B + GLM-4.6V-Flash | 0.0037 | 0.967 | 4 / 240 |
| **Gemma 4 31B + Qwen3.6-35B-A3B (default)** | **0.0068** | **0.971** | **7 / 240** |
| Gemma 4 31B alone | 0.0061 | 0.963 | 1 / 240 |
| best all-Qwen: Qwen3.6-35B-A3B + Qwen3.5-397B | 0.0181 | 0.938 | 15 / 240 |

The default keeps one Qwen reader at ~$0.18 per 1k images; GLM is 4x the cost, slower and served
by one provider. **Caveat:** these are clean typographic renders. Agreement on real diffusion
output (busy scenes, stylised lettering, pseudo-text) is not yet measured.
