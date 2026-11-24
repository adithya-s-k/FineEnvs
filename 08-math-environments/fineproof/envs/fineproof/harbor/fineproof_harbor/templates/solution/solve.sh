#!/usr/bin/env bash
# Oracle: submit the reference proof (a DeepSeek-Math-V2 proof from lm-provers/FineProofs-SFT that Gemini 3 Pro
# graded 7 of 7, or the benchmark's own reference solution) so the judge's grade of a known-good proof can be checked.
set -euo pipefail
mkdir -p /workspace
cp /solution/proof.md /workspace/proof.md
echo "placed the reference proof ($(wc -c < /workspace/proof.md) bytes)"
