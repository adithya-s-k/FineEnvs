#!/usr/bin/env bash
# Unpack Mathlib's prebuilt cache in batches and truncate the files that importing Mathlib never
# reads (generated C and .ilean editor indexes), keeping their .hash files so Lake still sees every
# module as built. Lean + full Mathlib otherwise needs ~11 GB, more than a 10 GB sandbox disk.
set -euo pipefail
cd /opt/lean-env
trim() {
  find .lake/packages -path '*/.lake/build/*' \( -name '*.c' -o -name '*.ilean' \) -size +0 -exec truncate -s 0 {} +
  rm -rf "${HOME:-/root}/.cache/mathlib"
}
for dir in .lake/packages/mathlib/Mathlib/*/; do
  name=$(basename "$dir")
  echo "== Mathlib/$name"
  lake exe cache get ".lake/packages/mathlib/Mathlib/$name" >/dev/null
  trim
  df -h / | tail -1
done
lake exe cache get
trim
