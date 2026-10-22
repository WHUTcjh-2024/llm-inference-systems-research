#!/usr/bin/env bash
# Experiment A: exec-clone semantics.
#
# Usage:
#   ARCH=-arch=sm_90 ./run_a.sh          # H20
#   ARCH=-arch=sm_80 ./run_a.sh          # A100 / A800
#   ARCH=-arch=sm_89 ./run_a.sh          # RTX 4090
#
# Optional:
#   REPEATS=200 SPIN=50000000 ITERS=1048576 ./run_a.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:--arch=sm_90}"
REPEATS="${REPEATS:-200}"
SPIN="${SPIN:-50000000}"
ITERS="${ITERS:-1048576}"

make clean all ARCH="$ARCH"
mkdir -p results

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="results/exp_a_${STAMP}.csv"
META="results/exp_a_${STAMP}.meta.txt"

./clone_semantics \
  --iters "$ITERS" \
  --spin "$SPIN" \
  --repeats "$REPEATS" \
  --mode both \
  2>"$META" | tee "$CSV"

echo "csv:  $CSV"
echo "meta: $META"
python analyze_a.py --input "$CSV"
