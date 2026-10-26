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
PYTHON="${PYTHON:-python}"

make clean all clone_external ARCH="$ARCH"
mkdir -p results

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="results/exp_a_${STAMP}.csv"
META="results/exp_a_${STAMP}.meta.txt"

if ./clone_semantics \
  --iters "$ITERS" \
  --spin "$SPIN" \
  --repeats "$REPEATS" \
  --mode both \
  >"$CSV" 2>"$META"; then
  "$PYTHON" analyze_a.py --input "$CSV"
else
  status=$?
  if [[ "$status" != "3" ]] ||
     ! grep -q 'second_instantiation=not_supported' "$META"; then
    cat "$META" >&2
    exit "$status"
  fi
  echo "Graph allocation/free nodes forbid a second exec instance (CUDA API)."
  cat "$META"
fi

EXTERNAL="results/exp_a_external_${STAMP}.csv"
EXTERNAL_META="results/exp_a_external_${STAMP}.meta.txt"
./clone_external "$REPEATS" "$SPIN" "$ITERS" >"$EXTERNAL" 2>"$EXTERNAL_META"
"$PYTHON" analyze_a.py --input "$EXTERNAL"

echo "graph-memory metadata: $META"
echo "external control:     $EXTERNAL"
echo "external metadata:    $EXTERNAL_META"
