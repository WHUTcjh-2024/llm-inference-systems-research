#!/usr/bin/env bash
# Experiment B: one workload, one config.
#
# Usage:
#   WORKLOAD=uniform ./run_b.sh
#   WORKLOAD=mixed ./run_b.sh
#   WORKLOAD=preemption ./run_b.sh
#   ASYNC=off WORKLOAD=mixed ./run_b.sh
#
# Optional env:
#   MODEL=Qwen/Qwen3-8B
#   GPU_MEM=0.85
#   NUM_PROMPTS=64
#   OUTPUT_LEN=128
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
WORKLOAD="${WORKLOAD:-mixed}"
ASYNC="${ASYNC:-default}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
WARMUP_FULL="${WARMUP_FULL:-1}"
PYTHON="${PYTHON:-python}"
if [[ "$PYTHON" == */* ]]; then
  export PATH="$(dirname "$PYTHON"):$PATH"
fi

if [[ "$WORKLOAD" == "preemption" && -z "${GPU_MEM:-}" ]]; then
  GPU_MEM=0.60
fi

OUT_DIR="${OUT_DIR:-results}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/exp_b_${WORKLOAD}_async-${ASYNC}_${STAMP}.csv"
LOG="${OUT_DIR}/exp_b_${WORKLOAD}_async-${ASYNC}_${STAMP}.log"

ARGS=(
  --model "$MODEL"
  --workload "$WORKLOAD"
  --async-scheduling "$ASYNC"
  --gpu-memory-utilization "${GPU_MEM:-0.85}"
  --num-prompts "$NUM_PROMPTS"
  --output-len "$OUTPUT_LEN"
  --out "$OUT"
)

if [[ "$WARMUP_FULL" == "1" ]]; then
  ARGS+=(--warmup-full)
fi

if [[ "$WORKLOAD" == "mixed" ]]; then
  ARGS+=(--chunked-prefill)
fi

"$PYTHON" bench_b.py "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
"$PYTHON" analyze_b.py --input "$OUT" --out "${OUT%.csv}.summary.csv" 2>&1 | tee -a "$LOG"
