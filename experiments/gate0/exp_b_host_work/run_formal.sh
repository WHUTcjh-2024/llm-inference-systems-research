#!/usr/bin/env bash
# Formal Gate 0 Experiment B matrix for one GPU session.
#
#   MODEL=/root/autodl-tmp/models/Qwen3-8B-hf \
#   PYTHON=/root/autodl-tmp/gate0-env/bin/python \
#   ./run_formal.sh
#
# Optional:
#   REPEATS=3 WITH_PREEMPTION=1 WITH_NSYS=1 ./run_formal.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3-8B-hf}"
PYTHON="${PYTHON:-python}"
REPEATS="${REPEATS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
GPU_MEM="${GPU_MEM:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p results
COMMON=(--max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN")

run_config() {
  local workload="$1" async="$2" tag="$3" gpu_mem="$4"
  for i in $(seq 1 "$REPEATS"); do
    echo "=== ${tag} repeat ${i}/${REPEATS} ==="
    MODEL="$MODEL" \
      WORKLOAD="$workload" \
      ASYNC="$async" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      WARMUP_FULL=1 \
      GPU_MEM="$gpu_mem" \
      PYTHON="$PYTHON" \
      ./run_b.sh "${COMMON[@]}"
  done
}

echo "=== formal: uniform decode ==="
run_config uniform on uniform-on "$GPU_MEM"

echo "=== formal: mixed/ragged ==="
run_config mixed on mixed-on "$GPU_MEM"

echo "=== formal: mixed/ragged, async off ==="
run_config mixed off mixed-off "$GPU_MEM"

if [[ "${WITH_PREEMPTION:-0}" == "1" ]]; then
  echo "=== formal: preemption churn ==="
  run_config preemption on preemption-on 0.60
fi

echo "=== clean control: instrumentation vs engine multiprocessing ==="
for async in on off; do
  for mp in on off; do
    LOG="results/clean_mixed_async-${async}_mp-${mp}_$(date +%Y%m%d_%H%M%S).log"
    "$PYTHON" bench_clean.py \
      --model "$MODEL" \
      --workload mixed \
      --chunked-prefill \
      --num-prompts "$NUM_PROMPTS" \
      --output-len "$OUTPUT_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM" \
      --async-scheduling "$async" \
      --engine-multiprocessing "$mp" \
      --warmup-full 2>&1 | tee "$LOG"
  done
done

if [[ "${WITH_NSYS:-0}" == "1" ]]; then
  echo "=== Nsight: mixed async on/off ==="
  for async in on off; do
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUT="results/nsys_mixed_async-${async}_${STAMP}"
    MODEL="$MODEL" \
      WORKLOAD=mixed \
      ASYNC="$async" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      WARMUP_FULL=1 \
      GPU_MEM="$GPU_MEM" \
      PYTHON="$PYTHON" \
      nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --sample=none --cpuctxsw=none \
        --capture-range=nvtx --nvtx-capture=gate0:measure \
        --capture-range-end=stop \
        --output="$OUT" \
        ./run_b.sh "${COMMON[@]}" --nvtx
    nsys export --type sqlite --force-overwrite=true \
      --output="${OUT}.sqlite" "${OUT}.nsys-rep"
    "$PYTHON" analyze_nsys.py "${OUT}.sqlite"
  done
fi

echo "=== aggregate ==="
"$PYTHON" aggregate_b.py --dir results --out results/formal_aggregate.csv
