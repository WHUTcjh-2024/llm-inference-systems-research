#!/usr/bin/env bash
# One GPU session, minimum decision run.
#
#   ARCH=-arch=sm_90 ./run_all.sh
#   WITH_PREEMPTION=1 ARCH=-arch=sm_90 ./run_all.sh
#
# Runs A first: if clones are unsafe, you can stop before paying for B.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:--arch=sm_90}"
export ARCH

echo "=== Experiment A: exec clone semantics ==="
./exp_a_clone_semantics/run_a.sh

echo "=== Experiment B: uniform decode ==="
WORKLOAD=uniform ./exp_b_host_work/run_b.sh

echo "=== Experiment B: mixed/ragged ==="
WORKLOAD=mixed ./exp_b_host_work/run_b.sh

if [[ "${WITH_PREEMPTION:-0}" == "1" ]]; then
  echo "=== Experiment B: preemption churn ==="
  WORKLOAD=preemption NUM_PROMPTS=128 ./exp_b_host_work/run_b.sh
fi

echo
echo "Done. Fill week1_report_template.md from the CSVs under */results/."
