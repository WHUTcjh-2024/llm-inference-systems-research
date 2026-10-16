#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="$SCRIPT_DIR/e1.lock.json"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-tmp/dash-router-e1}"
PERSIST_ROOT="$(realpath -m -- "$PERSIST_ROOT")"
MODE="${1:-all}"

if [[ "$MODE" != "all" && "$MODE" != "prepare" && "$MODE" != "run" ]]; then
  printf 'Usage: bash run_e1.sh [prepare|run|all]\n' >&2
  exit 2
fi
if [[ "$PERSIST_ROOT" != /root/autodl-tmp/* ]]; then
  printf 'PERSIST_ROOT must be under /root/autodl-tmp: %s\n' "$PERSIST_ROOT" >&2
  exit 2
fi
if [[ ! -d /root/autodl-tmp ]] || [[ ! -w /root/autodl-tmp ]]; then
  printf '/root/autodl-tmp is absent or not writable; check the AutoDL data disk.\n' >&2
  exit 2
fi
if ! mountpoint -q /root/autodl-tmp; then
  printf '/root/autodl-tmp is not a mountpoint; refusing to write a checkpoint to the system disk.\n' >&2
  exit 2
fi

mkdir -p "$PERSIST_ROOT"/{source,checkpoints,cache,results}
mkdir -p "$PERSIST_ROOT/cache/tmp" "$PERSIST_ROOT/cache/triton"
export HF_HOME="$PERSIST_ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_CACHE_DIR=1
export TMPDIR="$PERSIST_ROOT/cache/tmp"
export TRITON_CACHE_DIR="$PERSIST_ROOT/cache/triton"

read_lock() {
  python - "$LOCK_FILE" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as file:
    print(json.load(file)[sys.argv[2]])
PY
}

UPSTREAM_REPO="$(read_lock upstream_repo)"
UPSTREAM_COMMIT="$(read_lock upstream_commit)"
MODEL_REPO="$(read_lock model_repo)"
MODEL_REVISION="$(read_lock model_revision)"
MODEL_WEIGHT_BYTES="$(read_lock model_weight_bytes)"
SOURCE_DIR="$PERSIST_ROOT/source/dash-attention"
MODEL_DIR="$PERSIST_ROOT/checkpoints/minicpm-4-8b-dashattention-$MODEL_REVISION"
VENV_DIR="$PERSIST_ROOT/.venv"

if [[ "$MODE" == "prepare" || "$MODE" == "all" ]]; then
  PREP_LOG="$PERSIST_ROOT/results/prepare-$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$PREP_LOG") 2>&1
  printf 'Preparation log: %s\n' "$PREP_LOG"
  python - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit("Python 3.10-3.12 is required for pinned PyTorch 2.6.0")
PY
  FREE_BYTES="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
  EXISTING_BYTES=0
  if [[ -f "$MODEL_DIR/pytorch_model.bin" ]]; then
    EXISTING_BYTES="$(stat -c '%s' "$MODEL_DIR/pytorch_model.bin")"
  fi
  REMAINING_BYTES=$((MODEL_WEIGHT_BYTES - EXISTING_BYTES))
  if (( REMAINING_BYTES < 0 )); then
    printf 'Existing checkpoint is larger than the pinned object.\n' >&2
    exit 2
  fi
  NEEDED_BYTES=$((REMAINING_BYTES + 12 * 1024 * 1024 * 1024))
  if (( FREE_BYTES < NEEDED_BYTES )); then
    printf 'Insufficient data-disk space: free=%s bytes, needed=%s bytes (remaining checkpoint + 12 GiB environment reserve).\n' "$FREE_BYTES" "$NEEDED_BYTES" >&2
    exit 2
  fi
  if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git clone --filter=blob:none "$UPSTREAM_REPO" "$SOURCE_DIR"
  fi
  if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]]; then
    printf 'Tracked upstream files are modified; inspect the source checkout before prepare.\n' >&2
    exit 2
  fi
  git -C "$SOURCE_DIR" fetch --filter=blob:none origin "$UPSTREAM_COMMIT"
  git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_COMMIT"
  [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]]
  python -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -e "$SOURCE_DIR" pytest huggingface_hub
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/download_checkpoint.py" \
    --repo "$MODEL_REPO" --revision "$MODEL_REVISION" --dest "$MODEL_DIR" \
    --lock "$LOCK_FILE"
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/verify_source.py" \
    --source "$SOURCE_DIR" --lock "$LOCK_FILE" --checkpoint "$MODEL_DIR"
fi

if [[ "$MODE" == "run" || "$MODE" == "all" ]]; then
  [[ -x "$VENV_DIR/bin/python" ]] || { printf 'Run prepare first.\n' >&2; exit 2; }
  command -v nvidia-smi >/dev/null
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "$GPU_NAME" != *H20* ]]; then
    printf 'Expected one H20, found: %s\n' "$GPU_NAME" >&2
    exit 2
  fi
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
  if [[ "$GPU_COUNT" -ne 1 ]]; then
    printf 'Expected one visible GPU, found: %s\n' "$GPU_COUNT" >&2
    exit 2
  fi
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/verify_source.py" \
    --source "$SOURCE_DIR" --lock "$LOCK_FILE" --checkpoint "$MODEL_DIR" \
    --rehash-weight
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="$PERSIST_ROOT/results/$RUN_ID"
  mkdir -p "$RUN_DIR"
  cp "$LOCK_FILE" "$RUN_DIR/e1.lock.json"
  nvidia-smi -q > "$RUN_DIR/nvidia-smi.txt"
  "$VENV_DIR/bin/python" -m pip freeze > "$RUN_DIR/pip-freeze.txt"
  "$VENV_DIR/bin/python" -m pytest -q \
    "$SOURCE_DIR/test/test_smoke.py" "$SOURCE_DIR/test/test_llama_dash_attn.py" \
    2>&1 | tee "$RUN_DIR/official-tests.log"
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/cache_smoke.py" \
    2>&1 | tee "$RUN_DIR/cache-smoke.log"
  "$VENV_DIR/bin/python" "$SOURCE_DIR/example/run_niah.py" \
    --model-path "$MODEL_DIR" --target-tokens 4096 --max-new-tokens 2 \
    --device cuda --attn-implementation dash_attn \
    --scaling-factor 0.34 --sparsity-exclude-prefix-tokens 0 \
    2>&1 | tee "$RUN_DIR/official-niah-smoke.log"
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/e1_profile.py" \
    --source "$SOURCE_DIR" --model "$MODEL_DIR" --lock "$LOCK_FILE" \
    --output "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/validate_results.py" \
    --results "$RUN_DIR/results.json" --lock "$LOCK_FILE" \
    2>&1 | tee "$RUN_DIR/validation.log"
  printf 'E1 results: %s\n' "$RUN_DIR"
fi
