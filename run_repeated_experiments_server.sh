#!/usr/bin/env bash
set -euo pipefail

# Server launcher for five-seed repeated experiments.
#
# Usage:
#   bash run_repeated_experiments_server.sh baseline
#   bash run_repeated_experiments_server.sh full
#   bash run_repeated_experiments_server.sh summarize
#
# Optional environment overrides:
#   PYTHON_BIN=/path/to/python
#   SEEDS=42,43,44,45,46
#   OUT_ROOT=/food_data/repeated_runs
#   TRAIN_DIR=/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600
#   VAL_DIR=/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600
#   EPOCHS=30
#   BATCH_SIZE=128
#   NUM_WORKERS=8
#   PK_P=64
#   PK_K=4
#   AMP_DTYPE=bf16
#   NO_CHANNELS_LAST=1
#   COMPILE=1

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: bash $0 {baseline|full|both|summarize}"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42,43,44,45,46}"
OUT_ROOT="${OUT_ROOT:-./repeated_runs}"
TRAIN_DIR="${TRAIN_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600}"
VAL_DIR="${VAL_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
PK_P="${PK_P:-}"
PK_K="${PK_K:-}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NO_CHANNELS_LAST="${NO_CHANNELS_LAST:-}"
NO_FUSED_ADAMW="${NO_FUSED_ADAMW:-}"
COMPILE="${COMPILE:-}"

COMMON_ARGS=(
  --seeds "${SEEDS}"
  --out-root "${OUT_ROOT}"
  --python "${PYTHON_BIN}"
  --train-dir "${TRAIN_DIR}"
  --val-dir "${VAL_DIR}"
)

if [[ -n "${EPOCHS}" ]]; then
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  COMMON_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  COMMON_ARGS+=(--num-workers "${NUM_WORKERS}")
fi
if [[ -n "${PK_P}" ]]; then
  COMMON_ARGS+=(--pk-p "${PK_P}")
fi
if [[ -n "${PK_K}" ]]; then
  COMMON_ARGS+=(--pk-k "${PK_K}")
fi
if [[ -n "${AMP_DTYPE}" ]]; then
  COMMON_ARGS+=(--amp-dtype "${AMP_DTYPE}")
fi
if [[ -n "${NO_CHANNELS_LAST}" ]]; then
  COMMON_ARGS+=(--no-channels-last)
fi
if [[ -n "${NO_FUSED_ADAMW}" ]]; then
  COMMON_ARGS+=(--no-fused-adamw)
fi
if [[ -n "${COMPILE}" ]]; then
  COMMON_ARGS+=(--compile)
fi

case "${MODE}" in
  baseline)
    "${PYTHON_BIN}" run_repeated_experiments.py \
      --model baseline \
      "${COMMON_ARGS[@]}"
    ;;
  full)
    "${PYTHON_BIN}" run_repeated_experiments.py \
      --model full \
      --teacher-root "${OUT_ROOT}" \
      "${COMMON_ARGS[@]}"
    ;;
  both)
    "${PYTHON_BIN}" run_repeated_experiments.py \
      --model both \
      --teacher-root "${OUT_ROOT}" \
      "${COMMON_ARGS[@]}"
    ;;
  summarize)
    "${PYTHON_BIN}" eval/summarize_repeated_runs.py \
      --root "${OUT_ROOT}" \
      --out "${OUT_ROOT}/repeated_runs_summary.csv"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash $0 {baseline|full|both|summarize}"
    exit 1
    ;;
esac
