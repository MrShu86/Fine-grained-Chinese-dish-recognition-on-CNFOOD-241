#!/usr/bin/env bash
set -euo pipefail

# Server launcher for R_KDAT ablation experiments.
#
# Usage:
#   bash run_ablation_experiments_server.sh
#   VARIANTS=core SEEDS=42 bash run_ablation_experiments_server.sh
#   VARIANTS=kd_mbc,kd_er,mbc_er,core SEEDS=1,25,50,100,42 bash run_ablation_experiments_server.sh
#
# Optional environment overrides:
#   PYTHON_BIN=/path/to/python
#   VARIANTS=kd_mbc,kd_er,mbc_er,core
#   SEEDS=1,25,50,100,42
#   OUT_ROOT=./ablation_runs
#   TRAIN_DIR=/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600
#   VAL_DIR=/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600
#   TEACHER_CKPT=/food_data/RegNetY-32GF/exp_regnety32/best.pt
#   TEACHER_ROOT=./repeated_runs
#   EPOCHS=40
#   BATCH_SIZE=128
#   NUM_WORKERS=12
#   PK_P=32
#   PK_K=4
#   AMP_DTYPE=fp16
#   NO_CHANNELS_LAST=1
#   NO_FUSED_ADAMW=1
#   COMPILE=1

PYTHON_BIN="${PYTHON_BIN:-python}"
VARIANTS="${VARIANTS:-kd_mbc,kd_er,mbc_er,core}"
SEEDS="${SEEDS:-1,25,50,100,42}"
OUT_ROOT="${OUT_ROOT:-./ablation_runs}"
TRAIN_DIR="${TRAIN_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600}"
VAL_DIR="${VAL_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
TEACHER_ROOT="${TEACHER_ROOT:-}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
PK_P="${PK_P:-}"
PK_K="${PK_K:-}"
AMP_DTYPE="${AMP_DTYPE:-}"
NO_CHANNELS_LAST="${NO_CHANNELS_LAST:-}"
NO_FUSED_ADAMW="${NO_FUSED_ADAMW:-}"
COMPILE="${COMPILE:-}"

ARGS=(
  --variants "${VARIANTS}"
  --seeds "${SEEDS}"
  --out-root "${OUT_ROOT}"
  --python "${PYTHON_BIN}"
  --train-dir "${TRAIN_DIR}"
  --val-dir "${VAL_DIR}"
)

if [[ -n "${TEACHER_CKPT}" ]]; then
  ARGS+=(--teacher-ckpt "${TEACHER_CKPT}")
fi
if [[ -n "${TEACHER_ROOT}" ]]; then
  ARGS+=(--teacher-root "${TEACHER_ROOT}")
fi
if [[ -n "${EPOCHS}" ]]; then
  ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  ARGS+=(--num-workers "${NUM_WORKERS}")
fi
if [[ -n "${PK_P}" ]]; then
  ARGS+=(--pk-p "${PK_P}")
fi
if [[ -n "${PK_K}" ]]; then
  ARGS+=(--pk-k "${PK_K}")
fi
if [[ -n "${AMP_DTYPE}" ]]; then
  ARGS+=(--amp-dtype "${AMP_DTYPE}")
fi
if [[ -n "${NO_CHANNELS_LAST}" ]]; then
  ARGS+=(--no-channels-last)
fi
if [[ -n "${NO_FUSED_ADAMW}" ]]; then
  ARGS+=(--no-fused-adamw)
fi
if [[ -n "${COMPILE}" ]]; then
  ARGS+=(--compile)
fi

"${PYTHON_BIN}" run_ablation_experiments.py "${ARGS[@]}"
