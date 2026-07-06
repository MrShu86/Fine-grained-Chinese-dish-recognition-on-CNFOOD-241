#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DIR="${TRAIN_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600}"
VAL_DIR="${VAL_DIR:-/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600}"
OUT_DIR="${OUT_DIR:-./exp_swin_b_300}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
RESUME_CKPT="${RESUME_CKPT:-}"

ARGS=(
  --train-dir "${TRAIN_DIR}"
  --val-dir "${VAL_DIR}"
  --out-dir "${OUT_DIR}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --amp-dtype "${AMP_DTYPE}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  ARGS+=(--resume-ckpt "${RESUME_CKPT}")
fi
if [[ "${NO_PRETRAINED:-}" == "1" ]]; then
  ARGS+=(--no-pretrained)
fi
if [[ "${NO_MIXUP_CUTMIX:-}" == "1" ]]; then
  ARGS+=(--no-mixup-cutmix)
fi
if [[ "${NO_RANDOM_ERASING:-}" == "1" ]]; then
  ARGS+=(--no-random-erasing)
fi
if [[ "${NO_DATAPARALLEL:-}" == "1" ]]; then
  ARGS+=(--no-dataparallel)
fi
if [[ "${CHANNELS_LAST:-}" == "1" ]]; then
  ARGS+=(--channels-last)
fi
if [[ "${COMPILE:-}" == "1" ]]; then
  ARGS+=(--compile)
fi

mkdir -p "${OUT_DIR}"
echo "[CMD] ${PYTHON_BIN} main.py ${ARGS[*]}" | tee "${OUT_DIR}/train.log"
"${PYTHON_BIN}" main.py "${ARGS[@]}" 2>&1 | tee -a "${OUT_DIR}/train.log"
