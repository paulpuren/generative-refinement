#!/usr/bin/env bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

# Example:
#   bash scripts/shanghai/eval_unet_shanghai.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/shanghai/eval_unet_shanghai.sh
#   TOTAL_INTERP_STEPS_EVAL=20 BATCH_SIZE=8 bash scripts/shanghai/eval_unet_shanghai.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai

# Run from repository root (script lives in scripts/shanghai/)
cd "$(dirname "$0")/../.."

# Support both filenames; repo currently has eval_shanghai_update.py.
if [[ -f "eval_shanghai_updated.py" ]]; then
  EVAL_SCRIPT="eval_shanghai_updated.py"
else
  EVAL_SCRIPT="eval_shanghai_update.py"
fi

# ---- checkpoint/run-name-matching args (must match training config) ----
# Default checkpoint target:
# checkpoints/checkpoint_Model_UNet_Data_shanghai_Optim_lion_lr1e-05_epoch100_stride64_T20_TfixedFalse.pt
MODEL="${MODEL:-UNet}"
DATA_NAME="${DATA_NAME:-shanghai}"
OPTIMIZER="${OPTIMIZER:-adam}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-400}"
STRIDE="${STRIDE:-32}"
TOTAL_INTERP_STEPS_TRAIN="${TOTAL_INTERP_STEPS_TRAIN:-8}"
IS_T_FIXED="${IS_T_FIXED:-False}"

# ---- evaluation args ----
BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-128}"
TIME_STEPS="${TIME_STEPS:-10}"
PREDICTION_TYPE="${PREDICTION_TYPE:-v}"
TOTAL_INTERP_STEPS_EVAL="${TOTAL_INTERP_STEPS_EVAL:-8}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5}"

python "${EVAL_SCRIPT}" \
  --model "${MODEL}" \
  --data_name "${DATA_NAME}" \
  --optimizer "${OPTIMIZER}" \
  --learning_rate "${LEARNING_RATE}" \
  --epochs "${EPOCHS}" \
  --stride "${STRIDE}" \
  --total_interp_steps_train "${TOTAL_INTERP_STEPS_TRAIN}" \
  --is_T_fixed "${IS_T_FIXED}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --time_steps "${TIME_STEPS}" \
  --prediction_type "${PREDICTION_TYPE}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS_EVAL}" \
  --scratch_dir "${SCRATCH_DIR}"
