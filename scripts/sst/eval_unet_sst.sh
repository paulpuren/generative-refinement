#!/usr/bin/env bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

# Example:
#   bash scripts/sst/eval_unet_sst.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/sst/eval_unet_sst.sh
#   TOTAL_INTERP_STEPS_EVAL=20 BATCH_SIZE=8 bash scripts/sst/eval_unet_sst.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai

# Run from repository root (script lives in scripts/sst/)
cd "$(dirname "$0")/../.."

USE_MULTI_GPU="${USE_MULTI_GPU:-true}"
if [[ "${USE_MULTI_GPU}" == "true" || "${USE_MULTI_GPU}" == "1" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    if (( ${#VISIBLE_GPU_ARRAY[@]} > SLURM_GPUS_ON_NODE )); then
      CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${VISIBLE_GPU_ARRAY[*]:0:${SLURM_GPUS_ON_NODE}}")"
      echo "Truncated CUDA_VISIBLE_DEVICES to Slurm allocation: ${CUDA_VISIBLE_DEVICES}" >&2
      export CUDA_VISIBLE_DEVICES
    fi
  fi
elif [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "Using a single GPU for eval: CUDA_VISIBLE_DEVICES=${FIRST_VISIBLE_GPU}" >&2
  export CUDA_VISIBLE_DEVICES="${FIRST_VISIBLE_GPU}"
fi
export USE_MULTI_GPU

# ---- checkpoint/run-name-matching args (must match training config) ----
# Matches:
# checkpoints/checkpoint_Model_UNet_Data_sea_temp_Optim_adam_lr0.0001_epoch30_stride64_T10_TfixedFalse.pt
MODEL="${MODEL:-UNet}"
DATA_NAME="${DATA_NAME:-sea_temp}"
OPTIMIZER="${OPTIMIZER:-adam}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-30}"
STRIDE="${STRIDE:-64}"
TOTAL_INTERP_STEPS_TRAIN="${TOTAL_INTERP_STEPS_TRAIN:-10}"
IS_T_FIXED="${IS_T_FIXED:-False}"

# ---- evaluation args ----
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-128}"
TIME_STEPS="${TIME_STEPS:-10}"
PREDICTION_TYPE="${PREDICTION_TYPE:-v}"
TOTAL_INTERP_STEPS_EVAL="${TOTAL_INTERP_STEPS_EVAL:-10}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
SST_EVAL_MAX_SAMPLES="${SST_EVAL_MAX_SAMPLES:-2048}"

SST_EVAL_MAX_SAMPLES="${SST_EVAL_MAX_SAMPLES}" \
python eval_sst.py \
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
