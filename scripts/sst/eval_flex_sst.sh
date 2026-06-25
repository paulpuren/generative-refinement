#!/usr/bin/env bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

# Example:
#   bash scripts/sst/eval_flex_sst.sh
#   TOTAL_INTERP_STEPS_EVAL=10 BATCH_SIZE=12 bash scripts/sst/eval_flex_sst.sh
#   CHECKPOINT_PATH=./checkpoints/checkpoint_Model_ft_FLEX_..._best.pt bash scripts/sst/eval_flex_sst.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
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

MODEL="${MODEL:-FLEX}"
DATA_NAME="${DATA_NAME:-sea_temp}"
FLEX_MODEL_SIZE="${FLEX_MODEL_SIZE:-small}"
FLEX_MLP_RATIO="${FLEX_MLP_RATIO:-2}"
USE_SCALAR_FILM="${USE_SCALAR_FILM:-true}"
USE_SPATIAL_COND="${USE_SPATIAL_COND:-true}"
SPATIAL_COND_MODE="${SPATIAL_COND_MODE:-gated}"
OPTIMIZER="${OPTIMIZER:-adam}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-100}"
STRIDE="${STRIDE:-32}"
TOTAL_INTERP_STEPS_TRAIN="${TOTAL_INTERP_STEPS_TRAIN:-10}"
IS_T_FIXED="${IS_T_FIXED:-False}"

BATCH_SIZE="${BATCH_SIZE:-12}"
PATCH_SIZE="${PATCH_SIZE:-128}"
TIME_STEPS="${TIME_STEPS:-3}"
PREDICTION_TYPE="${PREDICTION_TYPE:-v}"
TOTAL_INTERP_STEPS_EVAL="${TOTAL_INTERP_STEPS_EVAL:-10}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/checkpoints/checkpoint_Model_ft_FLEX3_small_mlp2_Data_sea_temp_Optim_adam_cosine_lr0.0001_epoch100_patchsize128_stride32_T10_TfixedFalse_best.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python eval_sst.py
  --model "${MODEL}"
  --data_name "${DATA_NAME}"
  --flex_model_size "${FLEX_MODEL_SIZE}"
  --flex_mlp_ratio "${FLEX_MLP_RATIO}"
  --use_scalar_film "${USE_SCALAR_FILM}"
  --use_spatial_cond "${USE_SPATIAL_COND}"
  --spatial_cond_mode "${SPATIAL_COND_MODE}"
  --optimizer "${OPTIMIZER}"
  --learning_rate "${LEARNING_RATE}"
  --epochs "${EPOCHS}"
  --stride "${STRIDE}"
  --total_interp_steps_train "${TOTAL_INTERP_STEPS_TRAIN}"
  --is_T_fixed "${IS_T_FIXED}"
  --batch_size "${BATCH_SIZE}"
  --patch_size "${PATCH_SIZE}"
  --time_steps "${TIME_STEPS}"
  --prediction_type "${PREDICTION_TYPE}"
  --total_interp_steps "${TOTAL_INTERP_STEPS_EVAL}"
  --scratch_dir "${SCRATCH_DIR}"
  --checkpoint_path "${CHECKPOINT_PATH}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
