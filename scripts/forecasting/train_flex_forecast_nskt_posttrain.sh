#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262
#
# Example:
#   bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh
#   EPOCHS=500 bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh
#   CHECKPOINT_PATH=checkpoints/checkpoint_Model_FLEXForecast_..._last.pt bash scripts/forecasting/train_flex_forecast_nskt_posttrain.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/forecasting/flex_forecast_nskt_posttrain.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/checkpoints/checkpoint_Model_ft_FLEXForecast_ReNorm40000_small_mlp2_Data_forecasting_nskt_Optim_adam_cosine_lr0.0001_epoch300_patchsize256_stride32_oneStepForecast_last.pt}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"



CMD=(
  python forecasting/train.py
  --config "${CONFIG_PATH}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${EPOCHS}" ]]; then
  CMD+=(--epochs "${EPOCHS}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
