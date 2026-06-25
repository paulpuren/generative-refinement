#!/bin/bash
set -euo pipefail

# Example:
#   bash scripts/forecasting/eval_flex_forecast_nskt.sh
#   RE_ID=9 FORECAST_HORIZON=20 bash scripts/forecasting/eval_flex_forecast_nskt.sh
#   CHECKPOINT_PATH=./checkpoints/checkpoint_Model_FLEXForecast_..._best.pt bash scripts/forecasting/eval_flex_forecast_nskt.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/forecasting/eval_flex_forecast_nskt.yaml}"
RE_ID="${RE_ID:-5}"
FORECAST_HORIZON="${FORECAST_HORIZON:-20}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/checkpoints/checkpoint_Model_ft_FLEXForecast_ReNorm40000_small_mlp2_Data_forecasting_nskt_Optim_adam_cosine_lr0.0001_epoch100_patchsize256_stride32_oneStepForecast_best.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python forecasting/eval.py
  --config "${CONFIG_PATH}"
  --re_id "${RE_ID}"
  --forecast_horizon "${FORECAST_HORIZON}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${CMD[@]}"
