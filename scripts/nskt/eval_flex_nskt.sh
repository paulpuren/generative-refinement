#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

# Example:
#   bash scripts/nskt/eval_flex_nskt.sh
#   RE_ID=0 TOTAL_INTERP_STEPS=20 bash scripts/nskt/eval_flex_nskt.sh
#   CHECKPOINT_PATH=./checkpoints/checkpoint_Model_FLEX_..._best.pt bash scripts/nskt/eval_flex_nskt.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

# select 12000, 24000, and 36000: 5, 7, 9 for eval OOD generalization

CONFIG_PATH="${CONFIG_PATH:-config/nskt/flex_posttrain.yaml}"
RE_ID="${RE_ID:-5}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/checkpoint_Model_ft_FLEX3_small_mlp2_Data_nskt_Optim_adam_cosine_lr0.0001_epoch300_patchsize256_stride32_T20_TfixedFalse_best.pt}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/checkpoint_Model_ft_FLEX_small_mlp2_Data_nskt_Optim_adam_cosine_lr0.0001_epoch150_patchsize256_stride32_T20_TfixedTrue_best.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python eval.py
  --config "${CONFIG_PATH}"
  --re_id "${RE_ID}"
  --total_interp_steps "${TOTAL_INTERP_STEPS}"
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
