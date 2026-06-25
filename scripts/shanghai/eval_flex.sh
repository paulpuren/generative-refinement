#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

# Example:
#   bash scripts/shanghai/eval_flex.sh
#   TOTAL_INTERP_STEPS=8 BATCH_SIZE=12 bash scripts/shanghai/eval_flex.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/shanghai/flex.yaml}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python eval_shanghai_update.py \
  --config "${CONFIG_PATH}" \
  --total_interp_steps "${TOTAL_INTERP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  ${EXTRA_ARGS}
