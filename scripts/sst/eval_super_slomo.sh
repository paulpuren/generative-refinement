#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "Using a single GPU for eval: CUDA_VISIBLE_DEVICES=${FIRST_VISIBLE_GPU}" >&2
  export CUDA_VISIBLE_DEVICES="${FIRST_VISIBLE_GPU}"
fi

CONFIG_PATH="${CONFIG_PATH:-config/sst/eval_super_slomo.yaml}"

python eval_super_slomo.py --config "${CONFIG_PATH}"
