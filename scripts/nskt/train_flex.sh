#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/flex_posttrain.yaml}"

python train.py --config "${CONFIG_PATH}"
