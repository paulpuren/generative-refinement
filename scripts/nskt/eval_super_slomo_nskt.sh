#!/bin/bash
set -euo pipefail

# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/eval_super_slomo.yaml}"

python eval_super_slomo.py --config "${CONFIG_PATH}"
