#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

CONFIG_PATH="${CONFIG_PATH:-configs/eval_eden_shanghai.yaml}"

VAE_CKPT="${VAE_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/013/checkpoints/0001800.pt}"

DIT_CKPT="${DIT_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_DiT/014/checkpoints/0002000.pt}"

SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" eval_scientific.py
  --config "${CONFIG_PATH}"
  --vae_checkpoint_path "${VAE_CKPT}"
  --dit_checkpoint_path "${DIT_CKPT}")
if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(--data_path "${SCRATCH_DIR}")
fi

"${CMD[@]}"
