#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_dit_sst.yaml}"
PRETRAINED_VAE_PATH="${PRETRAINED_VAE_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/012/checkpoints/0012000.pt}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" train_dit.py --config "${CONFIG_PATH}" --pretrained_vae_path "${PRETRAINED_VAE_PATH}")
if [[ -n "${RESUME_FROM_CKPT}" ]]; then
  CMD+=(--resume_from_ckpt "${RESUME_FROM_CKPT}")
fi
if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(--data_path "${SCRATCH_DIR}")
fi

"${CMD[@]}"
