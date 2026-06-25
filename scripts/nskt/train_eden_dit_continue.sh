#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_dit_nskt_science_continue.yaml}"
PRETRAINED_VAE_PATH="${PRETRAINED_VAE_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/005/checkpoints/0030000.pt}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_DiT/002/checkpoints/0050000.pt}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/}"
CHECK_FILE="${SCRATCH_DIR%/}/2000_2048_2048_seed_2150.h5"

echo "NSKT scratch_dir: ${SCRATCH_DIR}"
if [[ ! -f "${CHECK_FILE}" ]]; then
  echo "NSKT dataset not found: ${CHECK_FILE}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_VAE_PATH}" ]]; then
  echo "VAE checkpoint not found: ${PRETRAINED_VAE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${RESUME_FROM_CKPT}" ]]; then
  echo "DiT checkpoint not found: ${RESUME_FROM_CKPT}" >&2
  exit 1
fi

CMD=(
  python -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  train_dit.py
  --config "${CONFIG_PATH}"
  --pretrained_vae_path "${PRETRAINED_VAE_PATH}"
  --resume_from_ckpt "${RESUME_FROM_CKPT}"
  --scratch_dir "${SCRATCH_DIR}"
)

"${CMD[@]}"
