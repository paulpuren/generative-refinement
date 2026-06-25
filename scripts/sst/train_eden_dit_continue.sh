#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_dit_sst.yaml}"
PRETRAINED_VAE_PATH="${PRETRAINED_VAE_PATH:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/012/checkpoints/0012000.pt}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_DiT/012/checkpoints/0012000.pt}"
DATA_PATH="${DATA_PATH:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
EPOCHS="${EPOCHS:-100}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_VAE_PATH}" ]]; then
  echo "VAE checkpoint not found: ${PRETRAINED_VAE_PATH}" >&2
  exit 1
fi

if [[ -z "${RESUME_FROM_CKPT}" ]]; then
  echo "Set RESUME_FROM_CKPT to the EDEN DiT checkpoint you want to continue from." >&2
  echo "Example: RESUME_FROM_CKPT=/path/to/checkpoints/0012000.pt $0" >&2
  exit 1
fi

if [[ ! -f "${RESUME_FROM_CKPT}" ]]; then
  echo "DiT checkpoint not found: ${RESUME_FROM_CKPT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "SST data directory not found: ${DATA_PATH}" >&2
  exit 1
fi

CMD=(
  python -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  train_dit.py
  --config "${CONFIG_PATH}"
  --pretrained_vae_path "${PRETRAINED_VAE_PATH}"
  --resume_from_ckpt "${RESUME_FROM_CKPT}"
  --data_path "${DATA_PATH}"
  --epochs "${EPOCHS}"
)

"${CMD[@]}"
