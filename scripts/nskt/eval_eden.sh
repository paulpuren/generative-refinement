#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

CONFIG_PATH="${CONFIG_PATH:-configs/eval_eden_nskt.yaml}"
VAE_CKPT="${VAE_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/014/checkpoints/0020000.pt}"
DIT_CKPT="${DIT_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_DiT/015/checkpoints/0012000.pt}"

RE_ID="${RE_ID:-9}"
TOTAL_INTERP_STEPS="${TOTAL_INTERP_STEPS:-24}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/}"
CHECK_FILE="${SCRATCH_DIR%/}/600_2048_2048_seed_3407.h5"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

echo "NSKT scratch_dir: ${SCRATCH_DIR}"
echo "NSKT total interpolation steps: ${TOTAL_INTERP_STEPS}"
if [[ ! -f "${CHECK_FILE}" ]]; then
  echo "NSKT dataset not found: ${CHECK_FILE}" >&2
  exit 1
fi

CMD=(python -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision no
  --dynamo_backend no
  eval_scientific.py
  --config "${CONFIG_PATH}"
  --vae_checkpoint_path "${VAE_CKPT}"
  --dit_checkpoint_path "${DIT_CKPT}"
  --re_id "${RE_ID}"
  --total_interp_steps "${TOTAL_INTERP_STEPS}")
CMD+=(--scratch_dir "${SCRATCH_DIR}")

"${CMD[@]}"
