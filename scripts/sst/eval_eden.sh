#!/usr/bin/env bash
set -euo pipefail

module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/../../EDEN"

CONFIG_PATH="${CONFIG_PATH:-configs/eval_eden_sst.yaml}"
VAE_CKPT="${VAE_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_VAE/012/checkpoints/0012000.pt}"
DIT_CKPT="${DIT_CKPT:-/pscratch/sd/p/puren93/puren/Interp-DM/EDEN/output/experiments-EDEN_DiT/013/checkpoints/0008000.pt}"
SCRATCH_DIR="${SCRATCH_DIR:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
      echo 0
      return
    fi
    awk -F, '{ print NF }' <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi

  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi

  echo "${NUM_PROCESSES}"
}

VISIBLE_GPUS="$(visible_gpu_count)"
if (( VISIBLE_GPUS < 1 )); then
  echo "No visible CUDA GPUs. Check the Slurm allocation or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
if (( NUM_PROCESSES > VISIBLE_GPUS )); then
  echo "Reducing NUM_PROCESSES from ${NUM_PROCESSES} to ${VISIBLE_GPUS} visible GPU(s)." >&2
  NUM_PROCESSES="${VISIBLE_GPUS}"
fi

if (( NUM_PROCESSES == 1 )) && [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "Using a single GPU for eval: CUDA_VISIBLE_DEVICES=${FIRST_VISIBLE_GPU}" >&2
  export CUDA_VISIBLE_DEVICES="${FIRST_VISIBLE_GPU}"
fi

echo "Launching EDEN SST eval with NUM_PROCESSES=${NUM_PROCESSES}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2

CMD=(python -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" eval_sst.py
  --config "${CONFIG_PATH}"
  --vae_checkpoint_path "${VAE_CKPT}"
  --dit_checkpoint_path "${DIT_CKPT}")
if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(--data_path "${SCRATCH_DIR}")
fi

"${CMD[@]}"
