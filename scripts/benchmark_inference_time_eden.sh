#!/usr/bin/env bash
set -euo pipefail

# Runs GPU-only per-frame timing for EDEN on NSKT, Shanghai, and SST.
# This script uses the EDEN conda environment and keeps EDEN checkpoint/config
# defaults separate from the root-repo UNet/SuperSloMo/FLEX sweep.

module load python
conda activate "${EDEN_CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/eden}"

cd "$(dirname "$0")/.."

NUM_BATCHES="${NUM_BATCHES:-20}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_CSV="${OUTPUT_CSV:-assets/inference_time_per_frame.csv}"

run_eden_bench() {
  python scripts/benchmark_inference_time.py \
    --model EDEN \
    --num_batches "${NUM_BATCHES}" \
    --warmup_batches "${WARMUP_BATCHES}" \
    --num_workers "${NUM_WORKERS}" \
    --output_csv "${OUTPUT_CSV}" \
    "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

# NSKT, Re=36000 by default.
NSKT_RE_ID="${NSKT_RE_ID:-9}"
NSKT_STEPS="${NSKT_STEPS:-20}"
NSKT_SCRATCH="${NSKT_SCRATCH:-/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/}"
NSKT_EDEN_CONFIG="${NSKT_EDEN_CONFIG:-EDEN/configs/eval_eden_nskt.yaml}"
NSKT_EDEN_VAE_CHECKPOINT="${NSKT_EDEN_VAE_CHECKPOINT:-EDEN/output/experiments-EDEN_VAE/014/checkpoints/0020000.pt}"
NSKT_EDEN_DIT_CHECKPOINT="${NSKT_EDEN_DIT_CHECKPOINT:-EDEN/output/experiments-EDEN_DiT/015/checkpoints/0012000.pt}"
require_file "${NSKT_EDEN_CONFIG}"
require_file "${NSKT_EDEN_VAE_CHECKPOINT}"
require_file "${NSKT_EDEN_DIT_CHECKPOINT}"

run_eden_bench --config "${NSKT_EDEN_CONFIG}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --data_path "${NSKT_SCRATCH}" \
  --vae_checkpoint_path "${NSKT_EDEN_VAE_CHECKPOINT}" \
  --dit_checkpoint_path "${NSKT_EDEN_DIT_CHECKPOINT}"

# Shanghai.
SHANGHAI_STEPS="${SHANGHAI_STEPS:-8}"
SHANGHAI_DATA="${SHANGHAI_DATA:-/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5}"
SHANGHAI_EDEN_CONFIG="${SHANGHAI_EDEN_CONFIG:-EDEN/configs/eval_eden_shanghai.yaml}"
SHANGHAI_EDEN_VAE_CHECKPOINT="${SHANGHAI_EDEN_VAE_CHECKPOINT:-EDEN/output/experiments-EDEN_VAE/013/checkpoints/0001800.pt}"
SHANGHAI_EDEN_DIT_CHECKPOINT="${SHANGHAI_EDEN_DIT_CHECKPOINT:-EDEN/output/experiments-EDEN_DiT/014/checkpoints/0002000.pt}"
require_file "${SHANGHAI_EDEN_CONFIG}"
require_file "${SHANGHAI_EDEN_VAE_CHECKPOINT}"
require_file "${SHANGHAI_EDEN_DIT_CHECKPOINT}"

run_eden_bench --config "${SHANGHAI_EDEN_CONFIG}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --data_path "${SHANGHAI_DATA}" \
  --vae_checkpoint_path "${SHANGHAI_EDEN_VAE_CHECKPOINT}" \
  --dit_checkpoint_path "${SHANGHAI_EDEN_DIT_CHECKPOINT}"

# SST / sea temperature.
SST_STEPS="${SST_STEPS:-10}"
SST_DATA="${SST_DATA:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
SST_EDEN_CONFIG="${SST_EDEN_CONFIG:-EDEN/configs/eval_eden_sst.yaml}"
SST_EDEN_VAE_CHECKPOINT="${SST_EDEN_VAE_CHECKPOINT:-EDEN/output/experiments-EDEN_VAE/012/checkpoints/0012000.pt}"
SST_EDEN_DIT_CHECKPOINT="${SST_EDEN_DIT_CHECKPOINT:-EDEN/output/experiments-EDEN_DiT/013/checkpoints/0008000.pt}"
require_file "${SST_EDEN_CONFIG}"
require_file "${SST_EDEN_VAE_CHECKPOINT}"
require_file "${SST_EDEN_DIT_CHECKPOINT}"

run_eden_bench --config "${SST_EDEN_CONFIG}" \
  --data_name sst --total_interp_steps "${SST_STEPS}" \
  --data_path "${SST_DATA}" \
  --vae_checkpoint_path "${SST_EDEN_VAE_CHECKPOINT}" \
  --dit_checkpoint_path "${SST_EDEN_DIT_CHECKPOINT}"

echo "EDEN timing results appended to ${OUTPUT_CSV}"
