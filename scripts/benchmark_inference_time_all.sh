#!/usr/bin/env bash
set -euo pipefail

# Runs GPU-only per-frame timing for UNet, Super SloMo, and FLEX on
# NSKT, Shanghai, and SST. EDEN uses a different environment; run
# scripts/benchmark_inference_time_eden.sh for EDEN timings.
module load python
conda activate "${CONDA_ENV_PATH:-/pscratch/sd/p/puren93/conda_env/genai}"

cd "$(dirname "$0")/.."

NUM_BATCHES="${NUM_BATCHES:-20}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_CSV="${OUTPUT_CSV:-assets/inference_time_per_frame.csv}"

run_bench() {
  python scripts/benchmark_inference_time.py \
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
NSKT_UNET_CHECKPOINT="${NSKT_UNET_CHECKPOINT:-checkpoints/checkpoint_Model_UNet_Data_nskt_Optim_adam_lr0.0001_epoch200_stride32_T20_TfixedFalse.pt}"
NSKT_SLOMO_CHECKPOINT="${NSKT_SLOMO_CHECKPOINT:-checkpoints/checkpoint_Model_SuperSloMo_Data_nskt_Optim_adam_lr6e-05_epoch200_stride32_T20_TfixedTrue.pt}"
NSKT_FLEX_CHECKPOINT="${NSKT_FLEX_CHECKPOINT:-checkpoints/checkpoint_Model_ft_FLEX3_small_mlp2_Data_nskt_Optim_adam_cosine_lr0.0001_epoch300_patchsize256_stride32_T20_TfixedFalse_best.pt}"
require_file "${NSKT_UNET_CHECKPOINT}"
require_file "${NSKT_SLOMO_CHECKPOINT}"
require_file "${NSKT_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${NSKT_UNET_CONFIG:-config/nskt/unet.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_UNET_CHECKPOINT}" \
  --epochs 200 --patch_size 256

run_bench --model SuperSloMo --config "${NSKT_SLOMO_CONFIG:-config/nskt/eval_super_slomo.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${NSKT_FLEX_CONFIG:-config/nskt/flex_posttrain.yaml}" \
  --data_name nskt --re_id "${NSKT_RE_ID}" --total_interp_steps "${NSKT_STEPS}" \
  --scratch_dir "${NSKT_SCRATCH}" --checkpoint_path "${NSKT_FLEX_CHECKPOINT}" \
  --epochs 300 --is_T_fixed False

# Shanghai.
SHANGHAI_STEPS="${SHANGHAI_STEPS:-8}"
SHANGHAI_DATA="${SHANGHAI_DATA:-/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5}"
SHANGHAI_UNET_CHECKPOINT="${SHANGHAI_UNET_CHECKPOINT:-checkpoints/checkpoint_Model_UNet_Data_shanghai_Optim_adam_lr0.0001_epoch400_stride32_T8_TfixedFalse.pt}"
SHANGHAI_SLOMO_CHECKPOINT="${SHANGHAI_SLOMO_CHECKPOINT:-checkpoints/checkpoint_Model_SuperSloMo_Data_shanghai_Optim_adam_lr0.0001_epoch400_patchsize128_stride32_T8_TfixedTrue_best.pt}"
SHANGHAI_FLEX_CHECKPOINT="${SHANGHAI_FLEX_CHECKPOINT:-checkpoints/checkpoint_Model_FLEX3_small_mlp2_Data_shanghai_Optim_adam_cosine_lr0.0001_epoch300_stride32_T8_TfixedFalse_best.pt}"
require_file "${SHANGHAI_UNET_CHECKPOINT}"
require_file "${SHANGHAI_SLOMO_CHECKPOINT}"
require_file "${SHANGHAI_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${SHANGHAI_UNET_CONFIG:-config/shanghai/unet.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_UNET_CHECKPOINT}" \
  --optimizer adam --learning_rate 1e-4 --epochs 400 \
  --stride 32 --total_interp_steps_train 8 --is_T_fixed False \
  --batch_size 12 --patch_size 128 --time_steps 10

run_bench --model SuperSloMo --config "${SHANGHAI_SLOMO_CONFIG:-config/shanghai/eval_super_slomo.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${SHANGHAI_FLEX_CONFIG:-config/shanghai/flex.yaml}" \
  --data_name shanghai --total_interp_steps "${SHANGHAI_STEPS}" \
  --scratch_dir "${SHANGHAI_DATA}" --checkpoint_path "${SHANGHAI_FLEX_CHECKPOINT}"

# SST / sea temperature.
SST_STEPS="${SST_STEPS:-10}"
SST_DATA="${SST_DATA:-/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/}"
SST_UNET_CHECKPOINT="${SST_UNET_CHECKPOINT:-checkpoints/checkpoint_Model_UNet_Data_sea_temp_Optim_adam_lr0.0001_epoch30_stride64_T10_TfixedFalse.pt}"
SST_SLOMO_CHECKPOINT="${SST_SLOMO_CHECKPOINT:-checkpoints/checkpoint_Model_SuperSloMo_Data_sea_temp_Optim_adam_lr0.0001_epoch200_patchsize128_stride32_T10_TfixedTrue_best.pt}"
SST_FLEX_CHECKPOINT="${SST_FLEX_CHECKPOINT:-checkpoints/checkpoint_Model_ft_FLEX3_small_mlp2_Data_sea_temp_Optim_adam_cosine_lr0.0001_epoch100_patchsize128_stride32_T10_TfixedFalse_best.pt}"
require_file "${SST_UNET_CHECKPOINT}"
require_file "${SST_SLOMO_CHECKPOINT}"
require_file "${SST_FLEX_CHECKPOINT}"

run_bench --model UNet --config "${SST_UNET_CONFIG:-config/sst/unet.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_UNET_CHECKPOINT}"

run_bench --model SuperSloMo --config "${SST_SLOMO_CONFIG:-config/sst/eval_super_slomo.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_SLOMO_CHECKPOINT}"

run_bench --model FLEX --config "${SST_FLEX_CONFIG:-config/sst/flex_posttrain.yaml}" \
  --data_name sea_temp --total_interp_steps "${SST_STEPS}" \
  --scratch_dir "${SST_DATA}" --checkpoint_path "${SST_FLEX_CHECKPOINT}"

echo "Timing results appended to ${OUTPUT_CSV}"
