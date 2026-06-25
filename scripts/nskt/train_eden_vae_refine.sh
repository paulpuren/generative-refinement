#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/train_vae_nskt_refine.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH}" NUM_PROCESSES="${NUM_PROCESSES}" bash "${SCRIPT_DIR}/train_eden_vae.sh"
