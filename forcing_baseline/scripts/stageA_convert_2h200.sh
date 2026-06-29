#!/usr/bin/env bash
# Stage-A (CPU/single GPU): convert bidirectional DreamID-V-Faster weights into
# a causal init checkpoint consumed by Stage-1 (configs/ar_diffusion_2h200.yaml).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}
DREAMIDV_CKPT=${DREAMIDV_CKPT:-${MODELS_DIR}/dreamidv_faster.pth}
OUTPUT_CKPT=${OUTPUT_CKPT:-${PROJECT_ROOT}/checkpoints/causal_init_2h200.pt}

mkdir -p "$(dirname "${OUTPUT_CKPT}")"

python -m tools.convert_dreamidv_to_causal \
  --dreamidv_ckpt "${DREAMIDV_CKPT}" \
  --output_ckpt "${OUTPUT_CKPT}"
