#!/usr/bin/env bash
# Stage-1 (2xH200): causal AR diffusion (teacher forcing).
# Assumes you have already activated your env. No conda/export here by request.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC=${NPROC:-2}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage1_ar_2h200}
mkdir -p "${LOGDIR}"

# wandb is configured (key/project) inside configs/ar_diffusion_2h200.yaml.
# Set DISABLE_WANDB=1 to turn logging off without editing the config.
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29511 \
  train.py \
  --config_path configs/ar_diffusion_2h200.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
