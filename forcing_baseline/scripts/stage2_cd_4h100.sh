#!/usr/bin/env bash
# Stage-2 (4xH100): causal Consistency Distillation (CD).
# Init = Stage-1 final ckpt; with max_steps=5000 that is fixed at
# checkpoints/chunkwise/stage1_ar_4h100/checkpoint_model_005000/model.pt
# (already baked into configs/causal_cd_4h100.yaml, no manual edit needed).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-4}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage2_cd_4h100}
mkdir -p "${LOGDIR}"

# wandb is configured (key/project) inside configs/causal_cd_4h100.yaml.
# Set DISABLE_WANDB=1 to turn logging off without editing the config.
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29522 \
  train.py \
  --config_path configs/causal_cd_4h100.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
