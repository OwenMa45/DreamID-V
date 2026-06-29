#!/usr/bin/env bash
# Stage-1: causal AR diffusion (teacher forcing).
set -euo pipefail

NPROC=${NPROC:-8}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage1_ar}

torchrun --nproc_per_node="${NPROC}" --master_port=29501 \
  train.py \
  --config_path configs/ar_diffusion.yaml \
  --logdir "${LOGDIR}" \
  --disable-wandb
