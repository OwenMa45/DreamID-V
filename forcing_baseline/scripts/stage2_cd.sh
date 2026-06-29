#!/usr/bin/env bash
# Stage-2: causal Consistency Distillation (CD).
set -euo pipefail

NPROC=${NPROC:-8}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage2_cd}

torchrun --nproc_per_node="${NPROC}" --master_port=29502 \
  train.py \
  --config_path configs/causal_cd.yaml \
  --logdir "${LOGDIR}" \
  --disable-wandb
