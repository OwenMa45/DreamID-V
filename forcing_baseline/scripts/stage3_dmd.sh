#!/usr/bin/env bash
# Stage-3: DMD distillation -> final few-step streaming face-swapper.
set -euo pipefail

NPROC=${NPROC:-8}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage3_dmd}

torchrun --nproc_per_node="${NPROC}" --master_port=29503 \
  train.py \
  --config_path configs/dmd.yaml \
  --logdir "${LOGDIR}" \
  --disable-wandb
