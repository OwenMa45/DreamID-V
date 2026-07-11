#!/usr/bin/env bash
# Stage-1 (2xH100): causal AR diffusion (teacher forcing).
# Assumes you have already activated your env. No conda/export of paths here.
#
# Robustness vs the earlier NCCL _ALLGATHER_BASE timeout: the trainer now
# barrier()s after the (network-backed) T5/VAE/ckpt loading so no rank races into
# the first FSDP all-gather while another rank is still reading from shared
# storage, and the process-group timeout is bumped via DIST_TIMEOUT_MIN below.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC=${NPROC:-2}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage1_ar_2h100}
mkdir -p "${LOGDIR}"

# Tolerate slow first-reads off /inspire (shared HDD): raise the NCCL/PG timeout.
export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# wandb is configured (key/project) inside configs/ar_diffusion_2h100.yaml.
# Set DISABLE_WANDB=1 to turn logging off without editing the config.
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29531 \
  train.py \
  --config_path configs/ar_diffusion_2h100.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
