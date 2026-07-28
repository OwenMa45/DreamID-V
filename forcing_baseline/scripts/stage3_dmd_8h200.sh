#!/usr/bin/env bash
# Stage-3 (8-GPU node, H200/H100): DMD distillation -> final few-step streaming
# face-swapper.
# Init = Stage-2 final ckpt (trained on 2xH100):
# checkpoints/chunkwise/stage2_cd_2h100/checkpoint_model_005000/model.pt
# (already baked into configs/dmd_8h200.yaml, no manual edit needed).
# 1000 steps, ckpt every 100; the step-500 ckpt is the inference deliverable.
# NOTE: this stage holds 3 DiT models (generator + real_score + fake_score) plus
#       per-GPU Self-Forcing rollout activations that do NOT shard; 8-way FSDP
#       makes the static shards small (~9 GB/GPU), leaving ample headroom.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage3_dmd_8h200}
mkdir -p "${LOGDIR}"

export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Reduce allocator fragmentation; helpful given the rollout memory spikes.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# wandb is disabled in configs/dmd_8h200.yaml (offline server).
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29543 \
  train.py \
  --config_path configs/dmd_8h200.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
