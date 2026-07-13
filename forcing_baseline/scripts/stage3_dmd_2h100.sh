#!/usr/bin/env bash
# Stage-3 (2xH100): DMD distillation -> final few-step streaming face-swapper.
# Init = Stage-2 final ckpt; with max_steps=5000 that is fixed at
# checkpoints/chunkwise/stage2_cd_2h100/checkpoint_model_005000/model.pt
# (already baked into configs/dmd_2h100.yaml, no manual edit needed).
# NOTE: this stage holds 3 DiT models (generator + real_score + fake_score);
#       keep gradient_checkpointing=true. On 2xH100 (80GB) FSDP full-shard is
#       tighter than H200 -- if you hit OOM, set text_encoder_cpu_offload:true.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC=${NPROC:-2}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage3_dmd_2h100}
mkdir -p "${LOGDIR}"

export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Reduce allocator fragmentation (Stage-3 holds generator+real_score+fake_score+
# text-encoder); helps the one-off flex-attention mask build fit.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# wandb is configured (key/project) inside configs/dmd_2h100.yaml.
# Set DISABLE_WANDB=1 to turn logging off without editing the config.
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29533 \
  train.py \
  --config_path configs/dmd_2h100.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
