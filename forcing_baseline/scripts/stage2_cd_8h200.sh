#!/usr/bin/env bash
# Stage-2 (8-GPU node, H200/H100): causal Consistency Distillation (CD).
# Init = Stage-1 final ckpt (trained on 2xH100, ckpts are hardware-independent):
# checkpoints/chunkwise/stage1_ar_2h100/checkpoint_model_005000/model.pt
# (already baked into configs/causal_cd_8h200.yaml, no manual edit needed).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage2_cd_8h200}
mkdir -p "${LOGDIR}"

# Tolerate slow first-reads off shared storage; reduce allocator fragmentation
# (Stage-2 holds student+EMA+teacher+text-encoder).
export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# wandb is disabled in configs/causal_cd_8h200.yaml (offline server).
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

# Resume control (default: auto-resume student/EMA from latest ckpt in LOGDIR;
# the frozen teacher always keeps its Stage-1 init).
#   RESUME_CKPT=/path/to/checkpoint_model_00XXXX -> resume from a specific ckpt.
#   NO_AUTO_RESUME=1                             -> ignore existing ckpts, start fresh.
RESUME_ARG=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARG="--resume-ckpt ${RESUME_CKPT}"
[ "${NO_AUTO_RESUME:-0}" = "1" ] && RESUME_ARG="${RESUME_ARG} --no-auto-resume"

torchrun --nproc_per_node="${NPROC}" --master_port=29542 \
  train.py \
  --config_path configs/causal_cd_8h200.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG} ${RESUME_ARG}
