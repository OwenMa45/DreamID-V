#!/usr/bin/env bash
# Stage-1 (4xH100): causal AR diffusion (teacher forcing).
# Assumes you have already activated your env. No conda/export here by request.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-4}
LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage1_ar_4h100}
mkdir -p "${LOGDIR}"

# wandb is configured (key/project) inside configs/ar_diffusion_4h100.yaml.
# Set DISABLE_WANDB=1 to turn logging off without editing the config.
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

# Resume control (default: auto-resume from latest ckpt in LOGDIR).
#   RESUME_CKPT=/path/to/checkpoint_model_00XXXX -> resume from a specific ckpt.
#   NO_AUTO_RESUME=1                             -> ignore existing ckpts, start fresh.
RESUME_ARG=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARG="--resume-ckpt ${RESUME_CKPT}"
[ "${NO_AUTO_RESUME:-0}" = "1" ] && RESUME_ARG="${RESUME_ARG} --no-auto-resume"

torchrun --nproc_per_node="${NPROC}" --master_port=29521 \
  train.py \
  --config_path configs/ar_diffusion_4h100.yaml \
  --logdir "${LOGDIR}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG} ${RESUME_ARG}
