#!/usr/bin/env bash
# Stage-3a (8-GPU node, H200/H100): DMD distillation initialised from a *chosen*
# Stage-2 CD checkpoint (instead of the step-5000 one baked into
# configs/dmd_8h200.yaml). Trainer / config / hyper-params are identical to
# stage3_dmd_8h200.sh; only the generator init and the logdir differ.
#
# Interface (CKPT_STEP semantics mirror infer_latest_*.sh):
#   bash scripts/stage3a_dmd_8h200.sh                  # init from LATEST stage2 ckpt
#   CKPT_STEP=3000 bash scripts/stage3a_dmd_8h200.sh   # init from checkpoint_model_003000
#   STAGE2_DIR=checkpoints/chunkwise/stage2_cd_8h200 CKPT_STEP=2000 \
#     bash scripts/stage3a_dmd_8h200.sh                # pick another stage2 run
#   GENERATOR_CKPT=/abs/path/model.pt bash scripts/stage3a_dmd_8h200.sh
#                                                      # bypass STAGE2_DIR/CKPT_STEP
#
# Each init gets its own logdir: checkpoints/chunkwise/stage3a_dmd_8h200_from<step>
# (override via LOGDIR=...), so runs from different Stage-2 ckpts never clobber
# each other. Inference: scripts/infer_latest_1h200.sh stage id "3a".
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}

# --- resolve the Stage-2 checkpoint to distil from ---------------------------
STAGE2_DIR=${STAGE2_DIR:-checkpoints/chunkwise/stage2_cd_2h100}
if [ -z "${GENERATOR_CKPT:-}" ]; then
  if [ -n "${CKPT_STEP:-}" ]; then
    # pin a specific step (e.g. CKPT_STEP=3000 -> checkpoint_model_003000)
    pick="${STAGE2_DIR}/checkpoint_model_$(printf '%06d' "${CKPT_STEP}")"
  else
    pick="$(ls -d "${STAGE2_DIR}"/checkpoint_model_*/ 2>/dev/null | sort | tail -n 1 || true)"
    pick="${pick%/}"
  fi
  if [ -z "${pick}" ] || [ ! -f "${pick}/model.pt" ]; then
    echo "[stage3a] no usable stage2 checkpoint (${pick:-none}) under ${STAGE2_DIR}" >&2
    exit 1
  fi
  GENERATOR_CKPT="${pick}/model.pt"
fi
init_step="$(basename "$(dirname "${GENERATOR_CKPT}")" | sed 's/checkpoint_model_//')"
echo "[stage3a] DMD init from stage2 ckpt: ${GENERATOR_CKPT} (step ${init_step})"

LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage3a_dmd_8h200_from${init_step}}
mkdir -p "${LOGDIR}"

export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Reduce allocator fragmentation; helpful given the rollout memory spikes.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# wandb is disabled in configs/dmd_8h200.yaml (offline server).
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

torchrun --nproc_per_node="${NPROC}" --master_port=29544 \
  train.py \
  --config_path configs/dmd_8h200.yaml \
  --logdir "${LOGDIR}" \
  --generator-ckpt "${GENERATOR_CKPT}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG}
