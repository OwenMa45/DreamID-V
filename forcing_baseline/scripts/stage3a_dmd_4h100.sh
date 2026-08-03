#!/usr/bin/env bash
# Stage-3a (4xH100): DMD distillation initialised from a *chosen* Stage-2 CD
# checkpoint (instead of the step-5000 one baked into configs/dmd_4h100.yaml).
# Trainer / config / hyper-params are identical to stage3_dmd_4h100.sh (incl.
# text_encoder_cpu_offload for the tighter 80 GB cards); only the generator
# init and the logdir differ. 4-GPU twin of stage3a_dmd_8h200.sh.
#
# Ckpt cadence: saves every 50 steps and stops at step 500 (the DMD deliverable
# zone; over-training DMD degrades diversity/quality). This overrides the
# 100/1000 baked into the shared yaml -- tune via LOG_ITERS / MAX_STEPS.
#
# Interface (CKPT_STEP semantics mirror infer_latest_*.sh):
#   bash scripts/stage3a_dmd_4h100.sh                  # init from LATEST stage2 ckpt
#   CKPT_STEP=3000 bash scripts/stage3a_dmd_4h100.sh   # init from checkpoint_model_003000
#   STAGE2_DIR=checkpoints/chunkwise/stage2_cd_8h200 CKPT_STEP=2000 \
#     bash scripts/stage3a_dmd_4h100.sh                # pick another stage2 run
#   GENERATOR_CKPT=/abs/path/model.pt bash scripts/stage3a_dmd_4h100.sh
#                                                      # bypass STAGE2_DIR/CKPT_STEP
#
# Each init gets its own logdir: checkpoints/chunkwise/stage3a_dmd_4h100_from<step>
# (override via LOGDIR=...), so runs from different Stage-2 ckpts never clobber
# each other. Inference: infer_latest_*.sh stage id "3a" (auto-discovers the
# most recent stage3a_dmd_*_from<step> run on the shared storage).
#
# Crash-resume: every checkpoint also writes <LOGDIR>/resume_state.pt (step +
# raw generator + critic + EMA). Re-running this script with the SAME CKPT_STEP
# lands in the same LOGDIR and continues from that state automatically.
#   NO_AUTO_RESUME=1     -> ignore the saved state, restart from the stage2 ckpt
#   RESUME_CKPT=/path/to/logdir_or_resume_state.pt -> resume a specific state
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-4}

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

LOGDIR=${LOGDIR:-checkpoints/chunkwise/stage3a_dmd_4h100_from${init_step}}
mkdir -p "${LOGDIR}"

# Ckpt every 50 steps, stop at 500 (overrides log_iters/max_steps in the yaml).
MAX_STEPS=${MAX_STEPS:-500}
LOG_ITERS=${LOG_ITERS:-50}

export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Reduce allocator fragmentation; critical here given the rollout memory spikes.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# wandb is disabled in configs/dmd_4h100.yaml (offline server).
WANDB_FLAG=""
[ "${DISABLE_WANDB:-0}" = "1" ] && WANDB_FLAG="--disable-wandb"

# Resume control (default: auto-resume from LOGDIR/resume_state.pt if present).
RESUME_ARG=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARG="--resume-ckpt ${RESUME_CKPT}"
[ "${NO_AUTO_RESUME:-0}" = "1" ] && RESUME_ARG="${RESUME_ARG} --no-auto-resume"

torchrun --nproc_per_node="${NPROC}" --master_port=29524 \
  train.py \
  --config_path configs/dmd_4h100.yaml \
  --logdir "${LOGDIR}" \
  --generator-ckpt "${GENERATOR_CKPT}" \
  --max-steps "${MAX_STEPS}" \
  --log-iters "${LOG_ITERS}" \
  --wandb-save-dir "${LOGDIR}" \
  ${WANDB_FLAG} ${RESUME_ARG}
