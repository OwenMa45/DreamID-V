#!/usr/bin/env bash
# Streaming face-swap inference with the distilled causal generator (single GPU).
# Generator = Stage-3 final ckpt; with max_steps=5000 that is fixed at
# checkpoints/chunkwise/stage3_dmd_2h200/checkpoint_model_005000/model.pt
# (already baked into configs/inference_2h200.yaml, no manual edit needed).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V}
MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Example inputs (official test case shipped with DreamID-V).
REF_VIDEO=${REF_VIDEO:-${DREAMIDV_ROOT}/assets/test_case/ref_video/a_girl.mp4}
REF_IMAGE=${REF_IMAGE:-${DREAMIDV_ROOT}/assets/test_case/ref_image/an_1.jpg}
SAVE_FILE=${SAVE_FILE:-${PROJECT_ROOT}/outputs/swapped_2h200.mp4}
# Reuse the teacher's fixed "change face" embedding to skip loading T5.
CONTEXT_PATH=${CONTEXT_PATH:-${DREAMIDV_ROOT}/dreamidv_wan_faster/context.pth}

# DWPose onnx for the driving-video mask (same as Stage-0).
POSE_DIR="${DREAMIDV_ROOT}/pose/models"
mkdir -p "${POSE_DIR}"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
  if [ ! -s "${POSE_DIR}/${f}" ] && [ -s "${MODELS_DIR}/${f}" ]; then
    ln -sf "${MODELS_DIR}/${f}" "${POSE_DIR}/${f}"
  fi
done

python inference.py \
  --config_path configs/inference_2h200.yaml \
  --dreamidv_root "${DREAMIDV_ROOT}" \
  --ref_video "${REF_VIDEO}" \
  --ref_image "${REF_IMAGE}" \
  --save_file "${SAVE_FILE}" \
  --context_path "${CONTEXT_PATH}" \
  --size 832*480 \
  --frame_num 81 \
  --fps 24
