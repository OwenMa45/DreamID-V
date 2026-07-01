#!/usr/bin/env bash
# Stage-0 (single H100): self-distill training data with the bidirectional
# DreamID-V-Faster teacher. Produces the LMDB consumed by Stage-1/2/3.
#
# Same data/paths as the 2xH200 variant -- Stage-0 is a single-process teacher
# inference (no torchrun), so this just pins it to one H100 card.
#
# Data: LivingSwap-style groups, each <base>{.mp4,_mask.mp4,_ref.jpg}:
#   <base>.mp4      driving/source video (face to be replaced)
#   <base>_mask.mp4 face-region mask (PROVIDED -> no DWPose needed)
#   <base>_ref.jpg  reference identity face
# The corpus.jsonl manifest is auto-built from INPUT_DIR if missing.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V}
CKPT_DIR=${CKPT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/wan2.1-t2v-1.3b}
MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}
DREAMIDV_CKPT=${DREAMIDV_CKPT:-${MODELS_DIR}/dreamidv_faster.pth}

# Paired data directory (1000 groups) and the manifest derived from it.
INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
MANIFEST=${MANIFEST:-${PROJECT_ROOT}/corpus.jsonl}
OUTPUT_LMDB=${OUTPUT_LMDB:-${PROJECT_ROOT}/dataset/swap_latents}
mkdir -p "${OUTPUT_LMDB}"

# Single H100: pin to one card (device_id 0 inside the tool maps to it).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Auto-build the manifest from the paired input dir if not already present.
if [ ! -f "${MANIFEST}" ]; then
  echo "[stage0] manifest not found -> building from ${INPUT_DIR}"
  python -m tools.build_manifest --input_dir "${INPUT_DIR}" --output "${MANIFEST}"
fi

# Masks are provided, so DWPose is normally unused. As a fallback (groups missing
# a mask), link the onnx models into the hard-coded pose/models/ location.
POSE_DIR="${DREAMIDV_ROOT}/pose/models"
mkdir -p "${POSE_DIR}"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
  if [ ! -s "${POSE_DIR}/${f}" ] && [ -s "${MODELS_DIR}/${f}" ]; then
    ln -sf "${MODELS_DIR}/${f}" "${POSE_DIR}/${f}"
    echo "[stage0] linked ${f} -> ${POSE_DIR}/${f}"
  fi
done

# size 832*480 sets the target area; 640x640 inputs -> ~624x624 -> 78x78 latent.
python -m tools.syncid_generate_data \
  --dreamidv_root "${DREAMIDV_ROOT}" \
  --ckpt_dir "${CKPT_DIR}" \
  --dreamidv_ckpt "${DREAMIDV_CKPT}" \
  --manifest "${MANIFEST}" \
  --output_lmdb "${OUTPUT_LMDB}" \
  --size 832*480 \
  --frame_num 81 \
  --sampling_steps 12 \
  --sample_shift 5.0 \
  --guide_scale_img 4.0
