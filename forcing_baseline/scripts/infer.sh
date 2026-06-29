#!/usr/bin/env bash
# Streaming face-swap inference with the distilled causal generator.
set -euo pipefail

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/mnt/nas/share/home/lzk/mrq/swap/DreamID-V}
CKPT_DIR=${CKPT_DIR:-checkpoints/Wan2.1}
REF_VIDEO=${REF_VIDEO:-assets/driving.mp4}
REF_IMAGE=${REF_IMAGE:-assets/ref.jpg}
SAVE_FILE=${SAVE_FILE:-outputs/swapped.mp4}

python inference.py \
  --config_path configs/inference.yaml \
  --dreamidv_root "${DREAMIDV_ROOT}" \
  --ckpt_dir "${CKPT_DIR}" \
  --ref_video "${REF_VIDEO}" \
  --ref_image "${REF_IMAGE}" \
  --save_file "${SAVE_FILE}"
