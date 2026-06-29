#!/usr/bin/env bash
# Stage-0: self-distill training data with the bidirectional DreamID-V-Faster teacher.
#   manifest.jsonl: one JSON per line -> {"ref_video": "...", "ref_image": "...", "mask": "(optional)", "prompt": "(optional)"}
set -euo pipefail

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/mnt/nas/share/home/lzk/mrq/swap/DreamID-V}
CKPT_DIR=${CKPT_DIR:-checkpoints/Wan2.1}
DREAMIDV_CKPT=${DREAMIDV_CKPT:-checkpoints/dreamidv_faster.pth}
MANIFEST=${MANIFEST:-corpus.jsonl}
OUTPUT_LMDB=${OUTPUT_LMDB:-dataset/swap_latents}

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
