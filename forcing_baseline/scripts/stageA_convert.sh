#!/usr/bin/env bash
# Stage-A: convert bidirectional DreamID-V-Faster weights into a causal init ckpt.
set -euo pipefail

DREAMIDV_CKPT=${DREAMIDV_CKPT:-checkpoints/dreamidv_faster.pth}
OUTPUT_CKPT=${OUTPUT_CKPT:-checkpoints/causal_init.pt}

python -m tools.convert_dreamidv_to_causal \
  --dreamidv_ckpt "${DREAMIDV_CKPT}" \
  --output_ckpt "${OUTPUT_CKPT}"
