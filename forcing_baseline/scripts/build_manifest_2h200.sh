#!/usr/bin/env bash
# Build corpus.jsonl from the paired LivingSwap input directory.
# Each group <base>{.mp4,_mask.mp4,_ref.jpg} -> one JSON line with ref_video /
# ref_image / mask / prompt. Run this (or let stage0 auto-run it) before Stage-0.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/corpus.jsonl}
# Set MAX_SAMPLES=2 (etc.) to smoke-test on a few groups first.
MAX_SAMPLES=${MAX_SAMPLES:--1}

python -m tools.build_manifest \
  --input_dir "${INPUT_DIR}" \
  --output "${OUTPUT}" \
  --max_samples "${MAX_SAMPLES}"
