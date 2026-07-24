#!/usr/bin/env bash
# Streaming face-swap inference using the LATEST checkpoint of each stage, over
# the FIRST few (default 3) data groups from the Stage-0 input directory.
#
# Data groups share a base name (same triple Stage-0 consumed):
#   <base>.mp4       source/driving video
#   <base>_mask.mp4  face-region mask  (passed via --mask, so no DWPose)
#   <base>_ref.jpg   reference identity face
#
# For every requested stage we auto-pick checkpoints/chunkwise/<stage>/
# checkpoint_model_XXXXXX with the highest step and run inference.py on each
# group. Stages without any checkpoint are skipped. Outputs -> forcing_baseline/outputs/.
#
# Usage:
#   bash scripts/infer_latest_2h100.sh              # all stages that have ckpts
#   STAGES=1 bash scripts/infer_latest_2h100.sh     # only stage 1
#   NUM_GROUPS=5 bash scripts/infer_latest_2h100.sh # first 5 groups
#   STAGES=3 CKPT_STEP=500 bash scripts/infer_latest_2h100.sh
#       -> pin a specific step instead of the latest (e.g. the stage-3 DMD
#          deliverable is the step-500 ckpt of the 1000-step run)
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V}
MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}
INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
CONTEXT_PATH=${CONTEXT_PATH:-${DREAMIDV_ROOT}/dreamidv_wan_faster/context.pth}
CONFIG=${CONFIG:-configs/inference_2h100.yaml}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs}

# Match the Stage-0 latent geometry so the conditioning distribution lines up.
SIZE=${SIZE:-832*480}
FRAME_NUM=${FRAME_NUM:-81}
FPS=${FPS:-24}
NUM_GROUPS=${NUM_GROUPS:-3}

# Inference is single-GPU.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "${OUT_DIR}"

# Masks are provided so DWPose is not used; still link the onnx as a harmless fallback.
POSE_DIR="${DREAMIDV_ROOT}/pose/models"
mkdir -p "${POSE_DIR}"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
  if [ ! -s "${POSE_DIR}/${f}" ] && [ -s "${MODELS_DIR}/${f}" ]; then
    ln -sf "${MODELS_DIR}/${f}" "${POSE_DIR}/${f}"
  fi
done

# Stage -> (logdir, short name). Add other hardware dirs here if needed.
declare -A LOGDIR=(
  [1]="checkpoints/chunkwise/stage1_ar_2h100"
  [2]="checkpoints/chunkwise/stage2_cd_2h100"
  [3]="checkpoints/chunkwise/stage3_dmd_2h100"
)
declare -A SNAME=( [1]="stage1_ar" [2]="stage2_cd" [3]="stage3_dmd" )

STAGES=${STAGES:-"1 2 3"}

# First N groups (sorted by *_ref.jpg -> same pairing/order as build_manifest).
mapfile -t REFS < <(ls "${INPUT_DIR}"/*_ref.jpg 2>/dev/null | sort | head -n "${NUM_GROUPS}")
if [ "${#REFS[@]}" -eq 0 ]; then
  echo "[infer] no *_ref.jpg found under ${INPUT_DIR}"
  exit 1
fi
echo "[infer] ${#REFS[@]} group(s), config=${CONFIG}, size=${SIZE}, frames=${FRAME_NUM}"

for st in ${STAGES}; do
  dir="${LOGDIR[$st]:-}"
  if [ -z "${dir}" ]; then
    echo "[infer][stage${st}] unknown stage id -- skipping."
    continue
  fi
  if [ -n "${CKPT_STEP:-}" ]; then
    # pin a specific step (e.g. CKPT_STEP=500 -> checkpoint_model_000500)
    latest="${dir}/checkpoint_model_$(printf '%06d' "${CKPT_STEP}")"
  else
    latest="$(ls -d "${dir}"/checkpoint_model_*/ 2>/dev/null | sort | tail -n 1 || true)"
    latest="${latest%/}"
  fi
  if [ -z "${latest}" ] || [ ! -f "${latest}/model.pt" ]; then
    echo "[infer][stage${st}] no usable checkpoint (${latest:-none}) under ${dir} -- skipping."
    continue
  fi
  ckpt="${latest}/model.pt"
  step="$(basename "${latest}" | sed 's/checkpoint_model_//')"
  echo "[infer][stage${st}] latest ckpt: ${ckpt} (step ${step})"

  # Per-stage output subfolder: outputs/stage1/, outputs/stage2/, outputs/stage3/.
  stage_out="${OUT_DIR}/stage${st}"
  mkdir -p "${stage_out}"

  i=0
  for ref in "${REFS[@]}"; do
    base="${ref%_ref.jpg}"
    video="${base}.mp4"
    mask="${base}_mask.mp4"
    name="$(basename "${base}")"
    save="${stage_out}/${SNAME[$st]}_step${step}_g$(printf '%02d' "${i}")_${name}.mp4"
    i=$((i + 1))
    if [ ! -f "${video}" ] || [ ! -f "${mask}" ]; then
      echo "  [skip] missing video/mask for ${name}"
      continue
    fi
    echo "  -> group $((i - 1)): ${name}"
    python inference.py \
      --config_path "${CONFIG}" \
      --generator_ckpt "${ckpt}" \
      --dreamidv_root "${DREAMIDV_ROOT}" \
      --ref_video "${video}" \
      --ref_image "${ref}" \
      --mask "${mask}" \
      --save_file "${save}" \
      --context_path "${CONTEXT_PATH}" \
      --size "${SIZE}" \
      --frame_num "${FRAME_NUM}" \
      --fps "${FPS}" \
      || echo "  [warn][stage${st}] inference FAILED for ${name} (see traceback above)"
  done
done

echo "[infer] done. Outputs in ${OUT_DIR}"
