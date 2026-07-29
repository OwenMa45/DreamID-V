#!/usr/bin/env bash
# Streaming face-swap inference on a SINGLE H200 card (1xH200; default card 0,
# override via CUDA_VISIBLE_DEVICES).
# Runs the LATEST checkpoint of each stage over the FIRST few (default 8) data
# groups from the Stage-0 input directory.
#
# Stage checkpoint locations (all machines share the same storage):
#   stage1 / stage2 were trained on the 2xH100 machine -> *_2h100 dirs;
#   stage3 (DMD) was trained on the 8xH200 node        -> stage3_dmd_8h200.
# Override per stage via STAGE{1,2,3}_DIR if you retrain elsewhere.
#
# Data groups share a base name (same triple Stage-0 consumed):
#   <base>.mp4       source/driving video
#   <base>_mask.mp4  face-region mask  (passed via --mask, so no DWPose)
#   <base>_ref.jpg   reference identity face
#
# Outputs -> forcing_baseline/outputs_1h200/ (kept separate from the 2xH100
# runs: filenames carry no machine tag, so a shared outputs/ would clobber).
#
# Usage:
#   bash scripts/infer_latest_1h200.sh              # all stages that have ckpts
#   STAGES=3 bash scripts/infer_latest_1h200.sh     # only stage 3
#   NUM_GROUPS=5 bash scripts/infer_latest_1h200.sh # first 5 groups
#   CUDA_VISIBLE_DEVICES=3 bash scripts/infer_latest_1h200.sh  # use card 3
#   STAGES=3 CKPT_STEP=500 bash scripts/infer_latest_1h200.sh
#       -> pin a specific step instead of the latest (e.g. the stage-3 DMD
#          deliverable is the step-500 ckpt of the 1000-step run)
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V}
MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}
INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
CONTEXT_PATH=${CONTEXT_PATH:-${DREAMIDV_ROOT}/dreamidv_wan_faster/context.pth}
# Reuse the existing 8h200-machine inference config (same shared-storage model
# paths; inference itself is single-GPU regardless).
CONFIG=${CONFIG:-configs/inference_8h200.yaml}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs_1h200}

# Match the Stage-0 latent geometry so the conditioning distribution lines up.
SIZE=${SIZE:-832*480}
# fps/duration alignment: FRAME_NUM<=0 processes the WHOLE source video and
# FPS<=0 saves at the source video's own fps -> output has the same frame
# count / fps / duration as the driving video. Set positive values to override.
FRAME_NUM=${FRAME_NUM:-0}
FPS=${FPS:-0}
NUM_GROUPS=${NUM_GROUPS:-8}

# Single H200 card (pick which one via CUDA_VISIBLE_DEVICES).
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

# Stage -> (logdir, short name). Stages 1-2 come from the 2xH100 training runs
# (shared storage), stage 3 from this node's 8-GPU DMD run.
declare -A LOGDIR=(
  [1]="${STAGE1_DIR:-checkpoints/chunkwise/stage1_ar_2h100}"
  [2]="${STAGE2_DIR:-checkpoints/chunkwise/stage2_cd_2h100}"
  [3]="${STAGE3_DIR:-checkpoints/chunkwise/stage3_dmd_8h200}"
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

  # Per-stage output subfolder: outputs_1h200/stage1/, .../stage2/, .../stage3/.
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
