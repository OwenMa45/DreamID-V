#!/usr/bin/env bash
# Stage-0 (8xH200 data-parallel): self-distill training data with the
# bidirectional DreamID-V-Faster teacher, split across 8 GPUs.
#
# Each GPU runs one process over a round-robin 1/8 slice of corpus.jsonl and
# writes its own LMDB shard; the shards are then merged into the single LMDB the
# trainer reads (${OUTPUT_LMDB}). The merged result is identical to a 1-GPU run,
# so no training/config changes are needed.
#
# Data: LivingSwap-style groups, each <base>{.mp4,_mask.mp4,_ref.jpg}
# (masks PROVIDED -> DWPose normally unused).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

DREAMIDV_ROOT=${DREAMIDV_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V}
CKPT_DIR=${CKPT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/wan2.1-t2v-1.3b}
MODELS_DIR=${MODELS_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/models/DreamID-V}
DREAMIDV_CKPT=${DREAMIDV_CKPT:-${MODELS_DIR}/dreamidv_faster.pth}

# Paired data directory (~1000 groups) and the manifest derived from it.
INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
MANIFEST=${MANIFEST:-${PROJECT_ROOT}/corpus.jsonl}
OUTPUT_LMDB=${OUTPUT_LMDB:-${PROJECT_ROOT}/dataset/swap_latents}

NUM_GPUS=${NUM_GPUS:-8}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}     # physical GPUs to use, comma-separated
LOGDIR=${LOGDIR:-${PROJECT_ROOT}/dataset/stage0_logs}
KEEP_SHARDS=${KEEP_SHARDS:-0}             # set 1 to keep per-GPU shard LMDBs
mkdir -p "${LOGDIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]; then
  echo "[stage0-8gpu] ERROR: GPU_LIST has ${#GPUS[@]} entries but NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

# Auto-build the manifest from the paired input dir if not already present.
if [ ! -f "${MANIFEST}" ]; then
  echo "[stage0-8gpu] manifest not found -> building from ${INPUT_DIR}"
  python -m tools.build_manifest --input_dir "${INPUT_DIR}" --output "${MANIFEST}"
fi

# DWPose fallback (groups missing a mask): link the onnx models into the
# hard-coded pose/models/ location before spawning workers.
POSE_DIR="${DREAMIDV_ROOT}/pose/models"
mkdir -p "${POSE_DIR}"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
  if [ ! -s "${POSE_DIR}/${f}" ] && [ -s "${MODELS_DIR}/${f}" ]; then
    ln -sf "${MODELS_DIR}/${f}" "${POSE_DIR}/${f}"
    echo "[stage0-8gpu] linked ${f} -> ${POSE_DIR}/${f}"
  fi
done

SHARD_DIRS=()
PIDS=()
echo "[stage0-8gpu] launching ${NUM_GPUS} workers over GPUs ${GPU_LIST}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  gpu=${GPUS[$i]}
  shard_dir="${OUTPUT_LMDB}_shard${i}"
  SHARD_DIRS+=("${shard_dir}")
  log="${LOGDIR}/stage0_shard${i}.log"
  echo "[stage0-8gpu]  shard ${i} -> GPU ${gpu}, out=${shard_dir}, log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m tools.syncid_generate_data \
    --dreamidv_root "${DREAMIDV_ROOT}" \
    --ckpt_dir "${CKPT_DIR}" \
    --dreamidv_ckpt "${DREAMIDV_CKPT}" \
    --manifest "${MANIFEST}" \
    --output_lmdb "${shard_dir}" \
    --num_shards "${NUM_GPUS}" \
    --shard_id "${i}" \
    --device_id 0 \
    --size 832*480 \
    --frame_num 81 \
    --sampling_steps 12 \
    --sample_shift 5.0 \
    --guide_scale_img 4.0 \
    > "${log}" 2>&1 &
  PIDS+=("$!")
done

# Wait for every worker; collect failures without aborting the rest.
fail=0
for i in $(seq 0 $((NUM_GPUS - 1))); do
  if wait "${PIDS[$i]}"; then
    echo "[stage0-8gpu] shard ${i} OK  ($(grep -h 'accepted' "${LOGDIR}/stage0_shard${i}.log" | tail -n1))"
  else
    echo "[stage0-8gpu] shard ${i} FAILED (see ${LOGDIR}/stage0_shard${i}.log):" >&2
    tail -n 15 "${LOGDIR}/stage0_shard${i}.log" >&2 || true
    fail=1
  fi
done
if [ "${fail}" -ne 0 ]; then
  echo "[stage0-8gpu] one or more shards failed; NOT merging. Fix and rerun." >&2
  exit 1
fi

echo "[stage0-8gpu] merging ${NUM_GPUS} shards -> ${OUTPUT_LMDB}"
python -m tools.merge_lmdb_shards --shards "${SHARD_DIRS[@]}" --output "${OUTPUT_LMDB}"

if [ "${KEEP_SHARDS}" -eq 0 ]; then
  echo "[stage0-8gpu] removing shard LMDBs (set KEEP_SHARDS=1 to keep)"
  for d in "${SHARD_DIRS[@]}"; do rm -rf "${d}"; done
fi

echo "[stage0-8gpu] DONE -> ${OUTPUT_LMDB}"
