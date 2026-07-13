#!/usr/bin/env bash
# Build a self-contained HTML report (source video | reference face | swapped
# result) from the inference outputs, for showing/reporting to others.
#
# It scans outputs/stage<N>/ (produced by scripts/infer_latest_2h100.sh), matches
# each result back to its Stage-0 source video + reference image, copies all three
# into report_html/report/assets/ and writes report_html/report/index.html.
#
# Usage:
#   bash scripts/make_report_2h100.sh                 # all stages that have results
#   STAGES=1 bash scripts/make_report_2h100.sh        # only stage 1
#   STAGES=1,3 bash scripts/make_report_2h100.sh      # stage 1 and 3
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/mrq/swapsf/pure_dreamidv/DreamID-V/forcing_baseline}
cd "${PROJECT_ROOT}"

INPUT_DIR=${INPUT_DIR:-/inspire/hdd/global_user/liumingyu-253208120284/lzk/codes/Causal-Forcing_LivingSwap/datasets/humanvid_5000/part_004/input}
OUTPUTS_DIR=${OUTPUTS_DIR:-${PROJECT_ROOT}/outputs}
REPORT_DIR=${REPORT_DIR:-${PROJECT_ROOT}/report_html/report}
STAGES=${STAGES:-all}
TITLE=${TITLE:-"Causal DreamID-V · Face-Swap Report"}

python report_html/gen_report.py \
  --outputs_dir "${OUTPUTS_DIR}" \
  --input_dir "${INPUT_DIR}" \
  --report_dir "${REPORT_DIR}" \
  --stages "${STAGES}" \
  --title "${TITLE}"

echo
echo "[report] serve locally with:"
echo "    python -m http.server -d ${REPORT_DIR} 8000"
echo "[report] then open http://<server-ip>:8000/  (or copy ${REPORT_DIR} elsewhere)"
