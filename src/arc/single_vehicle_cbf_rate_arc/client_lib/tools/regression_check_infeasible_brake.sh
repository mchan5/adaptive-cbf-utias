#!/usr/bin/env bash
# Fast regression fixture for the QP-infeasibility braking fix (rate_autopilot_core.cpp,
# 2026-08-23).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/../ros2_ws/build/single_vehicle_cbf_rate_arc/body_rate_closed_loop_diag"
SCENES="${REPO_ROOT}/penn_gat_training/results/closed_loop_calibration_dense_20260823/scenes_seed20260921.pt"
CKPT="${1:-${REPO_ROOT}/penn_gat_training/nn_model/checkpoint/gat_penn_weights_widerange.pt}"
OUT_CSV="$(mktemp)"

if [[ ! -x "${BIN}" ]]; then
  echo "FATAL: ${BIN} not built. Build single_vehicle_cbf_rate_arc first." >&2
  exit 1
fi

PENN_GAMMA_MIN=1.3 PENN_GAMMA_MAX=8.0 \
EPISTEMIC_THRESHOLD=0.10687255859375 CVAR_BOUNDARY=0.0023882027249783394 LCB_K=0.0 \
ADAPTIVE_ONLY=1 \
  "${BIN}" "${SCENES}" "${CKPT}" "${OUT_CSV}" > /dev/null 2>&1

fail=0
for scene in 25 26; do
  min_h=$(awk -F, -v s="${scene}" '$3==s {print $6}' "${OUT_CSV}")
  if [[ -z "${min_h}" ]]; then
    echo "FAIL: scene ${scene} not found in output" >&2
    fail=1
    continue
  fi
  ok=$(awk -v v="${min_h}" 'BEGIN{print (v>=0)?"1":"0"}')
  if [[ "${ok}" == "1" ]]; then
    echo "PASS: scene ${scene} min_h_physical=${min_h} (no collision)"
  else
    echo "FAIL: scene ${scene} min_h_physical=${min_h} (HARD COLLISION -- braking fix regressed)"
    fail=1
  fi
done

rm -f "${OUT_CSV}"
exit "${fail}"
