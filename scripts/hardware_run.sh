#!/usr/bin/env bash
# Source this (not execute) in every hardware-campaign terminal: sets campaign vars, activates the
# venv, sources the workspace. Per-machine values come from .arc-local.env (run scripts/setup.sh).

_hr_src="${BASH_SOURCE[0]:-$0}"
export ARC_ROOT="$(cd "$(dirname "$_hr_src")/.." && pwd)"

if [ -f "$ARC_ROOT/.arc-local.env" ]; then
  # shellcheck disable=SC1091
  source "$ARC_ROOT/.arc-local.env"
else
  echo "hardware_run: no .arc-local.env — run scripts/setup.sh first" >&2
fi

# Campaign layout (override CAMPAIGN_DIR before sourcing to name a run).
export SCENES="${SCENES:-$ARC_ROOT/research/penn_gat/results/hardware_scenes_final_20260831/scenes}"
export CAMPAIGN_DIR="${CAMPAIGN_DIR:-$ARC_ROOT/research/penn_gat/results/hardware_campaign_$(date +%Y%m%d)}"
export HW_YAML="$ARC_ROOT/src/arc/single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc_hardware.yaml"
mkdir -p "$CAMPAIGN_DIR/bags" "$CAMPAIGN_DIR/logs"

[ -f "$ARC_ROOT/.venv/bin/activate" ] && source "$ARC_ROOT/.venv/bin/activate"
for d in "${ROS_DISTRO_EXPECTED:-}" humble jazzy; do
  [ -n "$d" ] && [ -f "/opt/ros/$d/setup.bash" ] && { source "/opt/ros/$d/setup.bash"; break; }
done
source "$ARC_ROOT/install_${ROS_DISTRO}/setup.bash" 2>/dev/null \
  || source "$ARC_ROOT/install/setup.bash" 2>/dev/null \
  || echo "hardware_run: no install/ tree — run scripts/build.sh"

set_arm() {
  case "$1" in
    adaptive) sed -i 's/^\( *penn_enabled: *\).*/\1true/' "$HW_YAML" ;;
    fixed5)   sed -i 's/^\( *penn_enabled: *\).*/\1false/' "$HW_YAML"
              sed -i 's/^\( *cbf_gamma_obs: *\).*/\15.0/'  "$HW_YAML" ;;
    fixed8)   sed -i 's/^\( *penn_enabled: *\).*/\1false/' "$HW_YAML"
              sed -i 's/^\( *cbf_gamma_obs: *\).*/\18.0/'  "$HW_YAML" ;;
    *) echo "usage: set_arm adaptive|fixed5|fixed8" >&2; return 1 ;;
  esac
  echo "arm set to $1"
}

cleanup_trial() {
  pkill -9 -f 'autopilot_sv_cbf_rate_node' 2>/dev/null
  pkill -9 -f 'synthetic_obstacle_publisher' 2>/dev/null
  pkill -9 -f 'lidar_obstacle_publisher' 2>/dev/null
  pkill -9 -f 'rviz2' 2>/dev/null
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  echo "cleaned up"
}

echo "hardware_run sourced — device=${ARC_DEVICE:-?} MOTIVE_PC_IP=${MOTIVE_PC_IP:-unset} UAV_PREFIX=${UAV_PREFIX:-unset}"
