#!/usr/bin/env bash
# Repeated-trial SITL consistency measurement: launches the SAME obstacle
# scene N times in a row and logs each run separately, so
# parse_repeat_consistency.py can quantify how much the resulting flight
# path actually varies run to run. Companion to the dt-measurement fix in
# rate_autopilot_core.cpp/single_vehicle_cbf_rate_client.cpp -- run once
# with DISABLE_MEASURED_DT=1 (old always-nominal-dt behavior) and once
# without, same scene/N, to compare spread before/after.
#
# Process-cleanup pattern (cleanup() below) reused verbatim from
# run_arming_transient_sweep.sh -- SIGKILL'ing a `timeout`-bounded
# `ros2 launch` only kills the launch orchestrator, not the node processes
# it spawned, which then silently linger and contaminate every subsequent
# run's topics/ports (see run_arming_transient_sweep.sh's 2026-08-24 note).
set -o pipefail

REPO=/home/matt/arc_2026/control-barrier-functions-arc
SCENE="${SCENE:-$REPO/penn_gat_training/results/sitl_battery_20260824_20260824/scene_03.json}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to a results directory}"
TAG="${TAG:?set TAG, e.g. before or after}"
N="${N:-6}"
WINDOW_S="${WINDOW_S:-75}"
mkdir -p "$OUT_DIR/runs"

source /home/matt/arc_2026/.venv/bin/activate
source /opt/ros/*/setup.bash 2>/dev/null
source /home/matt/arc_2026/install/setup.bash 2>/dev/null

export HEADLESS=1
export PX4_SRC_DIR=/home/matt/PX4-Autopilot
export MICRO_XRCE_AGENT_SRC_DIR=/home/matt/PX4-Autopilot/Micro-XRCE-DDS-Agent

cleanup() {
  pkill -9 -f 'synthetic_obstacle_publisher' 2>/dev/null
  pkill -9 -f 'autopilot_sv_cbf_rate_node' 2>/dev/null
  pkill -9 -f 'ground_station_stub_node' 2>/dev/null
  pkill -9 -f 'px4_odom_bridge_node' 2>/dev/null
  pkill -9 -f 'rviz2' 2>/dev/null
  pkill -9 -f 'px4_sitl_default/bin/px4' 2>/dev/null
  pkill -9 -f 'gz sim --verbose' 2>/dev/null
  pkill -9 -f 'MicroXRCEAgent' 2>/dev/null
  return 0
}
cleanup

echo "=== $TAG battery: scene=$SCENE N=$N DISABLE_MEASURED_DT=${DISABLE_MEASURED_DT:-<unset>} start $(date +%T) ==="
for i in $(seq 0 $((N - 1))); do
  log="$OUT_DIR/runs/${TAG}_rep_$(printf '%02d' "$i").log"
  echo "  rep $i -> $log ($(date +%T))"
  timeout "${WINDOW_S}s" ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py \
    launch_rviz:=false \
    obstacle_file:="$SCENE" \
    > "$log" 2>&1
  cleanup
  sleep 5
done
echo "=== $TAG battery done $(date +%T) ==="
echo "REPEAT_BATTERY_DONE:$TAG"
