#!/usr/bin/env bash
# Tier 4b (TESTING.md): isolates the arming/infeasible-braking recovery
# transient found 2026-08-24 (every authority level has the same benign
# QP-infeasible tick right after arming; what changes with authority is how
# violently the RECOVERY from it overshoots). Obstacle-free scenes, short
# (~12s, not 75s) window -- the transient resolves or doesn't within the
# first few seconds, no goal/obstacle needed to observe it.
#
# Sweeps nom_accel_max/nom_pos_kp/nom_vel_kd by editing the deploy YAML
# between batches (same pattern the 2026-08-24 bisection scripts used),
# running N cycles per authority level.
set -o pipefail

REPO=/home/matt/arc_2026/control-barrier-functions-arc
YAML="$REPO/single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml"
SCENES_DIR="$REPO/penn_gat_training/results/sitl_empty_20260824"
OUT_DIR=/tmp/claude-1000/-home-matt-arc-2026/d0f1c836-ac05-401a-87f7-21d68eb30bba/scratchpad/arming_transient_20260824
STUB=/tmp/claude-1000/-home-matt-arc-2026/d0f1c836-ac05-401a-87f7-21d68eb30bba/scratchpad/xrce_stub
WINDOW_S="${WINDOW_S:-12}"
N_PER_LEVEL="${N_PER_LEVEL:-15}"
mkdir -p "$OUT_DIR/runs"

source /home/matt/arc_2026/.venv/bin/activate
source /opt/ros/*/setup.bash 2>/dev/null
source /home/matt/arc_2026/ros2_ws/install/setup.bash 2>/dev/null

cleanup() {
  # 2026-08-24 fix: the original 3-pattern cleanup (still below) missed the
  # ROS2-launched node processes entirely -- when a launch is torn down via
  # SIGKILL (not a clean timeout exit), those never get a chance to shut
  # down and are orphaned, invisible to every SUBSEQUENT launch's own
  # process tree. Found via a synthetic_obstacle_publisher instance that
  # had been running since the session's very first (SIGKILL'd) launch,
  # silently feeding stale obstacle data into every later SITL episode's
  # /arc/obstacles topic alongside each new episode's real publisher. See
  # TESTING.md's Tier 4 section for the full incident.
  pkill -9 -f 'synthetic_obstacle_publisher' 2>/dev/null
  pkill -9 -f 'autopilot_sv_cbf_rate_node' 2>/dev/null
  pkill -9 -f 'ground_station_stub_node' 2>/dev/null
  pkill -9 -f 'px4_odom_bridge_node' 2>/dev/null
  pkill -9 -f 'rviz2' 2>/dev/null
  pkill -9 -f 'PX4-Autopilot/build' 2>/dev/null
  pkill -9 -f 'ign gazebo|gz-sim|gzserver' 2>/dev/null
  pkill -9 -f 'build/MicroXRCEAgent' 2>/dev/null
  return 0
}

set_authority() {
  local accel="$1" kp="$2" kd="$3"
  python3 - "$YAML" "$accel" "$kp" "$kd" << 'EOF'
import re, sys
yaml_path, accel, kp, kd = sys.argv[1:5]
with open(yaml_path) as f:
    text = f.read()
text = re.sub(r'(\n\s*nom_accel_max:)\s*[0-9.]+', rf'\g<1> {accel}', text)
text = re.sub(r'(\n\s*nom_pos_kp:)\s*[0-9.]+', rf'\g<1> {kp}', text)
text = re.sub(r'(\n\s*nom_vel_kd:)\s*[0-9.]+', rf'\g<1> {kd}', text)
with open(yaml_path, 'w') as f:
    f.write(text)
EOF
}

# accel,kp,kd combos -- the 2026-08-24 bisection points (kp=kd=accel/2),
# spanning known-clean (8,10,12) through known-failed (13,14).
combos=(
  "8 4 4"
  "10 5 5"
  "12 6 6"
  "13 6.5 6.5"
  "14 7 7"
)

for combo in "${combos[@]}"; do
  read -r accel kp kd <<< "$combo"
  tag="a${accel//./_}"
  set_authority "$accel" "$kp" "$kd"
  echo "=== authority ($accel,$kp,$kd) -> $tag start $(date +%T) ==="
  for i in $(seq 0 $((N_PER_LEVEL - 1))); do
    log="$OUT_DIR/runs/${tag}_cycle_$(printf '%02d' "$i").log"
    timeout "${WINDOW_S}s" ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py \
      obstacle_file:="$SCENES_DIR/scene_00.json" \
      px4_src_dir:=/home/matt/PX4-Autopilot \
      micro_xrce_agent_src_dir:="$STUB" \
      > "$log" 2>&1
    cleanup
    sleep 8
  done
  echo "=== authority ($accel,$kp,$kd) done $(date +%T) ==="
done
echo "ARMING_TRANSIENT_SWEEP_DONE"
