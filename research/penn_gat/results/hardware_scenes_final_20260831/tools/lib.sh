#!/usr/bin/env bash
ARC=/home/matt/arc_2026
REPO=/home/matt/arc_2026/control-barrier-functions-arc
CAMP=/tmp/claude-1000/-home-matt-arc-2026/136e952b-df92-4f40-ac17-379ba775814a/scratchpad/authsweep
PX4_SRC=/home/matt/PX4-Autopilot
XRCE_SRC=/home/matt/PX4-Autopilot/Micro-XRCE-DDS-Agent
# INSTALL yaml is what the launch actually reads (get_package_share_directory);
# the git-tracked SOURCE yaml is left pristine.
YAML="$ARC/install/single_vehicle_cbf_rate_arc/share/single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml"
SRC_YAML="$REPO/single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml"

# Kill SITL processes. Patterns match ACTUAL executable/node names only --
# NOT bare repo paths, which also appear in this session's shell command
# lines (a broad `pkill -f single_vehicle_cbf_rate` was killing the caller).
sitl_cleanup() {
  pkill -9 -f 'ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch' 2>/dev/null
  pkill -9 -f 'autopilot_sv_cbf_rate_node'      2>/dev/null
  pkill -9 -f 'ground_station_stub_node'        2>/dev/null
  pkill -9 -f 'px4_odom_bridge_node'            2>/dev/null
  pkill -9 -f 'synthetic_obstacle_publisher'    2>/dev/null
  pkill -9 -f 'rviz2 -d'                        2>/dev/null
  pkill -9 -f 'px4_sitl_default/bin/px4'        2>/dev/null
  pkill -9 -f 'PX4-Autopilot/build/px4_sitl'    2>/dev/null
  pkill -9 -f 'gz sim --verbose'                2>/dev/null
  pkill -9 -f 'ruby .*gz sim'                   2>/dev/null
  pkill -9 -f 'Micro-XRCE-DDS-Agent/build/MicroXRCEAgent' 2>/dev/null
  sleep 1
  return 0
}
sitl_env() {
  source "$ARC/.venv/bin/activate" 2>/dev/null
  source /opt/ros/humble/setup.bash 2>/dev/null
  source "$ARC/install/setup.bash" 2>/dev/null
  export HEADLESS=1 PX4_SRC_DIR="$PX4_SRC" MICRO_XRCE_AGENT_SRC_DIR="$XRCE_SRC"
}
set_authority() {
  python3 - "$YAML" "$1" "$2" "$3" <<'PY'
import re, sys
y, a, kp, kd = sys.argv[1:5]
t = open(y).read()
t = re.sub(r'(\n\s*nom_accel_max:)\s*[0-9.]+', rf'\g<1> {float(a)}', t)
t = re.sub(r'(\n\s*nom_pos_kp:)\s*[0-9.]+',   rf'\g<1> {float(kp)}', t)
t = re.sub(r'(\n\s*nom_vel_kd:)\s*[0-9.]+',   rf'\g<1> {float(kd)}', t)
open(y, 'w').write(t)
PY
}
set_penn() {
  python3 - "$YAML" "$1" <<'PY'
import re, sys
y, v = sys.argv[1:3]
t = open(y).read()
t = re.sub(r'(\n\s*penn_enabled:)\s*\w+', rf'\g<1> {v}', t)
open(y, 'w').write(t)
PY
}
set_gamma() {
  python3 - "$YAML" "$1" <<'PY'
import re, sys
y, g = sys.argv[1:3]
t = open(y).read()
t = re.sub(r'(\n\s*cbf_gamma_obs:)\s*[0-9.]+', rf'\g<1> {float(g)}', t)
open(y, 'w').write(t)
PY
}
show_authority() { grep -E "nom_accel_max:|nom_pos_kp:|nom_vel_kd:|cbf_gamma_obs:|penn_enabled:" "$YAML" | sed 's/#.*//'; }
set_stall() {  # $1 = stall_speed_threshold (0.0 disables stall-escape)
  python3 - "$YAML" "$1" <<'PY'
import re, sys
y, v = sys.argv[1:3]
t = open(y).read()
t = re.sub(r'(\n\s*stall_speed_threshold:)\s*[0-9.]+', rf'\g<1> {float(v)}', t)
open(y, 'w').write(t)
PY
}
