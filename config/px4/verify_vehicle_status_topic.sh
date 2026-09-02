#!/usr/bin/env bash
# Run this once MicroXRCEAgent is bridging the REAL Pixhawk (not SITL),
# before trusting single_vehicle_cbf_rate_arc or operator_arm_node's
# vehicle_status subscriptions.
#
# The versioned suffix PX4's uxrce_dds_client publishes vehicle_status under
# has drifted twice already: SITL PX4 v1.17.0 published vehicle_status_v1
# (not _v4); SITL PX4 v1.15.4 (as of 2026-09-01) publishes only the
# unversioned "vehicle_status" (both _v1 and _v4 have zero publishers). This
# has already caused a silent failure -- armed_ never updates and the real
# rate-setpoint publish, gated on armed_, never fires -- so don't assume the
# real Pixhawk's firmware build matches whatever SITL last published; check
# it directly.
#
# Usage:
#   source /opt/ros/humble/setup.bash
#   source <your_ws>/install/setup.bash
#   ./verify_vehicle_status_topic.sh

set -euo pipefail

check_topic() {
  local topic="$1"
  local info
  info=$(ros2 topic info -v "$topic" 2>/dev/null || true)
  local pub_count
  pub_count=$(echo "$info" | grep -A1 "Publisher count:" | head -1 | grep -oE '[0-9]+' || echo "0")
  echo "$pub_count"
}

echo "Checking fmu/out/vehicle_status, _v1, and _v4 for live publishers..."
echo "(if all three show 0, MicroXRCEAgent likely isn't running/connected yet -- fix that first)"
echo

unversioned_count=$(check_topic "fmu/out/vehicle_status")
v1_count=$(check_topic "fmu/out/vehicle_status_v1")
v4_count=$(check_topic "fmu/out/vehicle_status_v4")

echo "fmu/out/vehicle_status    publisher count: ${unversioned_count}"
echo "fmu/out/vehicle_status_v1 publisher count: ${v1_count}"
echo "fmu/out/vehicle_status_v4 publisher count: ${v4_count}"
echo

live_count=0
live_topic=""
for pair in "unversioned:${unversioned_count}" "_v1:${v1_count}" "_v4:${v4_count}"; do
  name="${pair%%:*}"
  count="${pair##*:}"
  if [ "$count" -gt 0 ]; then
    live_count=$((live_count + 1))
    live_topic="$name"
  fi
done

code_files="  control-barrier-functions-arc/single_vehicle_cbf_rate_arc/client_lib/src/single_vehicle_cbf_rate_client.cpp
  control-barrier-functions-arc/sitl_test_support/sitl_test_support/ground_station_stub_node.py
  control-barrier-functions-arc/hardware_test_support/hardware_test_support/operator_arm_node.py
  control-barrier-functions-arc/arc_experiment_gui/arc_experiment_gui/ros_node.py
  control-barrier-functions-arc/arc_experiment_gui/arc_experiment_gui/launcher.py"

if [ "$live_count" -eq 1 ] && [ "$live_topic" = "unversioned" ]; then
  echo "RESULT: unversioned vehicle_status is live -- matches what all five files above"
  echo "currently assume (fixed 2026-09-01 for PX4 v1.15.4). No code change needed."
elif [ "$live_count" -eq 1 ]; then
  echo "RESULT: ${live_topic} is live, the others are not -- this DIFFERS from what the"
  echo "code assumes (unversioned). Fix required in:"
  echo "$code_files"
  echo "before flying -- as-is, armed_/_armed never updates and the real rate/arm-command"
  echo "publish paths gated on it never fire."
elif [ "$live_count" -gt 1 ]; then
  echo "RESULT: more than one variant has publishers -- unexpected, investigate which one"
  echo "is the real uxrce_dds_client output before trusting any of them."
else
  echo "RESULT: none have a publisher. MicroXRCEAgent is probably not running or not"
  echo "connected -- this doesn't tell you which variant is right yet, it just means"
  echo "nothing is bridged. Fix the agent connection and re-run this script."
fi
