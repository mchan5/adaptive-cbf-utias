#!/usr/bin/env bash
# Record a bench bag for test/test_bag_replay.py. Run during lab time with the
# Livox driver AND an odom source (mocap or onboard EKF) both live.
#
#   ros2 launch livox_ros_driver2 msg_MID360s_launch.py     # terminal A
#   ros2 topic hz /livox/lidar                              # confirm it flows
#   ros2 topic hz /state_estimator/local_position/odom      # confirm this too
#   tools/record_bench_bag.sh                               # terminal B, Ctrl-C to stop
#
# Keep the take short (~20-40 s):
#   1. Put a box / pole at a position you can MEASURE in the odom (world/ENU)
#      frame. Stay inside max_range_m and the [z_min_m, z_max_m] band.
#   2. Start recording.
#   3. Move the vehicle (cart + mocap, or hand-carried with EKF live -- the
#      sensor must move WITH the body the odom describes) so the obstacle
#      sweeps through the FOV for several seconds.
#   4. Ctrl-C.
#   5. Fill test/fixtures/bench_replay.yaml: set bag_path, the measured
#      center_xyz / radius_m, and the mount offset you used. Commit that file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../test/fixtures/bench_replay.bag}"
PC_TOPIC="${PC_TOPIC:-/livox/lidar}"
ODOM_TOPIC="${ODOM_TOPIC:-/state_estimator/local_position/odom}"

if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing $OUT -- move it aside or pass a new path" >&2
  exit 1
fi

echo "recording:"
echo "  $PC_TOPIC"
echo "  $ODOM_TOPIC"
echo "-> $OUT   (Ctrl-C to stop)"
exec ros2 bag record -o "$OUT" "$PC_TOPIC" "$ODOM_TOPIC"
