#!/usr/bin/env bash
# Record a bench bag for test/test_bag_replay.py. Run during lab time with the Livox driver AND an
# odom source (mocap or onboard EKF) both live.
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
