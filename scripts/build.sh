#!/usr/bin/env bash
# Build the workspace into a per-distro tree (build_$ROS_DISTRO /
# install_$ROS_DISTRO) so the Jazzy desktop and the Humble Jetson never fight
# over the same output directory. All extra args pass straight to colcon.
#
#   scripts/build.sh                      # build everything
#   scripts/build.sh --packages-select single_vehicle_cbf_rate_arc
set -euo pipefail

ARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ARC_ROOT"

[ -f .arc-local.env ] || { echo "build: no .arc-local.env — run scripts/setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
source .arc-local.env

# ROS
if [ -z "${ROS_DISTRO:-}" ]; then
  for d in "${ROS_DISTRO_EXPECTED:-}" jazzy humble; do
    [ -n "$d" ] && [ -f "/opt/ros/$d/setup.bash" ] && { source "/opt/ros/$d/setup.bash"; break; }
  done
fi
[ -n "${ROS_DISTRO:-}" ] || { echo "build: no ROS environment found" >&2; exit 1; }
if [ -n "${ROS_DISTRO_EXPECTED:-}" ] && [ "$ROS_DISTRO" != "$ROS_DISTRO_EXPECTED" ]; then
  echo "build: WARNING sourced ROS '$ROS_DISTRO' but this device expects '$ROS_DISTRO_EXPECTED'"
fi

# libtorch: hand the cmake prefix to the one package that needs it
# (single_vehicle_cbf_rate_arc) as a -D flag. Prefer an explicit path from the
# environment (the Humble container image sets ARC_TORCH_CMAKE_PREFIX_PATH;
# TORCH_CMAKE is a shorthand), else probe the project venv (native builds on
# the Jetson). We never activate the venv or export VIRTUAL_ENV — it is
# isolated and lacks build-time deps (catkin_pkg, empy, lark), so letting
# ament pick its python breaks message generation for every rosidl package.
torch_cmake_args=()
tcp="${ARC_TORCH_CMAKE_PREFIX_PATH:-${TORCH_CMAKE:-}}"
if [ -z "$tcp" ] && [ -x .venv/bin/python3 ]; then
  tcp="$(.venv/bin/python3 -c 'import torch; print(torch.utils.cmake_prefix_path, end="")' 2>/dev/null || true)"
fi
[ -n "$tcp" ] && torch_cmake_args=(--cmake-args "-DARC_TORCH_CMAKE_PREFIX_PATH=$tcp" --no-warn-unused-cli)

# rosidl message generation runs under the system python; it needs these.
missing=$(/usr/bin/python3 -c "
import importlib.util as u
print(' '.join(m for m,p in [('catkin_pkg','python3-catkin-pkg'),('em','python3-empy'),('lark','python3-lark')] if u.find_spec(m) is None))
" 2>/dev/null || true)
[ -n "$missing" ] && echo "build: WARNING system python missing rosidl deps ($missing) — 'sudo apt install python3-catkin-pkg python3-empy python3-lark'"

echo "build: device=${ARC_DEVICE:-?} ros=$ROS_DISTRO -> build_$ROS_DISTRO/"
exec colcon build \
  --symlink-install \
  --build-base "build_$ROS_DISTRO" \
  --install-base "install_$ROS_DISTRO" \
  "${torch_cmake_args[@]}" \
  "$@"
