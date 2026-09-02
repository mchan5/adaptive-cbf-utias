#!/usr/bin/env bash
# Gazebo SITL for the Humble container (desktop/laptop only — the Jetson flies
# real hardware). PX4-Autopilot and the uXRCE-DDS agent live in a Docker volume
# (arc-px4), built once for jammy, so the host's own native PX4 build is never
# touched. `setup` snapshots the result as arc-humble:sitl.
#
#   scripts/sitl.sh setup          # one-time: seed volume, build PX4 + agent, snapshot image
#   scripts/sitl.sh run [scene]    # launch SITL headless (scene default: slalom)
#   scripts/sitl.sh gui [scene]    # same, with the Gazebo GUI + RViz (WSLg / X11)
#   scripts/sitl.sh shell          # bash into the SITL container
#   scripts/sitl.sh stop | rm
#
# Env overrides: PX4_SRC, XRCE_SRC (host source dirs, default ~/PX4-Autopilot,
# ~/Micro-XRCE-DDS-Agent), PX4_VER (clone tag when no host source), SITL_RESEED=1
# (re-copy source), SITL_REBUILD=1 (force PX4 rebuild).
set -euo pipefail

ARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ARC_ROOT"

BASE_IMAGE=arc-humble:dev
SITL_IMAGE=arc-humble:sitl
NAME=arc-sitl
VOL=arc-px4
PX4_VER="${PX4_VER:-v1.15.4}"
PX4_SRC="${PX4_SRC:-$HOME/PX4-Autopilot}"
XRCE_SRC="${XRCE_SRC:-$HOME/Micro-XRCE-DDS-Agent}"

img() { docker image inspect "$SITL_IMAGE" >/dev/null 2>&1 && echo "$SITL_IMAGE" || echo "$BASE_IMAGE"; }

ensure_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME" >/dev/null
    return
  fi
  local m=( -v "$ARC_ROOT":/ws -v "$VOL":/opt/px4 -w /ws --net=host
    -e ARC_ROOT=/ws
    -e PX4_SRC_DIR=/opt/px4/PX4-Autopilot
    -e MICRO_XRCE_AGENT_SRC_DIR=/opt/px4/Micro-XRCE-DDS-Agent
    -e HEADLESS=1
    -e GZ_SIM_RESOURCE_PATH=/opt/px4/PX4-Autopilot/Tools/simulation/gz/models:/opt/px4/PX4-Autopilot/Tools/simulation/gz/worlds
    -e LD_LIBRARY_PATH=/opt/torchvenv/lib/python3.10/site-packages/torch/lib )
  [ -d /mnt/wslg ] && m+=( -v /mnt/wslg:/mnt/wslg -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir )
  [ -d /tmp/.X11-unix ] && m+=( -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY )
  docker run -d --name "$NAME" "${m[@]}" "$(img)" sleep infinity >/dev/null
}

case "${1:-run}" in
setup)
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
    || { echo "build the base image first:  scripts/dev.sh build" >&2; exit 1; }
  docker volume create "$VOL" >/dev/null

  if [ "${SITL_RESEED:-0}" = 1 ] || ! docker run --rm -v "$VOL":/opt/px4 "$BASE_IMAGE" \
        test -d /opt/px4/PX4-Autopilot/.git 2>/dev/null; then
    echo "== seeding PX4 ($PX4_VER) + agent into volume $VOL =="
    if [ -d "$PX4_SRC/.git" ]; then
      docker run --rm --user root -v "$VOL":/opt/px4 -v "$PX4_SRC":/s-px4:ro -v "$XRCE_SRC":/s-agent:ro "$BASE_IMAGE" bash -lc '
        apt-get update -qq && apt-get install -y -qq rsync >/dev/null
        rsync -a --delete --exclude "/build/" /s-px4/   /opt/px4/PX4-Autopilot/
        rsync -a --delete --exclude "/build/" /s-agent/ /opt/px4/Micro-XRCE-DDS-Agent/
        chown -R 1000:1000 /opt/px4'
    else
      docker run --rm --user root -v "$VOL":/opt/px4 "$BASE_IMAGE" bash -lc "
        apt-get update -qq && apt-get install -y -qq git >/dev/null
        git clone --branch $PX4_VER --recurse-submodules --depth 1 https://github.com/PX4/PX4-Autopilot.git /opt/px4/PX4-Autopilot
        git clone --branch v2.4.2 --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/px4/Micro-XRCE-DDS-Agent
        chown -R 1000:1000 /opt/px4"
    fi
  fi

  docker rm -f "$NAME" 2>/dev/null || true
  ensure_container
  echo "== PX4 toolchain deps + build (agent, px4_sitl_default) =="
  docker exec "$NAME" bash -lc '
    set -e
    git config --global --add safe.directory "*"
    cd /opt/px4/PX4-Autopilot && bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
    source /opt/ros/humble/setup.bash
    export PATH=$HOME/.local/bin:$PATH
    if [ ! -x /opt/px4/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent ]; then
      cd /opt/px4/Micro-XRCE-DDS-Agent
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAGENT_BUILD_EXECUTABLE=ON
      cmake --build build -j"$(nproc)"
    fi
    cd /opt/px4/PX4-Autopilot
    if [ "'"${SITL_REBUILD:-0}"'" = 1 ] || [ ! -x build/px4_sitl_default/bin/px4 ]; then
      make px4_sitl_default -j"$(nproc)"
    fi
    test -x build/px4_sitl_default/bin/px4
    test -x /opt/px4/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent
    echo "PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc'
  echo "== snapshot $SITL_IMAGE =="
  docker commit -c 'CMD ["bash"]' "$NAME" "$SITL_IMAGE" >/dev/null
  docker rm -f "$NAME" >/dev/null
  echo "done.  scripts/sitl.sh run"
  ;;

run|gui)
  scene="${2:-slalom}"
  exec_args=( -it )
  if [ "$1" = gui ]; then
    rviz=true; hl_cmd='unset HEADLESS'
    exec_args+=( -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_QPA_PLATFORM=xcb )
    # recreate the container if it lacks the X11 socket mount
    docker inspect "$NAME" --format '{{range .Mounts}}{{.Destination}} {{end}}' 2>/dev/null \
      | grep -q /tmp/.X11-unix || docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "SITL+GUI: scene=$scene  (software GL — slow; Ctrl-C to stop)"
  else
    rviz=false; hl_cmd='export HEADLESS=1'
    echo "SITL: scene=$scene  (headless; Ctrl-C to stop)"
  fi
  ensure_container
  exec docker exec "${exec_args[@]}" "$NAME" bash -lc "
    source /opt/ros/humble/setup.bash && source /ws/install_humble/setup.bash
    export PATH=\$HOME/.local/bin:\$PATH; $hl_cmd
    exec ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py \
         obstacle_scene:=$scene launch_rviz:=$rviz"
  ;;

shell) ensure_container; exec docker exec -it "$NAME" bash ;;
stop)  docker stop "$NAME" ;;
rm)    docker rm -f "$NAME" ;;
*) echo "usage: scripts/sitl.sh {setup|run [scene]|shell|stop|rm}" >&2; exit 1 ;;
esac
