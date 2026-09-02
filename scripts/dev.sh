#!/usr/bin/env bash
# Start (or re-enter) the Humble dev container with this repo bind-mounted.
# The desktop and laptop run the whole ROS 2 stack this way; the Jetson builds
# natively and does not need it.
#
#   scripts/dev.sh            # build the image if missing, then start/attach
#   scripts/dev.sh build      # (re)build the image
#   scripts/dev.sh rm         # remove the container (keeps the image)
set -euo pipefail

ARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ARC_ROOT"

IMAGE=arc-humble:dev
NAME=arc-humble
build_args=(--build-arg "UID=$(id -u)" --build-arg "GID=$(id -g)")

case "${1:-up}" in
  build) exec docker build -t "$IMAGE" "${build_args[@]}" . ;;
  rm)    exec docker rm -f "$NAME" ;;
esac

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build -t "$IMAGE" "${build_args[@]}" .

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null
  exec docker exec -it "$NAME" bash
fi

run_args=(
  -it --name "$NAME"
  --net=host
  -e DISPLAY -e QT_X11_NO_MITSHM=1
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v "$ARC_ROOT":/ws -w /ws
)
[ -d /mnt/wslg ] && run_args+=( -v /mnt/wslg:/mnt/wslg:rw -e WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir )
[ -e /dev/dri ] && run_args+=( --device /dev/dri )

exec docker run "${run_args[@]}" "$IMAGE" bash
