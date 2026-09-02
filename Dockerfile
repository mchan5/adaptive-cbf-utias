# Humble dev image for the desktop and laptop (both Ubuntu 24.04, so Humble
# only runs here in a container). The Jetson runs Humble natively and does not
# use this. The repo is bind-mounted at /ws at run time (see scripts/dev.sh) —
# nothing from the workspace is COPYed in, so the image stays reusable.
FROM ros:humble

ARG USERNAME=arc
ARG UID=1000
ARG GID=1000
ARG WITH_GAZEBO=1
# Match the desktop/Jetson libtorch so the client_lib headers compile and the
# TorchScript checkpoint loads with identical numerics.
ARG TORCH_VERSION=2.13.0+cpu

# System + ROS build dependencies for the whole workspace.
# The PX4 uXRCE-DDS agent is a runtime component, not a build dep — run it as a
# sidecar (`docker run --rm --net=host microros/micro-ros-agent:humble udp4 -p
# 8888`) or on the host. Revisit once the SITL host/container split is settled.
# GCC 12: jammy's default GCC 11 is too old for recent libtorch ATen headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget curl gnupg lsb-release sudo unzip \
      gcc-12 g++-12 \
      python3-pip python3.10-venv \
      python3-colcon-common-extensions python3-catkin-pkg python3-empy python3-lark \
      libeigen3-dev libboost-system-dev \
      ros-humble-tf2-ros ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs \
      ros-humble-visualization-msgs ros-humble-interactive-markers \
      ros-humble-rmw-cyclonedds-cpp ros-humble-eigen3-cmake-module \
      python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# Gazebo Harmonic (gz-transport13 / gz-msgs10) — PX4 v1.15 SITL and the
# gz_optitrack_ros2_emulator link against it. --build-arg WITH_GAZEBO=0 to skip
# (e.g. if Gazebo runs on the host instead).
RUN if [ "$WITH_GAZEBO" = "1" ]; then \
      wget -qO /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
        https://packages.osrfoundation.org/gazebo.gpg && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
      apt-get update && apt-get install -y --no-install-recommends gz-harmonic && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# libtorch for single_vehicle_cbf_rate_arc/client_lib, isolated in a venv. A
# plain `pip install torch` pulls a new setuptools into system python, which
# breaks Humble's ament_python setup.py-develop path for every pure-Python
# package. The venv keeps that upgrade off the system interpreter that colcon
# uses. ARC_TORCH_CMAKE_PREFIX_PATH is what scripts/build.sh passes through to
# the one package that needs it.
RUN python3 -m venv --system-site-packages /opt/torchvenv \
    && /opt/torchvenv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/torchvenv/bin/pip install --no-cache-dir "torch==${TORCH_VERSION}" \
         --index-url https://download.pytorch.org/whl/cpu
ENV ARC_TORCH_CMAKE_PREFIX_PATH=/opt/torchvenv/lib/python3.10/site-packages/torch/share/cmake

# RViz for the CBF/obstacle visualisation (desktop/laptop). Headless SITL runs
# with launch_rviz:=false; this is only for interactive use.
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID so bind-mounted build_humble/ and
# install_humble/ are not written as root. dev.sh passes the real values.
RUN groupadd -g "$GID" "$USERNAME" \
    && useradd -m -u "$UID" -g "$GID" -s /bin/bash "$USERNAME" \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME" \
    && chmod 0440 /etc/sudoers.d/"$USERNAME"
USER $USERNAME

RUN git config --global --add safe.directory /ws \
    && { echo 'source /opt/ros/humble/setup.bash'; \
         echo 'export ARC_ROOT=/ws'; \
         echo 'export TORCH_CMAKE=$ARC_TORCH_CMAKE_PREFIX_PATH'; \
         echo '[ -f /ws/install_humble/setup.bash ] && source /ws/install_humble/setup.bash'; \
       } >> ~/.bashrc

WORKDIR /ws
CMD ["bash"]
