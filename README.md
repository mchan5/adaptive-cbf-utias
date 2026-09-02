# arc_2026

Online, uncertainty-aware adaptation of Control Barrier Function (CBF) safety parameters for autonomous quadrotors, from offline research through real-time PX4 flight.

This repository integrates a **Probabilistic Ensemble Neural Network + Graph Attention Network (PENN+GAT)** model into a CBF-QP safety filter that adjusts its own gain (γ / body-rate α) online, based on the robot's state, nearby obstacles, and the model's epistemic uncertainty in its own prediction — backing off toward conservative, fixed-gain behavior whenever a scene looks out-of-distribution.

The flight path is the body-rate HOCBF-QP safety filter (`single_vehicle_cbf_rate_arc`): C++/libtorch, running against PX4 firmware over the ROS 2 / DDS bridge, validated in Gazebo SITL and flown on hardware.

## Repository layout

| Path | What it is |
|---|---|
| [`src/arc/`](src/arc) | Our ROS 2 packages: the body-rate CBF-QP + PENN/GAT node (`single_vehicle_cbf_rate_arc`), obstacle perception (`lidar_obstacle_publisher`, `synthetic_obstacle_publisher`), the experiment GUI, and the HITL / SITL / hardware bench support. |
| [`src/vendor/`](src/vendor) | Third-party ROS packages, pinned to upstream commits (see [`src/vendor/VENDOR.md`](src/vendor/VENDOR.md)), never edited: `px4_msgs`, `px4_ros_com`, `fsc_autopilot_ros2_msgs`, `livox_ros_driver2`, `natnet_ros2`, `gz_optitrack_ros2_emulator`. |
| [`research/penn_gat/`](research/penn_gat) | PENN+GAT model, training, and CCCP calibration pipeline. Single source of truth for the model. |
| [`research/models/`](research/models) | Deployed checkpoints (Git LFS). Loaded at runtime by `penn_gamma_selector` on the flight path. |
| [`config/`](config) | Shared ROS params, PX4 config, and `config/devices/` — the per-machine env files that let the desktop, laptop and Jetson share one branch. |
| [`scripts/`](scripts) | `setup.sh` (detect device → write `.arc-local.env`), `dev.sh` (Humble dev container for desktop/laptop), `build.sh`, `hardware_run.sh`. |
| [`docs/`](docs) | Architecture, hardware runbook, and condensed design decisions. |
| `reference/` | Local only, gitignored — not on the Jetson, not in history. Dormant research kept for citation: `safe_control` (tkkim-robot sim framework), `online_adaptive_cbf` (2D ground-robot lineage, arXiv 2409.14616), `drone_cbf_penn` (superseded 3D model copy), `fsc_autopilot_ros2` (the FSC flight stack this work was extracted from). |
| `PX4-Autopilot/`, `micro-ROS-Agent/`, `Micro-XRCE-DDS-Agent/` | Third-party firmware / DDS bridge — clone separately, excluded via `.gitignore`. |

## Why adaptive CBF parameters

A CBF-QP safety filter is only as good as its gain: too conservative and the robot stalls or takes needlessly cautious paths, too aggressive and safety margins erode near obstacles. This project learns to pick that gain online instead of hand-tuning a fixed constant — a PENN+GAT model predicts, from the current scene graph (robot + obstacles + goal), which candidate gain minimizes risk, with a calibrated epistemic-uncertainty filter that falls back to safe defaults whenever the model is asked to extrapolate beyond its training distribution.

## Getting started

```sh
git clone <this repo> arc_2026 && cd arc_2026
scripts/setup.sh                  # detects the machine, writes .arc-local.env

# Desktop / laptop (Ubuntu 24.04 — Humble runs in a container):
scripts/dev.sh                    # build the image if needed, drop into a shell
scripts/build.sh                  # inside the container: colcon build → build_humble/

# Jetson (native Humble):
scripts/build.sh                  # builds directly into build_humble/
```

All three machines run ROS 2 Humble — native on the Jetson, in Docker
(`scripts/dev.sh`, repo bind-mounted at `/ws`) on the desktop and laptop.
`setup.sh` resolves `$ARC_ROOT` and the per-machine env for the box it runs on,
so the same `master` branch builds everywhere.

### Gazebo SITL (desktop / laptop)

```sh
scripts/sitl.sh setup            # one-time: PX4 v1.15.4 + uXRCE agent into a
                                 # Docker volume, built for jammy, snapshot as
                                 # arc-humble:sitl (host's native PX4 untouched)
scripts/sitl.sh run slalom       # fly a scene headless; scripts/sitl.sh shell for a prompt
```

Uses `~/PX4-Autopilot` + `~/Micro-XRCE-DDS-Agent` if present, else clones them.
`px4_msgs` must match the firmware — `release/1.15` for PX4 v1.15.x (see
[`src/vendor/VENDOR.md`](src/vendor/VENDOR.md)).

PX4 firmware and the micro-ROS / XRCE-DDS agents are large third-party trees excluded from this repo's history — clone them separately alongside this repo to build the full flight stack.
