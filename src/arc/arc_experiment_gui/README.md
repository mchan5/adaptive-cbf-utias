# arc_experiment_gui

Operator GUI for the adaptive-vs-fixed-gamma obstacle-avoidance campaign.
Launches, configures, runs, and logs trials against SITL and hardware
through one interface. See `../EXPERIMENT_GUI_PLAN_20260829.md` for the full
design.

**The GUI is never in the control path.** It only *reads* telemetry the
100 Hz CBF loop already produced and *writes* occasional setup commands
(`SetParameters`, a `goal_waypoint` publish, an `operator/arm_confirm`, a
`ros2 bag` / `ros2 launch` subprocess). If it freezes or dies mid-flight the
C++ node keeps solving the QP unchanged.

## Run

```
ros2 run arc_experiment_gui dashboard --ros-args -p uav_prefix:=uav_0
```

Needs a display (WSLg / X11). Requires the workspace overlay sourced so
`single_vehicle_cbf_rate_arc/msg/CbfDiagnostics` and `px4_msgs` resolve.

## Tabs

| Tab | What it does |
|---|---|
| **dashboard** | status banner (ARM / OFFBOARD / QP feasible / mode / γ / obstacle counts / goal), top-down obstacle view (detected + optional frozen-scene overlay, drone, goal, trail), min-obstacle-distance bar with amber@1 m / red@0.3 m alarm, state readouts, event log, odom-staleness watchdog, config-freeze (`git` hash + DIRTY) indicator |
| **launch** | bring the SITL/hardware stack up/down as a process-group subprocess; node-health row; `/arc/obstacles` duplicate-publisher detector; pre-flight checklist (CBF node, odom bridge, diagnostics streaming, odom fresh, one obstacle publisher, param service, disk free) with a READY aggregate; launch-log pane |
| **campaign** | create/load a `Plan` (`N scenes x M arms x K trials`, arm order shuffled per scene from a seed); `TrialRunner` state machine per trial (set arm params → fly to start → `ros2 bag record` → publish goal → wait `reached` / timeout / abort → operator confirms outcome → append `manifest.json` + `trials.csv`); progress matrix (scenes x arms, coloured by outcome); resume-safe (skips cells already in the manifest); `export bundle` zips the campaign dir |
| **lidar** | hardware bring-up aid: `/livox/lidar` publisher count, `lidar_obstacle_publisher_node` up/down, current cluster count, and a read-back of the clusterer's mount-offset / voxel / range parameters |

## Regimes

The `regime` selector (top bar) picks what `connect` brings up:

| regime | launch file | arming | obstacles |
|---|---|---|---|
| `synthetic` | `cbf_rate_arc_hardware_launch.py` | operator (RC / `operator/arm_confirm`) | frozen scene JSON |
| `LiDAR — live` | same, `obstacle_source:=lidar` | operator | live Livox clustering |
| `SITL (gz_x500)` | `cbf_rate_arc_sitl_test_launch.py` | `arm_mode:=auto` (stub auto-arms); GUI owns `goal_waypoint` via `publish_goal:=false` | frozen scene JSON |

SITL brings up PX4 SITL + gz_x500 + MicroXRCEAgent itself; on stop the
launcher sweeps those too (full-path `px4` match). `battery` reads `--` in
SITL (gz_x500 publishes no `battery_status`).

## Config

- `uav_prefix` ROS param (default `uav_0`) — namespace for all topics/services.
- `ARC_CBF_REPO` env — repo path for the config-freeze indicator
  (default `$ARC_ROOT/src/arc`).
- Packaged scene set —
  `single_vehicle_cbf_rate_arc/config/scenes/hw_identical_20260831`: 20 frozen
  scenes of **identical** 0.12 m obstacles at `z = 1.5` in a
  `(0, 0.5, 1.5) → (0, 5.5, 1.5)` corridor, matching the real floor cylinders.
  Because they are identical and the barrier is evaluated at flight altitude,
  a scene is the same geometry under `cbf_cylinder_barrier` either way. The
  retired mixed-radius `hw_campaign` set is kept only as provenance.
- `ARC_SCENE_DIR` env — overrides the packaged scene set. Point it at a
  frozen results dir (e.g.
  `$ARC_ROOT/research/penn_gat/results/hardware_scenes_final_20260831/scenes`,
  the source the packaged set was copied from) to fly that set in either
  regime without re-copying files into the package — this is what makes a
  SITL rehearsal and a hardware run fly the *same* scenes.
- `ARC_TRIAL_RECORD_CMD` env (optional, Phase 5) — command template with
  `{out}` run at each trial's bag-start and SIGINT'd at bag-stop, e.g.
  `ffmpeg -y -f x11grab -i :0 {out}/screen.mp4`.
- `MICRO_XRCE_AGENT_SRC_DIR` / `PX4_SRC_DIR` env — required for the
  `SITL (gz_x500)` regime only. `cbf_rate_arc_sitl_test_launch.py` defaults
  both to `$HOME/src/...`, which is almost never where they actually live;
  the GUI inherits its parent shell's environment unmodified (it does not
  set these itself), so export the real paths *before* starting the GUI,
  e.g. `export MICRO_XRCE_AGENT_SRC_DIR=$HOME/Micro-XRCE-DDS-Agent
  PX4_SRC_DIR=$HOME/PX4-Autopilot`. Left unset, `connect` in the SITL regime
  fails immediately with `FileNotFoundError` on the agent binary.

## Campaign artifacts

```
campaign_<ts>/
  plan.json          seed, scenes, per-scene shuffled arm order
  manifest.json      one object per completed trial
  trials.csv         same rows, flat, for analysis
  sceneNN_<arm>_trialK/   one ros2 bag per trial (raw)
```

Bag topic list (fixed): `cbf/diagnostics`, `state_estimator/local_position/odom`,
`/arc/obstacles`, `fmu/out/vehicle_status`, `fmu/in/vehicle_rates_setpoint`,
`goal_waypoint`, `fmu/out/battery_status`, `/tf`, `/tf_static`.

## Notes / limits

- SITL PX4 gz_x500 does not publish `battery_status`; that readout shows `--`
  in SITL, fine on hardware.
- `min_obstacle_dist` in `CbfDiagnostics` is `||pos - centre|| - radius`; it
  can go negative on a close pass without meaning a physical collision.
  Confirm the trial outcome in the dialog rather than trusting the
  auto-suggestion.
- Return-to-start between trials uses the fixed SITL corridor
  (`(0, 0.5, 1.5) -> (0, 5.5, 1.5)`); on hardware the operator repositions.
- A true LiDAR point-rate needs `ros2 topic hz` and is not shown here.
- Compact `w = QWidget(); w.setX(...)` widget-init style trips ament_flake8's
  E702; that is deliberate and repo-consistent.
