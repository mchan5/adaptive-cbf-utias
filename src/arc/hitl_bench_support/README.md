# hitl_bench_support

Shadow-HITL bench. Forwards the **SITL** vehicle's body-rate commands to a
**real, props-off Pixhawk** over the real serial uXRCE-DDS link, so the real
motors spin with the simulated flight.

It is a **bench test, not a flight test**. It closes the electrical/firmware
half of the loop and nothing more:

| Exercised for real | Still simulated |
|---|---|
| serial DDS bridge at 100 Hz (921600 baud) | vehicle physics (gz_x500) |
| firmware params, arming + OFFBOARD state machine | EKF / sensors |
| mixer → ESC → motor chain | thrust → motion (open loop) |

Why not true PX4 HITL: PX4 v1.15.4's `gz_bridge` has no HIL path — HITL goes
through `simulator_mavlink` (Gazebo Classic `iris`), a different airframe
that would invalidate the `gz_x500`-calibrated `vehicle_mass` /
`vehicle_thrust_scaling` / `qp_thrust_min` and every SITL result.

## Topology

```
gz_x500 physics ──▶ PX4 SITL (uav_0, MicroXRCEAgent udp4:8888)
     │ odom
     ▼
autopilot_sv_cbf_rate_node  (CBF-QP + PENN/GAT)
     │ vehicle_rates_setpoint + offboard_control_mode @100 Hz
     ├────────────▶ SITL PX4  (flies the sim, unchanged)
     │
     └──▶ hitl_rate_echo_node   ── re-stamp · clamp · gate · deadman ──▶
              MicroXRCEAgent serial /dev/ttyTHS1 @921600 (uav_bench)
              REAL Pixhawk (armed, PROPS OFF, strapped down)  ──▶  motors
```

The sim keeps namespace `uav_0` so GUI / launch parity is untouched. The real
board is brought up on a **separate** namespace `uav_bench` (its
`uxrce_dds_client -n uav_bench`), and both agents share one ROS domain.

## Safety contract (enforced by the node)

- **Re-stamp** `timestamp` with this machine's wall clock. SITL runs
  lockstepped sim-time; the real board rejects setpoints it thinks are
  stale. Passing the sim stamp through is the likeliest failure mode.
- **Thrust clamp** — normalised thrust magnitude capped at
  `thrust_clamp_norm` (default 0.15).
- **Explicit gate** — nothing is forwarded until a rising edge on
  `/<bench_prefix>/bench_enable` (`std_msgs/Bool`). Falling edge stops it.
- **Deadman** — if the sim stream is quiet for `deadman_timeout_sec`
  (default 0.1 s), publish a zero-rate / idle-thrust hold and keep the
  OFFBOARD heartbeat alive so the board stays level.

Operator responsibilities, every time: **props removed**, airframe strapped
down, RC kill switch mapped and tested *before* the first arm. This node
never arms — the operator arms from the RC.

## Offline smoke test (no board, no SITL)

```
colcon test --packages-select hitl_bench_support --event-handlers console_direct+
```

`test/test_rate_echo_smoke.py` spins the echo node against a harness node in
one process and checks the four contract points: the gate blocks until
`bench_enable`, the forwarded `timestamp` is wall-clock not the sim stamp, a
full-throttle demand is clamped to `thrust_clamp_norm`, and stopping the sim
stream produces a zero-rate hold. Run this before every bench session.

## Run (with hardware)

Full three-target procedure (workstation + companion + Pixhawk), including the
one-time board setup and the arm sequence: **[BENCH_RUNBOOK.md](BENCH_RUNBOOK.md)**.
Short version:

```
# 1. SITL (unchanged)
ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py launch_rviz:=false

# 2. Real board's serial agent, on the bench namespace
MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600
./px4_config/verify_vehicle_status_topic.sh          # confirm the bridge

# 3. This node
ros2 launch hitl_bench_support hitl_bench_launch.py

# 4. RC: OFFBOARD + arm.  Then release the gate:
ros2 topic pub -1 /uav_bench/bench_enable std_msgs/msg/Bool "{data: true}"
```

## What to measure

- Achieved `/uav_bench/fmu/in/vehicle_rates_setpoint` rate over serial
  (`ros2 topic hz`) — does it hold 100 Hz at 921600 baud? This is an
  unvalidated assumption of the whole hardware plan.
- Sim-vs-board latency.
- With a load cell / current probe: re-derive the `TODO(hardware)`
  thrust-curve params (`vehicle_thrust_scaling`, `vehicle_idle_thrust`,
  `qp_thrust_min`, `vehicle_mass`) still carrying Gazebo x500 values in
  `params_single_vehicle_cbf_rate_arc_hardware.yaml`.
