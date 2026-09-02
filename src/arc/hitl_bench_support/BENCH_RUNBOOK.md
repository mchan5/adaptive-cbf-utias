# Shadow-HITL bench — full run procedure

Real, **props-off** Pixhawk's motors spin tracking a Gazebo SITL flight.
Three machines/targets:

| target | role | runs |
|---|---|---|
| **workstation** ("this") | SITL physics + CBF-QP + echo bridge | PX4 SITL, gz_x500, `MicroXRCEAgent` (UDP, ns `uav_0`), `autopilot_sv_cbf_rate_node`, `hitl_rate_echo_node`, (optional) the GUI |
| **companion** ("the drone") | serial link to the FMU | `MicroXRCEAgent` (serial, ns `uav_bench`) |
| **Pixhawk** | real arming / mixer / ESCs / motors | firmware + params only |

Workstation and companion must share one LAN and the **same `ROS_DOMAIN_ID`**.
The echo node forwards `/uav_0/fmu/in/{vehicle_rates_setpoint,offboard_control_mode}`
(sim) → `/uav_bench/fmu/in/...` (real board).

---

## 0. Safety — every session, no exceptions

- **Propellers removed.** Airframe bolted / strapped to the bench.
- RC transmitter on, **kill switch mapped and physically tested** (arm on the
  bench with props off, hit kill, confirm instant motor stop) before step 6.
- Battery current-limited or on a bench supply if you have one.
- Nobody's hands near the motors once armed.
- This node never arms. **You** arm from the RC and you kill from the RC.

---

## 1. Pixhawk one-time setup (QGroundControl + MAVLink shell)

Do this once per board / after a firmware change. See
`px4_config/README.md` for the rationale on every parameter.

1. **Firmware** — flash the version this repo targets (code verified against
   PX4 v1.17.0 in SITL; v1.15.4 is what SITL runs here — pick one and know
   which). Different firmware ⇒ re-check every "confirmed" item.
2. **Params** — QGC → Vehicle Setup → Parameters → Tools → *Load from file* →
   `px4_config/hardware_bringup.params`. Then **read every value back** and
   set the ENUM params (GF_ACTION=RTL, EKF2_HGT_REF, COM_OBL_RC_ACT, kill-switch
   channel, …) from QGC's dropdowns — do not guess enum integers.
3. **Rate-loop tuning** — autotune or manual, *before* any offboard test. The
   CBF node only commands body rates; it has zero authority over how well the
   inner rate loop tracks them.
4. **uXRCE-DDS client → serial, namespaced `uav_bench`.** In the MAVLink shell
   (QGC → Analyze → MAVLink Console), pick the wired port (example: TELEM2):

   ```
   param set UXRCE_DDS_CFG 0          # disable the auto-started (un-namespaced) client
   param set SER_TEL2_BAUD 921600
   param set UXRCE_DDS_DOM_ID <your ROS_DOMAIN_ID>
   param save
   ```

   Then put the namespaced client in the SD card's `etc/extras.txt`
   (create it if absent) so it starts every boot:

   ```
   uxrce_dds_client stop
   uxrce_dds_client start -t serial -d /dev/ttyS3 -b 921600 -n uav_bench
   ```

   `/dev/ttyS3` = TELEM2 on most FMUv5/v6; confirm with `ls /dev/tty*` in the
   shell. Reboot the board.

   *Alternative, no SD edit:* leave `UXRCE_DDS_CFG` on TELEM2 (auto-start,
   root namespace `/fmu/...`) and instead launch the echo node with
   `bench_prefix:=fmu` — but keeping `-n uav_bench` matches the design and
   this doc.

---

## 2. Network / domain check (both machines)

On **workstation** and **companion**, in every shell:

```bash
export ROS_DOMAIN_ID=<pick one, same on both>      # e.g. 42
source /opt/ros/jazzy/setup.bash
```

Verify they see each other before wiring in the FMU: run
`ros2 topic list` on one while `ros2 topic pub /disc std_msgs/msg/String "{}"`
runs on the other — `/disc` should appear. If not: same subnet? multicast
allowed? firewall? (fallback: `export ROS_LOCALHOST_ONLY=0` and/or a
`FASTRTPS_DEFAULT_PROFILES_FILE` peers list.)

---

## 3. Companion computer ("the drone")

Pixhawk wired to the companion's UART (or USB) at 921600.

```bash
export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash
source ~/<your_ws>/install/setup.bash          # needs px4_msgs

# serial agent for the REAL board — matches etc/extras.txt device/baud
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600
```

`/dev/ttyUSB0` = whatever the FMU enumerates as on the companion
(`/dev/ttyTHS1` on Jetson UART, `/dev/ttyACM0` for native USB, …).

In a second companion shell, confirm the board is actually bridged and on the
topic version the code expects:

```bash
export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash && source ~/<your_ws>/install/setup.bash
ros2 topic list | grep uav_bench                       # /uav_bench/fmu/out/... present
ros2 topic echo /uav_bench/fmu/out/vehicle_status_v1 --once   # one message
# repo's version check (expects v1 live, v4 absent):
$ARC_ROOT/config/px4/verify_vehicle_status_topic.sh
```

If `verify_vehicle_status_topic.sh` says `_v4` is live instead of `_v1`, stop —
`single_vehicle_cbf_rate_client.cpp` and `operator_arm_node.py` need patching
first (the script prints exactly where).

---

## 4. Workstation ("this")

```bash
cd "$ARC_ROOT"
export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash
# libtorch is picked up by scripts/build.sh; no venv activation needed
source "install_$ROS_DISTRO/setup.bash"
export PX4_SRC_DIR=/home/matt/PX4-Autopilot
export MICRO_XRCE_AGENT_SRC_DIR=/home/matt/Micro-XRCE-DDS-Agent

# one-time (or after code changes)
colcon build --packages-select single_vehicle_cbf_rate_arc hitl_bench_support \
  --allow-overriding single_vehicle_cbf_rate_arc --cmake-args -DBUILD_TESTING=ON
```

### 4a. Smoke-test the echo node first (no board involved)

```bash
colcon test --packages-select hitl_bench_support --event-handlers console_direct+
```

Must be `3 passed`. Do this every session — it's the cheapest check that
re-stamp / clamp / gate / deadman still work.

### 4b. Bring up SITL + the CBF node

Terminal W1:

```bash
ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py \
  launch_rviz:=false arm_mode:=auto obstacle_scene:=one
```

This starts PX4 SITL, gz_x500, the UDP `MicroXRCEAgent` (ns `uav_0`), the CBF
node, and `ground_station_stub_node` (auto-arms SITL ~5 s in, flies it to the
goal). Confirm rates are streaming:

```bash
ros2 topic hz /uav_0/fmu/in/vehicle_rates_setpoint      # ~100 Hz
```

*(Or use the GUI instead: `ros2 run arc_experiment_gui dashboard --ros-args -p
uav_prefix:=uav_0`, regime = `SITL (gz_x500)`, connect. Same result.)*

### 4c. Start the echo bridge

Terminal W2:

```bash
cd "$ARC_ROOT"
export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash && source "install_$ROS_DISTRO/setup.bash"

ros2 launch hitl_bench_support hitl_bench_launch.py \
  sim_prefix:=uav_0 bench_prefix:=uav_bench \
  thrust_clamp_norm:=0.15 deadman_timeout_sec:=0.1 require_bench_enable:=true
```

It logs `WAITING for bench_enable rising edge`. Nothing is forwarded yet.

Confirm it can see both ends:

```bash
ros2 topic hz /uav_0/fmu/in/vehicle_rates_setpoint        # source, ~100 Hz
ros2 node info /hitl_rate_echo_node                        # subs uav_0, pubs uav_bench
```

---

## 5. Arm the real board (props off) and release the gate

1. RC: put the board in **OFFBOARD**. It should hold (the echo node is already
   sending the OFFBOARD heartbeat + a zero-rate hold because the gate is
   closed — the board sees a valid offboard stream).
2. RC: **ARM**. Motors idle. Hand on the kill switch.
3. Workstation, terminal W3 — open the gate:

   ```bash
   ros2 topic pub -1 /uav_bench/bench_enable std_msgs/msg/Bool "{data: true}"
   ```

   The echo node logs `bench_enable rising edge -- forwarding sim rates`. Motor
   speeds now track the simulated vehicle's body-rate + (clamped) thrust
   demand as SITL flies its trajectory.
4. To stop forwarding at any time: publish `{data: false}`, or hit the RC kill
   switch, or Ctrl-C W2. Any of the three; the deadman also kicks in ~100 ms
   after W1 dies.

---

## 6. What to measure

```bash
# does 100 Hz actually survive the serial DDS link?
ros2 topic hz /uav_bench/fmu/in/vehicle_rates_setpoint

# sim -> board latency: compare header stamps / eyeball motor response vs gz
ros2 topic delay /uav_bench/fmu/in/vehicle_rates_setpoint   # if a source stamp is usable

# with a load cell / current clamp on one arm, sweep the sim through hover +
# aggressive maneuvers and re-derive the params still carrying gz_x500 values
# in params_single_vehicle_cbf_rate_arc_hardware.yaml:
#   vehicle_thrust_scaling, vehicle_idle_thrust, qp_thrust_min, vehicle_mass
```

---

## 7. Shutdown order

1. RC **kill**, then **disarm**.
2. W3: `ros2 topic pub -1 /uav_bench/bench_enable std_msgs/msg/Bool "{data: false}"`.
3. Ctrl-C W2 (echo), then W1 (SITL — or GUI *disconnect*, which also sweeps
   `px4` / `gz sim` / `MicroXRCEAgent`).
4. Ctrl-C the companion's serial `MicroXRCEAgent`.
5. Battery off. Props stay off until the airframe is off the bench.
