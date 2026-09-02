# PX4 hardware bring-up config

This directory is not a ROS2 package -- it's the PX4-side (firmware parameter)
half of hardware bring-up, sitting next to the ROS2-side packages
(`hardware_test_support`, `lidar_obstacle_publisher`) in this repo. None of
this runs on the companion computer; it's loaded onto the Pixhawk itself via
QGroundControl or the MAVLink shell.

**Every value here was sourced from PX4's own docs (docs.px4.io, fetched
2026-08-18) or PX4-Autopilot's issue tracker, not written from memory.**
Even so, treat this file as a first draft, not a trusted final config:
PX4 parameter names, bit layouts, and enum values genuinely do shift between
versions, and the only way to be certain of what a given parameter means
*on your firmware build* is to check its live description/tooltip in
QGroundControl once connected to the real Pixhawk. Confidence level is
called out per parameter below.

## Files

- `hardware_bringup.params` -- QGroundControl-loadable parameter file
  (Vehicle Setup > Parameters > Tools > Load from file). Best-effort correct
  format (tab-separated, `MAV_PARAM_TYPE` type codes) but **never verified
  against a real QGC load** -- there's no Pixhawk in this dev environment to
  test it against. If it fails to load, use the `param set` commands below
  instead; those don't depend on file-format correctness.
- `verify_vehicle_status_topic.sh` -- run this once MicroXRCEAgent is up
  against the real Pixhawk, before trusting anything downstream. See
  "vehicle_status_v1 vs v4" below for why this specific check matters.

## Sequencing

1. Flash/confirm PX4 firmware version first (this repo's code was last
   verified against PX4-Autopilot v1.15.4 in SITL, 2026-09-01 -- see
   `single_vehicle_cbf_rate_client.cpp` and `ground_station_stub_node.py`'s
   comments, and "vehicle_status_v1 vs v4 vs unversioned" below). A
   different firmware version on the real Pixhawk changes which of the
   "confirmed" items below are still accurate.
2. Load `hardware_bringup.params` (or apply the `param set` commands one at
   a time via the MAVLink shell), then **read every value back in QGC and
   compare against what you intended** -- don't assume a load succeeded
   silently.
3. Run `verify_vehicle_status_topic.sh`.
4. Do the rate-controller tuning pass (autotune or manual) -- see bottom of
   this file. Do this BEFORE the first offboard/CBF flight; the CBF node
   only commands body rates, it has no authority over how well the inner
   rate loop tracks them.
5. Bench test with props off, then work through the staged flight
   progression already described in the hardware README/conversation
   history (tethered hover -> free hover -> obstacles).

## Parameter groups

### External vision (mocap) position source -- CONFIRMED parameter names, verify exact enum/bit values on your firmware

Feeds `natnet_ros2`'s pose data into PX4's own EKF2, not just our
`state_estimator/local_position/odom` topic. This is a separate integration
from what `hardware_test_support/mocap_odom_bridge_node.py` does -- that
bridge feeds *our* CBF node directly; these params make *PX4's own*
estimator (and therefore its own failsafes/arming checks) trust mocap too.
Whether you wire this up depends on whether you want PX4's own EKF using
vision, or only our CBF node's private copy -- doing only the latter means
PX4's internal position estimate stays GPS/baro-only and may not agree with
what the CBF node is flying on, which is its own risk. Recommended: wire
both.

- `EKF2_EV_CTRL` -- bitmask selecting which vision measurements to fuse
  (horizontal position / vertical position / velocity / yaw -- 4 bits per
  docs.px4.io). Set to `15` (all four bits on) if you want full mocap
  fusion including yaw (recommended indoors, since magnetometer fusion is
  often unreliable near motors/metal structures). **Confidence: the meaning
  of "all bits on" is solid; the exact bit-to-feature mapping wasn't
  independently confirmed -- if you need less than all four, check the
  parameter's live description in QGC first.**
- `EKF2_HGT_REF` -- set to use Vision as the primary height reference
  instead of baro/GPS. **Not confirmed: the exact enum integer for "Vision"
  on your firmware version.** Set this via QGC's parameter dropdown (it'll
  show you the enum labels directly) rather than trusting a numeric value
  in the `.params` file blindly -- left as a placeholder there.
- `EKF2_EV_DELAY` -- vision pipeline latency compensation, milliseconds.
  **Cannot be a sourced value** -- this depends on your actual network +
  natnet_ros2 + ROS2 DDS latency, which varies by deployment. Placeholder
  of 30ms in the params file; measure your actual end-to-end latency and
  correct this before flight.
- `EKF2_EV_POS_X/Y/Z` -- offset (meters, body frame) between the mocap
  rigid body's tracked origin and the vehicle's true body/IMU origin.
  **Must be physically measured** on your vehicle; defaults to 0 (i.e.
  "no offset") which is very unlikely to be exactly right once mocap
  markers are mounted.
- `EKF2_GPS_CTRL = 0`, `EKF2_BARO_CTRL = 0` -- disable GPS and barometer
  fusion for indoor-only operation, so they don't fight the vision
  estimate. **This param file is an indoor-only config** -- if the same
  vehicle later flies outdoors, these need to flip back on (keep a
  separate outdoor `.params` file rather than editing this one in place).
- GPS-required arming checks: could not find a current, confirmed
  parameter name for "allow arming without GPS" in the versions searched
  (older PX4 had `COM_ARM_WO_GPS`; may be renamed/removed). Once
  `EKF2_GPS_CTRL=0` and vision fusion is healthy, PX4's arming check
  should key off EKF2 estimate validity rather than GPS presence directly
  -- if arming is still blocked citing position/GPS, check the `COM_ARM_*`
  parameter group and the live pre-arm check list in QGC on your actual
  firmware.

### Geofence -- CONFIRMED parameter names, placeholder distances

This is the PX4-native geofence -- the one that matters, since it survives
a companion-computer or ROS2 crash. `hardware_test_support/geofence_monitor_node.py`
is a companion-computer-side backup to this, not a replacement.

- `GF_ACTION = 3` (Return/RTL) -- matches the same choice already made in
  `geofence_monitor_node.py` (RTL, not a raw disarm, since disarming
  mid-flight drops the vehicle).
- `GF_MAX_HOR_DIST`, `GF_MAX_VER_DIST` -- **placeholders (5.0m, 3.0m)**
  matching `geofence_monitor_node.py`'s own default bounds, purely for
  consistency between the two layers. Must match your actual test volume's
  real dimensions, which I don't know.
- `GF_SOURCE` -- exists per PX4's geofence implementation but its exact
  current semantics (GPS vs. any global-position source) weren't confirmed
  from the docs fetched here. Verify in QGC before trusting the geofence
  is actually evaluating against the vision-derived position, not a GPS
  position that won't exist indoors.

### Battery failsafe -- CONFIRMED parameter names, values are generic starting points

- `BAT_LOW_THR`, `BAT_CRIT_THR`, `BAT_EMERGEN_THR` -- percentage
  thresholds for warning / return / land. Left at conservative starting
  points (30% / 20% / 10%) in the params file -- these are generic, not
  derived from your actual battery's discharge curve. Confirm against your
  battery's real capacity and acceptable discharge depth.
- `COM_LOW_BAT_ACT` -- action for low battery (set to Return-then-Land).
- `COM_ARM_BAT_MIN` -- minimum percentage to allow arming at all.

### Offboard-loss failsafe -- CONFIRMED parameter names, likely more relevant here than plain RC loss

Given this system's actual failure mode is more likely "companion computer
or the CBF node crashes/stalls" than "RC transmitter loses signal", these
matter as much as or more than the generic RC-loss params:

- `COM_OF_LOSS_T` -- timeout (seconds) after which missing offboard
  setpoints/OffboardControlMode heartbeats trigger the offboard-loss
  failsafe. Placeholder of 0.5s -- the CBF node publishes at 100Hz, so this
  should trip fast relative to a real stall, but hasn't been tuned against
  real dropout behavior.
- `COM_OBL_RC_ACT` -- action when offboard is lost AND RC is available
  (should be a pilot-recoverable mode, e.g. Position or Altitude, so the
  safety pilot can take over -- do not set this to something that requires
  GPS if flying indoors without GPS).
  `COM_OBL_ACT` -- action when offboard is lost with no RC available
  either (should default toward Land, not Return, if `GF_SOURCE`/position
  validity indoors is in question).

### RC loss / kill switch

- `NAV_RCL_ACT` -- action on RC transmitter signal loss.
- Kill switch itself is **not just a parameter value** -- it's assigned to
  a transmitter channel via QGroundControl's Vehicle Setup > Flight Modes
  UI (which then writes the underlying channel-mapping parameter for you).
  The likely parameter name is `RC_MAP_KILLSWITCH`, **not independently
  confirmed** in the sources checked here -- use the QGC UI flow, not a
  hand-written param line, to avoid getting this specific one wrong. This
  is the single most safety-critical item in this whole file; do not trust
  a guessed parameter name for it.

## Rate-controller tuning -- not something a params file can respons­ibly ship

`MC_ROLLRATE_P/I/D`, `MC_PITCHRATE_P/I/D`, `MC_YAWRATE_P/I/D` and friends
are not included here with numeric values. The CBF node commands body
rates directly, so PX4's inner rate loop tracking those rates well is
load-bearing for the whole system being stable -- and unlike everything
else in this file, there is no "reasonable generic starting point" for
these that's responsible to hand out sight-unseen for a specific real
airframe. Use PX4's built-in autotune (safer, cheaper than manual tuning
for a first pass) or manual step tuning, in normal manual/acro flight,
before ever engaging offboard/CBF control.

## vehicle_status_v1 vs v4 vs unversioned

The versioned suffix PX4's uxrce_dds_client publishes `vehicle_status` under
has now drifted twice: SITL PX4 v1.17.0 published `fmu/out/vehicle_status_v1`
(not `_v4` -- `sitl_test_support/ground_station_stub_node.py` had it
hardcoded to `_v4` until that was fixed to match). As of 2026-09-01, SITL
PX4 v1.15.4's `dds_topics.yaml` defines only the **unversioned**
`fmu/out/vehicle_status` -- both `_v1` and `_v4` have zero publishers on
that build (`ros2 topic info -v`). All five subscribers (the CBF node,
`ground_station_stub_node.py`, `operator_arm_node.py`, and the GUI's
`ros_node.py`/`launcher.py` bag topic list) were fixed to the unversioned
name on 2026-09-01.

This failure mode is silent and severe: when the subscribed topic has no
publisher, the subscriber's armed-state tracking never updates, so any
control/arm path gated on "am I armed/offboard" never fires even though the
vehicle is genuinely armed -- the CBF node computes feasible QP commands
every tick that never reach PX4. Don't assume the real Pixhawk's firmware
build publishes the same variant as whatever SITL last used; run
`verify_vehicle_status_topic.sh` once MicroXRCEAgent is bridging the real
vehicle, before flight -- it now checks all three variants.
