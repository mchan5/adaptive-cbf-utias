#!/usr/bin/env python3
"""Shadow-HITL bench echo: forward the SITL vehicle's body-rate commands to a
real, PROPS-OFF Pixhawk so its motors spin with the simulated flight.

  gz_x500 physics --> PX4 SITL (uav_0, MicroXRCEAgent udp4:8888)
       |  odom
       v
  autopilot_sv_cbf_rate_node  (CBF-QP + PENN/GAT)
       |  vehicle_rates_setpoint + offboard_control_mode @100 Hz
       +-----------------> SITL PX4 (flies the sim)
       |
       +--> hitl_rate_echo_node  (this node)
                |  re-stamp, clamp, gate, deadman
                v  MicroXRCEAgent serial /dev/ttyTHS1 @921600 (uav_bench)
           REAL Pixhawk (armed, PROPS OFF, strapped down)
                v
           REAL motors spin

This is a BENCH TEST, not a flight test. It closes the electrical/firmware
half of the loop -- the real serial DDS link at 100 Hz, the real arming and
OFFBOARD state machine, the real mixer -> ESC -> motor chain -- but NOT the
physics: real thrust does not move the simulated vehicle. Fly nothing on
this node.

What it does per the safety contract:

  * re-stamp `timestamp` with THIS machine's wall clock. SITL PX4 runs on
    lockstepped sim-time; the real board runs wall-time and rejects
    OFFBOARD setpoints it considers stale. Passing the sim timestamp
    through is the most likely failure mode, so we never do.
  * clamp normalised thrust magnitude to `thrust_clamp_norm` (default
    0.15). Props are off; this is belt-and-braces against a commanded
    full-throttle spinning an unloaded motor to destruction.
  * gate on an explicit rising edge of `<bench_prefix>/bench_enable`
    (std_msgs/Bool). Nothing is forwarded until the operator confirms.
    A falling edge (or `require_bench_enable:=false` from the start) is
    handled too.
  * deadman: if the sim rate stream goes quiet for `deadman_timeout_sec`
    (default 0.1 s), publish a zero-rate / idle-thrust hold instead of the
    last command, and keep the OFFBOARD heartbeat alive so the board stays
    level rather than dropping mode.

Namespacing: the sim keeps `uav_0` (so the GUI / launch parity is
untouched); the real board is brought up on a SEPARATE namespace
(`uav_bench`) via `uxrce_dds_client start -n uav_bench` in its startup
script, and both agents run on the same ROS domain. Verify the bench board
is actually bridged with px4_config/verify_vehicle_status_topic.sh before
arming.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint


def _px4_qos():
    # Matches single_vehicle_cbf_rate_client.cpp's qos_px4 (the QoS it
    # publishes fmu/in/* with) and what the uXRCE-DDS agent expects.
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class HitlRateEchoNode(Node):
    def __init__(self):
        super().__init__('hitl_rate_echo_node')

        self.declare_parameter('sim_prefix', 'uav_0')
        self.declare_parameter('bench_prefix', 'uav_bench')
        self.declare_parameter('thrust_clamp_norm', 0.15)
        self.declare_parameter('deadman_timeout_sec', 0.1)
        self.declare_parameter('heartbeat_hz', 50.0)
        # When true (default) nothing is forwarded until a rising edge on
        # <bench_prefix>/bench_enable. Set false only for a hands-off rig
        # you have already validated.
        self.declare_parameter('require_bench_enable', True)

        self._sim = self.get_parameter('sim_prefix').value
        self._bench = self.get_parameter('bench_prefix').value
        self._clamp = abs(self.get_parameter('thrust_clamp_norm').value)
        self._deadman = self.get_parameter('deadman_timeout_sec').value
        self._enabled = not self.get_parameter('require_bench_enable').value

        qos = _px4_qos()
        self._rate_pub = self.create_publisher(
            VehicleRatesSetpoint, f'/{self._bench}/fmu/in/vehicle_rates_setpoint', qos)
        self._ocm_pub = self.create_publisher(
            OffboardControlMode, f'/{self._bench}/fmu/in/offboard_control_mode', qos)

        self.create_subscription(
            VehicleRatesSetpoint, f'/{self._sim}/fmu/in/vehicle_rates_setpoint',
            self._on_rate, qos)
        self.create_subscription(
            OffboardControlMode, f'/{self._sim}/fmu/in/offboard_control_mode',
            self._on_ocm, qos)
        self.create_subscription(
            Bool, f'/{self._bench}/bench_enable', self._on_enable, 10)

        self._last_rx = None            # monotonic time of last sim rate msg
        self._last_ocm = None           # last OffboardControlMode from sim
        self._prev_enable = False
        self._fwd_count = 0
        self._deadman_latched = False

        hb = max(1.0, self.get_parameter('heartbeat_hz').value)
        self.create_timer(1.0 / hb, self._heartbeat)

        gate = ('WAITING for bench_enable rising edge' if not self._enabled
                else 'ENABLED at start')
        self.get_logger().warn(
            f'hitl_rate_echo_node up: {self._sim} -> {self._bench}, '
            f'thrust clamp |{self._clamp:.2f}|, deadman {self._deadman * 1000:.0f} ms, '
            f'{gate}. PROPS OFF. Airframe strapped down. RC kill switch armed.')

    # gate
    def _on_enable(self, msg: Bool):
        if msg.data and not self._prev_enable:
            self._enabled = True
            self.get_logger().warn('bench_enable rising edge -- forwarding sim rates')
        elif not msg.data and self._prev_enable:
            self._enabled = False
            self.get_logger().warn('bench_enable falling edge -- forwarding STOPPED')
        self._prev_enable = msg.data

    # sim inputs
    def _on_ocm(self, msg: OffboardControlMode):
        self._last_ocm = msg

    def _on_rate(self, msg: VehicleRatesSetpoint):
        self._last_rx = self._now()
        if self._deadman_latched:
            self._deadman_latched = False
            self.get_logger().info('sim rate stream resumed')
        if not self._enabled:
            return
        self._rate_pub.publish(self._restamp_and_clamp(msg))
        self._fwd_count += 1
        if self._fwd_count % 500 == 0:
            self.get_logger().info(f'forwarded {self._fwd_count} rate setpoints')

    # outputs
    def _heartbeat(self):
        """Keep the bench board's OFFBOARD heartbeat alive, and publish a
        safe hold whenever the sim stream is stale or forwarding is off."""
        if not self._enabled:
            return

        ocm = OffboardControlMode()
        ocm.timestamp = self._stamp_us()
        if self._last_ocm is not None:
            ocm.position = self._last_ocm.position
            ocm.velocity = self._last_ocm.velocity
            ocm.acceleration = self._last_ocm.acceleration
            ocm.attitude = self._last_ocm.attitude
            ocm.body_rate = self._last_ocm.body_rate
            ocm.thrust_and_torque = self._last_ocm.thrust_and_torque
            ocm.direct_actuator = self._last_ocm.direct_actuator
        else:
            ocm.body_rate = True
        self._ocm_pub.publish(ocm)

        age = None if self._last_rx is None else (self._now() - self._last_rx)
        stale = age is None or age > self._deadman
        if stale:
            if not self._deadman_latched:
                self._deadman_latched = True
                self.get_logger().warn(
                    'DEADMAN: sim rate stream stale -- publishing zero-rate / idle-thrust hold')
            self._rate_pub.publish(self._safe_hold())

    # helpers
    def _restamp_and_clamp(self, src: VehicleRatesSetpoint) -> VehicleRatesSetpoint:
        out = VehicleRatesSetpoint()
        out.timestamp = self._stamp_us()
        out.roll = src.roll
        out.pitch = src.pitch
        out.yaw = src.yaw
        # thrust_body is normalised [-1, 1]; for a multicopter [2] is the
        # negative throttle demand. Clamp every component's magnitude.
        out.thrust_body = [
            max(-self._clamp, min(self._clamp, float(v))) for v in src.thrust_body
        ]
        out.reset_integral = src.reset_integral
        return out

    def _safe_hold(self) -> VehicleRatesSetpoint:
        out = VehicleRatesSetpoint()
        out.timestamp = self._stamp_us()
        out.roll = out.pitch = out.yaw = 0.0
        out.thrust_body = [0.0, 0.0, 0.0]
        return out

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp_us(self):
        # PX4 wants time-since-boot microseconds on THIS machine's clock, not
        # the sim's. get_clock() here is the wall/system clock (no use_sim_time).
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None):
    rclpy.init(args=args)
    node = HitlRateEchoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
