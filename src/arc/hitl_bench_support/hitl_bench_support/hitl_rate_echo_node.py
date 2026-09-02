#!/usr/bin/env python3
"""Shadow-HITL bench echo: forward the SITL vehicle's body-rate commands to a real, PROPS-OFF
Pixhawk so its motors spin with the simulated flight."""

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
        # <bench_prefix>/bench_enable.
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
