#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PointStamped
from px4_msgs.msg import VehicleLocalPosition, VehicleCommand, VehicleStatus, TrajectorySetpoint, OffboardControlMode

NAN = float('nan')

def ned_to_enu(v: np.ndarray) -> np.ndarray:
    """[x_N, y_E, z_D] → [x_E, y_N, z_U]"""
    return np.array([v[1], v[0], -v[2]])


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """[x_E, y_N, z_U] → [x_N, y_E, z_D]"""
    return np.array([v[1], v[0], -v[2]])


class BoundaryCBFNode(Node):

    ARMING_STATE_ARMED = 2

    def __init__(self):
        super().__init__('boundary_cbf_node')

        self.declare_parameter('boundary_x_min', -3.0)
        self.declare_parameter('boundary_x_max',  3.0)
        self.declare_parameter('boundary_y_min', -3.0)
        self.declare_parameter('boundary_y_max',  3.0)
        self.declare_parameter('boundary_z_min',  0.5)
        self.declare_parameter('boundary_z_max',  2.5)

        self.declare_parameter('waypoint_x', 0.0)
        self.declare_parameter('waypoint_y', 0.0)
        self.declare_parameter('waypoint_z', 1.5)

        self.declare_parameter('cbf_alpha', 0.5)
        self.declare_parameter('k_p',       0.5)
        self.declare_parameter('v_max',     0.5)
        self.declare_parameter('yaw_deg',   0.0)

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos)
        self._command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', pub_qos)
        
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_position_callback, sub_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v4', self.vehicle_status_callback, sub_qos)

        self.create_subscription(PointStamped, '/arc/boundary_min', self._arc_min_cb, 10)
        self.create_subscription(PointStamped, '/arc/boundary_max', self._arc_max_cb, 10)
        self.create_subscription(PointStamped, '/arc/waypoint',     self._arc_wp_cb,  10)

        self._pos_ned = np.zeros(3)
        self._pos_valid = False
        self._arming_state = -1
        self._tick_count = 0
        self._origin_ned = None
        self._last_pos_time = None
        self._at_waypoint = False  # True when within hover threshold — switches to position hold

        # Live boundary/waypoint from boundary_manager_node (ENU, relative to arm origin).
        # Falls back to parameters if no message received yet.
        self._arc_min = None
        self._arc_max = None
        self._arc_wp  = None

        self._yaw = math.radians(self.get_parameter('yaw_deg').value)

        self.create_timer(0.05, self._tick)
        self.get_logger().info('Boundary CBF Node has been started.')

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.get_logger().info(
            f'pos cb: xy_valid={msg.xy_valid} z_valid={msg.z_valid} x={msg.x:.2f} y={msg.y:.2f} z={msg.z:.2f}',
            throttle_duration_sec=2.0)
        self._pos_ned = np.array([msg.x, msg.y, msg.z])
        self._pos_valid = bool(msg.xy_valid and msg.z_valid)
        if self._pos_valid:
            self._last_pos_time = self.get_clock().now()

    def vehicle_status_callback(self, msg: VehicleStatus):
        self._arming_state = msg.arming_state
        self.get_logger().info(
            f'status cb: arming_state={msg.arming_state} nav_state={msg.nav_state}',
            throttle_duration_sec=2.0)

    def _arc_min_cb(self, msg: PointStamped):
        self._arc_min = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _arc_max_cb(self, msg: PointStamped):
        self._arc_max = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _arc_wp_cb(self, msg: PointStamped):
        self._arc_wp = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _tick(self):
        self._publish_heartbeat()
        self._tick_count += 1

        if self._tick_count < 50:
            return

        if self._tick_count == 50:
            self._engage_offboard()
            self._arm()
            self.get_logger().info('Sent 50 heartbeats, engaging offboard + arming...')
            return

        # Retry arm/offboard every 5 s if still not armed
        if self._arming_state != self.ARMING_STATE_ARMED and self._tick_count % 100 == 0:
            self.get_logger().warn(
                f'Not armed yet (state={self._arming_state}), retrying offboard+arm...')
            self._engage_offboard()
            self._arm()
            return

        if not self._pos_valid:
            self.get_logger().warn('Local position not valid yet, cannot publish setpoint.')
            self._publish_velocity(np.zeros(3))
            return

        # Safety: zero velocity if position feed goes stale (e.g. companion computer link drops)
        if self._last_pos_time is not None:
            age = (self.get_clock().now() - self._last_pos_time).nanoseconds * 1e-9
            if age > 2.0:
                self.get_logger().warn(
                    f'Position feed stale ({age:.1f}s) — zeroing velocity.',
                    throttle_duration_sec=1.0)
                self._publish_velocity(np.zeros(3))
                return

        if self._arming_state == self.ARMING_STATE_ARMED and self._origin_ned is None:
            self._origin_ned = self._pos_ned.copy()
            enu = ned_to_enu(self._origin_ned)
            self.get_logger().info(f'Captured origin at ENU {enu.round(3)}')

        if self._origin_ned is None:
            self._publish_velocity(np.zeros(3))
            return

        pos_enu = ned_to_enu(self._pos_ned - self._origin_ned)

        if self._arc_wp is not None:
            wp = self._arc_wp
        else:
            wp = np.array([
                self.get_parameter('waypoint_x').get_parameter_value().double_value,
                self.get_parameter('waypoint_y').get_parameter_value().double_value,
                self.get_parameter ('waypoint_z').get_parameter_value().double_value,
            ])

        alpha = self.get_parameter('cbf_alpha').get_parameter_value().double_value
        k_p = self.get_parameter('k_p').get_parameter_value().double_value
        v_max = self.get_parameter('v_max').get_parameter_value().double_value

        if self._arc_min is not None and self._arc_max is not None:
            mn, mx = self._arc_min, self._arc_max
            boundary = (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
        else:
            boundary = self._boundary()

        x_min, x_max, y_min, y_max, z_min, z_max = boundary
        clamped_wp = np.array([
            np.clip(wp[0], x_min, x_max),
            np.clip(wp[1], y_min, y_max),
            np.clip(wp[2], z_min, z_max),
        ])
        if not np.allclose(clamped_wp, wp, atol=0.01):
            self.get_logger().warn(
                f'Waypoint {wp.round(2)} outside boundary — clamping to {clamped_wp.round(2)}',
                throttle_duration_sec=2.0)
        wp = clamped_wp

        err = wp - pos_enu
        dist = float(np.linalg.norm(err))

        HOVER_ENTER = 0.3  # metres — enter position hold within this radius
        HOVER_EXIT = 0.5   # metres — must drift this far out before resuming velocity tracking

        if self._at_waypoint:
            self._at_waypoint = dist < HOVER_EXIT
        else:
            self._at_waypoint = dist < HOVER_ENTER

        if dist > 0.1 and not self._at_waypoint:
            self._yaw = math.atan2(err[0], err[1])

        if self._at_waypoint:
            # Position hold: tell PX4 to stay at the waypoint in NED absolute coords
            wp_ned = enu_to_ned(wp) + self._origin_ned
            self._publish_position(wp_ned)
        else:
            v_nominal = k_p * err
            speed = np.linalg.norm(v_nominal)
            if speed > v_max:
                v_nominal = v_nominal / speed * v_max
            v_safe = self._cbf_clip(pos_enu, v_nominal, boundary, alpha)
            self._publish_velocity(enu_to_ned(v_safe))

        if self._tick_count % 20 == 0:
            h_min = min(self._barrier_values(pos_enu, boundary))
            self.get_logger().info(
                f'pos={pos_enu.round(2)}  wp_err={dist:.2f}m  '
                f'h_min={h_min:.2f}  holding={self._at_waypoint}'
            )

    def _cbf_clip(self, pos, v_nominal, boundary, alpha):
        x, y, z = pos
        x_min, x_max, y_min, y_max, z_min, z_max = boundary
        vx = float(np.clip(v_nominal[0], -alpha * (x - x_min), -alpha * (x - x_max)))
        vy = float(np.clip(v_nominal[1], -alpha * (y - y_min), -alpha * (y - y_max)))
        vz = float(np.clip(v_nominal[2], -alpha * (z - z_min), -alpha * (z - z_max)))
        return np.array([vx, vy, vz])

    def _barrier_values(self, pos, boundary):
        x, y, z = pos
        x_min, x_max, y_min, y_max, z_min, z_max = boundary
        return [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y,
            z - z_min,
            z_max - z
        ]

    def _boundary(self):
        return (
            self.get_parameter('boundary_x_min').get_parameter_value().double_value,
            self.get_parameter('boundary_x_max').get_parameter_value().double_value,
            self.get_parameter('boundary_y_min').get_parameter_value().double_value,
            self.get_parameter('boundary_y_max').get_parameter_value().double_value,
            self.get_parameter('boundary_z_min').get_parameter_value().double_value,
            self.get_parameter('boundary_z_max').get_parameter_value().double_value
        )

    def _publish_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = self._at_waypoint
        msg.velocity = not self._at_waypoint
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self._ts()
        self._mode_pub.publish(msg)

    def _publish_position(self, pos_ned):
        msg = TrajectorySetpoint()
        msg.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = self._yaw
        msg.timestamp = self._ts()
        self._setpoint_pub.publish(msg)

    def _publish_velocity(self, vel_ned):
        msg = TrajectorySetpoint()
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = self._yaw
        msg.timestamp = self._ts()
        self._setpoint_pub.publish(msg)

    def _arm(self):
        self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                              param1=1.0, param2=21196.0)
        self.get_logger().info('Sent ARM command (force).')

    def _engage_offboard(self):
        self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Sent OFFBOARD command.')

    def _vehicle_command(self, command: int, **kw):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(kw.get('param1', 0.0))
        msg.param2 = float(kw.get('param2', 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._ts()
        self._command_pub.publish(msg)

    def _ts(self):
        return int(self.get_clock().now().nanoseconds / 1000)

def main(args=None):
    rclpy.init(args=args)
    node = BoundaryCBFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
