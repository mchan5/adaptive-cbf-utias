#!/usr/bin/env python3
"""Companion-computer-side geofence backup for real-hardware flights.

Watches state_estimator/local_position/odom and issues an RTL VehicleCommand
the moment position leaves a configured box. This has no SITL equivalent --
sim never needed a geofence -- so it's new, not a port of anything.

This is a BACKUP layer, not the primary one. The primary geofence must be
configured natively on the Pixhawk itself (PX4's GF_ACTION / GF_MAX_HOR_DIST
etc. params), because that enforcement keeps working even if the companion
computer, this node, or the ROS link between them dies -- which is exactly
the situation a geofence needs to survive. Do not fly relying on this node
alone.

Reacts with RTL (VEHICLE_CMD_NAV_RETURN_TO_LAUNCH), not a raw disarm --
disarming mid-flight drops the vehicle out of the sky. RTL is what PX4's own
GF_ACTION defaults to for a reason.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleCommand


class GeofenceMonitorNode(Node):
    def __init__(self):
        super().__init__('geofence_monitor_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('x_min', -5.0)
        self.declare_parameter('x_max', 5.0)
        self.declare_parameter('y_min', -5.0)
        self.declare_parameter('y_max', 5.0)
        self.declare_parameter('z_min', 0.0)
        self.declare_parameter('z_max', 3.0)

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self._vehicle_command_pub = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', cmd_qos)
        self.create_subscription(Odometry, 'state_estimator/local_position/odom', self._on_odom, sub_qos)

        self._breached = False

        bounds = (self.get_parameter('x_min').value, self.get_parameter('x_max').value,
                  self.get_parameter('y_min').value, self.get_parameter('y_max').value,
                  self.get_parameter('z_min').value, self.get_parameter('z_max').value)
        self.get_logger().info(
            f'geofence_monitor_node started: enabled={self.get_parameter("enabled").value}, '
            f'x=[{bounds[0]:.1f},{bounds[1]:.1f}] y=[{bounds[2]:.1f},{bounds[3]:.1f}] '
            f'z=[{bounds[4]:.1f},{bounds[5]:.1f}] -- BACKUP only, set PX4 GF_* params too')

    def _on_odom(self, msg: Odometry):
        if not self.get_parameter('enabled').value:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        inside = (self.get_parameter('x_min').value <= x <= self.get_parameter('x_max').value and
                  self.get_parameter('y_min').value <= y <= self.get_parameter('y_max').value and
                  self.get_parameter('z_min').value <= z <= self.get_parameter('z_max').value)

        if not inside and not self._breached:
            self._breached = True
            self.get_logger().error(
                f'GEOFENCE BREACH at pos=[{x:.2f} {y:.2f} {z:.2f}] -- commanding RTL')
            self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        elif inside and self._breached:
            self._breached = False
            self.get_logger().warn('Back inside geofence bounds')

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GeofenceMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
