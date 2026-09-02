#!/usr/bin/env python3
"""SITL-only stand-in for fsc_drone_state_estimator_ros2."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry

# Static frame quaternions (Hamilton, w,x,y,z), matching px4_ros_com's
# frame_transforms.h NED_ENU_Q / AIRCRAFT_BASELINK_Q constants.
NED_ENU_Q = (0.0, 0.70710678, 0.70710678, 0.0)
AIRCRAFT_BASELINK_Q = (0.0, 1.0, 0.0, 0.0)


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


class Px4OdomBridgeNode(Node):
    def __init__(self):
        super().__init__('px4_odom_bridge_node')

        self.declare_parameter('frame_id', 'map')

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self._pub = self.create_publisher(Odometry, 'state_estimator/local_position/odom', pub_qos)
        self.create_subscription(VehicleOdometry, 'fmu/out/vehicle_odometry', self._on_odom, sub_qos)

        self.get_logger().info('px4_odom_bridge_node started (NED fmu/out/vehicle_odometry -> ENU state_estimator/local_position/odom)')

    def _on_odom(self, msg: VehicleOdometry):
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.get_parameter('frame_id').value
        out.child_frame_id = 'odom'  # not "base_link" -- twist below is already world/ENU frame

        n, e, d = msg.position
        out.pose.pose.position.x = float(e)
        out.pose.pose.position.y = float(n)
        out.pose.pose.position.z = float(-d)

        vn, ve, vd = msg.velocity
        out.twist.twist.linear.x = float(ve)
        out.twist.twist.linear.y = float(vn)
        out.twist.twist.linear.z = float(-vd)

        qw, qx, qy, qz = (float(v) for v in msg.q)
        q_enu = qmul(NED_ENU_Q, qmul((qw, qx, qy, qz), AIRCRAFT_BASELINK_Q))
        out.pose.pose.orientation.w = q_enu[0]
        out.pose.pose.orientation.x = q_enu[1]
        out.pose.pose.orientation.y = q_enu[2]
        out.pose.pose.orientation.z = q_enu[3]

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = Px4OdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
