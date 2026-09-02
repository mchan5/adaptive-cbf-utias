#!/usr/bin/env python3
"""SITL-only stand-in for fsc_drone_state_estimator_ros2, backed by Gazebo ground truth instead
of PX4's EKF."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from fsc_autopilot_ros2_msgs.msg import Mocap


class MocapOdomBridgeNode(Node):
    def __init__(self):
        super().__init__('mocap_odom_bridge_node')

        self.declare_parameter('frame_id', 'map')
        # gz_optitrack_ros2_emulator publishes to the absolute topic "/<model_name>/mocap" (see
        # GazeboMocapEmulator::ConstructTopics), ignoring this node's own namespace -- must match …
        self.declare_parameter('model_name', 'uav_0')

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self._pub = self.create_publisher(Odometry, 'state_estimator/local_position/odom', pub_qos)

        model_name = self.get_parameter('model_name').value
        mocap_topic = '/' + model_name + '/mocap'
        self.create_subscription(Mocap, mocap_topic, self._on_mocap, sub_qos)

        self.get_logger().info(
            f'mocap_odom_bridge_node started ({mocap_topic} -> ENU state_estimator/local_position/odom)')

    def _on_mocap(self, msg: Mocap):
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.get_parameter('frame_id').value
        # Not "base_link" -- twist.linear below is already world/ENU frame
        # (gz_optitrack_emulator_node rotates it out of body/FLU already).
        out.child_frame_id = 'odom'

        out.pose.pose.position.x = msg.pose.position.x
        out.pose.pose.position.y = msg.pose.position.y
        out.pose.pose.position.z = msg.pose.position.z
        out.pose.pose.orientation = msg.pose.orientation

        out.twist.twist.linear.x = msg.twist.linear.x
        out.twist.twist.linear.y = msg.twist.linear.y
        out.twist.twist.linear.z = msg.twist.linear.z

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MocapOdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
