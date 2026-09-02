#!/usr/bin/env python3
"""Real-hardware mocap bridge: OptiTrack (via natnet_ros2) ->
state_estimator/local_position/odom."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class MocapOdomBridgeNode(Node):
    def __init__(self):
        super().__init__('mocap_odom_bridge_node')

        self.declare_parameter('frame_id', 'map')
        # Must match the rigid body's name as defined in Motive.
        self.declare_parameter('model_name', 'uav_0')
        # EMA filter weight on the newest finite-difference velocity sample, in [0, 1]. Higher =
        # more responsive/noisier, lower = smoother/more lagged.
        self.declare_parameter('velocity_filter_alpha', 0.3)
        # Guard against a bogus huge velocity spike from a dropped-frame gap or duplicate timestamp
        # -- if dt is outside this range, hold the last filtered velocity instead of updating from …
        self.declare_parameter('min_dt_sec', 0.001)
        self.declare_parameter('max_dt_sec', 0.5)

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
        pose_topic = '/' + model_name + '/pose'
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, sub_qos)

        self._have_prev = False
        self._prev_time = Time()
        self._prev_pos = (0.0, 0.0, 0.0)
        self._vel_filtered = (0.0, 0.0, 0.0)

        self.get_logger().info(
            f'mocap_odom_bridge_node started ({pose_topic} -> ENU state_estimator/local_position/odom) '
            '-- real OptiTrack (natnet_ros2) passthrough with finite-difference velocity, no IMU fusion')

    def _on_pose(self, msg: PoseStamped):
        now = Time.from_msg(msg.header.stamp)
        pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

        if self._have_prev:
            dt = (now - self._prev_time).nanoseconds * 1e-9
            min_dt = self.get_parameter('min_dt_sec').value
            max_dt = self.get_parameter('max_dt_sec').value
            if min_dt <= dt <= max_dt:
                alpha = self.get_parameter('velocity_filter_alpha').value
                raw_vel = tuple((pos[i] - self._prev_pos[i]) / dt for i in range(3))
                self._vel_filtered = tuple(
                    alpha * raw_vel[i] + (1.0 - alpha) * self._vel_filtered[i] for i in range(3))
            else:
                self.get_logger().warning(
                    f'Rejected mocap dt={dt:.4f}s (outside [{min_dt}, {max_dt}]) -- holding last '
                    'filtered velocity. Frequent hits mean dropped frames or a timestamp problem.',
                    throttle_duration_sec=2.0)
        else:
            self._have_prev = True

        self._prev_time = now
        self._prev_pos = pos

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.get_parameter('frame_id').value
        # Not "base_link" -- twist.linear below is already world/ENU frame.
        out.child_frame_id = 'odom'

        out.pose.pose.position.x = pos[0]
        out.pose.pose.position.y = pos[1]
        out.pose.pose.position.z = pos[2]
        out.pose.pose.orientation = msg.pose.orientation

        out.twist.twist.linear.x = self._vel_filtered[0]
        out.twist.twist.linear.y = self._vel_filtered[1]
        out.twist.twist.linear.z = self._vel_filtered[2]

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
