#!/usr/bin/env python3
"""Real-hardware mocap bridge: OptiTrack (via natnet_ros2) -> state_estimator/local_position/odom.

Target driver: L2S-lab/natnet_ros2 (https://github.com/L2S-lab/natnet_ros2),
vendored into this workspace at ros2_ws/src/natnet_ros2 and build-verified
against ROS2 Humble. It's a real NatNet 4 client for Motive, not a simulator
stand-in -- unlike an earlier version of this node, which assumed a
fsc_autopilot_ros2_msgs/msg/Mocap contract that, as far as could be found, no
real driver actually publishes (only the Gazebo/Isaac Sim *emulators* of
OptiTrack produce that message -- see gz_optitrack_ros2_emulator).

natnet_ros2 publishes each rigid body as geometry_msgs/PoseStamped on the
absolute topic /<body-name>/pose, where <body-name> is whatever the rigid
body is named inside Motive -- name it to match uav_prefix (e.g. "uav_0")
there so this node's model_name param lines up without translation. Absolute
topic, same caveat as elsewhere in this package: it will NOT land under this
node's ROS namespace automatically.

Critical difference from the old assumption: PoseStamped carries no
velocity. This node estimates linear velocity itself via finite difference
of consecutive poses (using each message's own header.stamp, not wall-clock
arrival time, since natnet_ros2 timestamps at capture) with a light
exponential-moving-average filter -- raw finite-difference of mocap-noise
position is noisy enough on its own to be a bad idea to feed straight into a
100Hz rate controller. velocity_filter_alpha trades responsiveness for
smoothness; tune it against your actual marker jitter, not blindly.

natnet_ros2 itself only needs network reachability to the Motive PC (NatNet
is UDP/multicast) -- it does not need to run on the same host as Motive, so
running it directly on the companion computer alongside this bridge (rather
than on a separate ground-station machine plus cross-host ROS2 discovery) is
the simpler option if the companion computer is on the same network as the
mocap PC. It's also a lifecycle node -- must be explicitly configured AND
activated (natnet_ros2's own launch file's activate:=true does this), or it
will sit there publishing nothing.

Deliberately launched separately from the main hardware stack (see
cbf_rate_arc_hardware_launch.py's comments) -- a wrong serverIP/clientIP
should fail loudly and visibly on its own, not get buried inside a combined
launch you're trying to trust for flight.

Still NOT a substitute for a real IMU-fused state estimator
(fsc_drone_state_estimator_ros2, referenced upstream, never vendored
anywhere in this repo) -- this is a raw mocap passthrough plus a bolted-on
velocity estimate, noisier/higher-latency than a proper fused estimate would
be. Validate it (compare against a known-good reference, check for spikes
across dropped frames) before trusting it in flight.
"""

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
        # EMA filter weight on the newest finite-difference velocity sample,
        # in [0, 1]. Higher = more responsive/noisier, lower = smoother/more
        # lagged. 0.3 is a starting point, not a validated value.
        self.declare_parameter('velocity_filter_alpha', 0.3)
        # Guard against a bogus huge velocity spike from a dropped-frame gap
        # or duplicate timestamp -- if dt is outside this range, hold the
        # last filtered velocity instead of updating from a bad sample.
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
