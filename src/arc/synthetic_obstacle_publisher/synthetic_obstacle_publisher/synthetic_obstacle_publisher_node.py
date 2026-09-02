import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


class SyntheticObstaclePublisher(Node):
    """
    Publishes a MarkerArray of synthetic spherical obstacles for visualization in RViz.
    """

    def __init__(self):
        super().__init__('synthetic_obstacle_publisher')

        # Flat list, groups of 4: x, y, z, diameter. Default: one obstacle
        # at (1.5, 0, 0) with diameter 1.0m, matching the geometry used in
        # the Stage 1 validation scripts (stress_test_thrust_floor.py).
        self.declare_parameter(
            'obstacles',
            [1.5, 0.0, 0.0, 1.0],
        )
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate_hz', 10.0)
        # Sensing envelope -- launch overrides these from the shared
        # obstacle_sensing_envelope.yaml (see class docstring). Defaults here
        # are only for a bare `ros2 run`.
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('z_min_m', 0.1)
        self.declare_parameter('z_max_m', 2.5)

        pub_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self._pub = self.create_publisher(MarkerArray, '/arc/obstacles', pub_qos)
        self.create_subscription(
            Odometry, 'state_estimator/local_position/odom', self._on_odom, odom_qos)

        self._obstacles_flat = None
        self._obstacles = []
        self._refresh_obstacles(initial=True)

        self._odom_valid = False
        self._pos_world = np.zeros(3)

        rate_hz = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f'synthetic_obstacle_publisher started: {len(self._obstacles)} obstacle(s) '
            f'on /arc/obstacles at {rate_hz:.1f} Hz, frame_id="{self.get_parameter("frame_id").value}", '
            f'sensing envelope max_range_m={self.get_parameter("max_range_m").value} '
            f'z=[{self.get_parameter("z_min_m").value},{self.get_parameter("z_max_m").value}]')

    def _refresh_obstacles(self, initial=False):
        """(Re-)parse the 'obstacles' parameter. Called once at construction and
        again every tick so a live SetParameters call (e.g. from the experiment
        GUI switching scenes) takes effect with no relaunch -- same "re-read
        tunables every tick" pattern the CBF node uses."""
        flat = list(self.get_parameter('obstacles').value)
        if flat == self._obstacles_flat:
            return
        if len(flat) % 4 != 0:
            msg = (f"'obstacles' parameter must be a flat list of (x,y,z,diameter) "
                   f"groups (length multiple of 4), got length {len(flat)}")
            if initial:
                raise ValueError(msg)
            self.get_logger().error(msg + ' -- keeping previous obstacle set')
            return
        self._obstacles_flat = flat
        self._obstacles = [tuple(flat[i:i + 4]) for i in range(0, len(flat), 4)]
        if not initial:
            self.get_logger().info(
                f'obstacles parameter updated live: now {len(self._obstacles)} obstacle(s)')

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pos_world = np.array([p.x, p.y, p.z])
        self._odom_valid = True

    def _tick(self):
        self._refresh_obstacles()
        if not self._odom_valid:
            self.get_logger().warning(
                'No odometry yet -- cannot evaluate the sensing envelope, publishing no obstacles',
                throttle_duration_sec=2.0)
            self._pub.publish(MarkerArray(markers=[]))
            return

        frame_id = self.get_parameter('frame_id').value
        stamp = self.get_clock().now().to_msg()
        max_range = self.get_parameter('max_range_m').value
        z_min = self.get_parameter('z_min_m').value
        z_max = self.get_parameter('z_max_m').value

        markers = []
        for idx, (x, y, z, diameter) in enumerate(self._obstacles):
            if not (z_min <= z <= z_max):
                continue
            radius = diameter / 2.0
            dist_to_surface = float(np.linalg.norm(self._pos_world - np.array([x, y, z]))) - radius
            if dist_to_surface > max_range:
                continue

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'synthetic_obstacles'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0
            m.scale.x = float(diameter)
            m.scale.y = float(diameter)
            m.scale.z = float(diameter)
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.b = 0.0
            m.color.a = 0.6
            markers.append(m)

        self._pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
