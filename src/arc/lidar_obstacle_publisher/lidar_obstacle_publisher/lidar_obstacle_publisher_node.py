#!/usr/bin/env python3
"""LiDAR-informed obstacle detection: PointCloud2 -> /arc/obstacles.

This is option 2 of "how obstacles get into /arc/obstacles" -- option 1 is
hand-measured positions via synthetic_obstacle_publisher fed an obstacle_file
(see cbf_rate_arc_hardware_launch.py's obstacle_source launch arg, which
selects between the two). Both publish the exact same
visualization_msgs/MarkerArray contract synthetic_obstacle_publisher already
uses (marker.pose.position = centroid, marker.scale.x = diameter,
marker.id = a stable per-obstacle id) -- single_vehicle_cbf_rate_arc's
obstaclesCallback() does not know or care which one is running.

Pipeline per point cloud: range/self filter -> transform sensor frame to
world/ENU using the latest state_estimator/local_position/odom pose -> world
z-band filter -> voxel-grid downsample -> connected-component clustering over
occupied voxels -> per-cluster centroid/radius -> greedy nearest-centroid ID
association against the previous frame's clusters (tracker.py), so a physical
obstacle keeps the same marker.id across frames.

/arc/obstacles is published exactly once per input cloud (no decoupled
timer): output rate follows the sensor, and a stalled LiDAR/driver shows up
as the topic going silent rather than as stale markers restamped "now".

Driver: Livox-SDK/livox_ros_driver2, vendored and build-verified against
ROS2 Humble at ros2_ws/src/livox_ros_driver2 (needs Livox-SDK2, also
vendored+built at $HOME/src/Livox-SDK2 and installed to $HOME/.local -- no
sudo/system install required, see that build's CMAKE_PREFIX_PATH). Target
unit: Livox Mid-360S (https://www.livoxtech.com/mid-360s), using the
driver's own MID360s_config.json / msg_MID360s_launch.py (a distinct
variant from the plain MID360_config.json -- the driver repo treats "S" as
its own product).

Confirmed by reading the driver source (not guessed):
  - xfer_format lives in launch_ROS2/msg_MID360s_launch.py (a Python
    variable, NOT in the json config, despite what an earlier version of
    this comment claimed), and has been changed from upstream's default of
    1 (custom livox_ros_driver2/msg/CustomMsg) to 0 (sensor_msgs/PointCloud2
    with PointXYZRTL fields -- an actual sensor_msgs/PointCloud2, this node
    just ignores the extra reflectivity/tag/line/timestamp fields beyond
    x/y/z). Only switch back to CustomMsg if something later needs that
    per-point data (e.g. LIO using the unit's built-in IMU).
  - The driver publishes on the relative topic "livox/lidar"
    (src/lddc.cpp), which resolves to the absolute /livox/lidar when
    launched unnamespaced (msg_MID360s_launch.py's Node has no namespace
    set) -- matching this repo's lidar_pointcloud_topic default already.
  - Data interface is Ethernet/UDP, not something else -- MID360s_config.json
    has host_net_info (this machine's IP + 5 UDP ports) and lidar_configs[0].ip
    (the sensor's factory-default IP, typically on the 192.168.1.x subnet).
    Both must be set to match your actual network before the driver will see
    the unit; the checked-in defaults are the Livox factory defaults, not
    validated against a specific network.
  - MID360s_config.json also has its own lidar_configs[0].extrinsic_parameter
    (roll/pitch/yaw/x/y/z) -- i.e. the driver itself can apply the sensor-to-body
    mount offset before publishing. Pick ONE place to apply it, not both: either
    that json field, or this node's sensor_offset_xyz/sensor_offset_rpy_deg
    params below, left at zero in the other. This repo defaults to doing it
    here (Python, no rebuild needed to retune).

Still unverified (wasn't on the livoxtech.com product page, wasn't needed to
get the driver building, and matters for wiring, not code): exact power input
voltage/wattage. Check the datasheet before wiring power to it.

Per Livox's published spec: 360 deg horizontal x 59 deg vertical FOV
(markedly asymmetric upward if mounted flat -- good for wall/pole-height
obstacles, worse for anything low right under the vehicle), ranging
accuracy not guaranteed under 0.2m (hence min_range_m's default), up to
40m detection range at 10% reflectivity (max_range_m defaults to a much
tighter 8m here since that's plenty for an indoor test volume and keeps
the point count/clustering cost down -- raise it if you actually need the
range).

Deliberately simple clustering (voxel connected-components, not a
statistical/learned detector): correct and easy to reason about, but tune
voxel_size_m / min_cluster_points / z_min / z_max / max_range_m for your
actual sensor and environment before trusting it, and expect false positives
from propwash-kicked debris or reflective surfaces that a fancier pipeline
would reject.

The shared sensing-envelope parameters (max_range_m, z_min_m, z_max_m,
max_obstacle_radius_m) are set for every launch path from
single_vehicle_cbf_rate_arc/config/obstacle_sensing_envelope.yaml, so this
node and synthetic_obstacle_publisher present the CBF node with the same
conditions. max_obstacle_radius_m defaults to 0.5 m there
(obs_d_min - obs_safety_margin): the CBF-QP collision constraint uses one
shared obs_d_min for all obstacles, so a cluster reported larger than that
would be under-cleared.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

from lidar_obstacle_publisher.tracker import GreedyCentroidTracker

_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def _quat_to_rotation_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rpy_deg_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """R = Rz * Ry * Rx (body-frame intrinsic roll-pitch-yaw). Re-derive/verify
    against your actual sensor mount before trusting obstacle positions."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _cluster_points(points, voxel_size, min_cluster_points, max_cluster_radius):
    """Voxel-grid connected-component clustering. Returns a list of
    (centroid_xyz, radius, num_points)."""
    if points.shape[0] == 0:
        return []

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    voxel_map = {}
    for i, key in enumerate(map(tuple, voxel_idx)):
        voxel_map.setdefault(key, []).append(i)

    occupied = set(voxel_map.keys())
    visited = set()
    clusters = []

    for start in occupied:
        if start in visited:
            continue
        visited.add(start)
        voxel_group = [start]
        stack = [start]
        while stack:
            v = stack.pop()
            for off in _NEIGHBOR_OFFSETS:
                nb = (v[0] + off[0], v[1] + off[1], v[2] + off[2])
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    voxel_group.append(nb)
                    stack.append(nb)

        point_idxs = [i for v in voxel_group for i in voxel_map[v]]
        if len(point_idxs) < min_cluster_points:
            continue

        pts = points[point_idxs]
        centroid = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).max())
        radius = min(radius, max_cluster_radius)
        clusters.append((centroid, radius, len(point_idxs)))

    return clusters


class LidarObstaclePublisherNode(Node):
    def __init__(self, **kwargs):
        # **kwargs is forwarded to rclpy.node.Node (e.g. parameter_overrides
        # from test/test_bag_replay.py); empty for the normal entry point.
        super().__init__('lidar_obstacle_publisher_node', **kwargs)

        self.declare_parameter('pointcloud_topic', 'lidar/points')
        self.declare_parameter('frame_id', 'map')

        self.declare_parameter('min_range_m', 0.2)
        # max_range_m / z_min_m / z_max_m / max_obstacle_radius_m below are the
        # shared "sensing envelope" -- every launch path overrides these from
        # single_vehicle_cbf_rate_arc/config/obstacle_sensing_envelope.yaml so
        # this node and synthetic_obstacle_publisher agree. These defaults only
        # apply to a bare `ros2 run lidar_obstacle_publisher ...`.
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('z_min_m', 0.1)   # world-frame obstacle band -- excludes
        self.declare_parameter('z_max_m', 2.5)   # floor returns and ceiling/overhead clutter

        self.declare_parameter('voxel_size_m', 0.15)
        self.declare_parameter('min_cluster_points', 8)
        self.declare_parameter('min_obstacle_diameter_m', 0.2)
        # 0.5 == obs_d_min - obs_safety_margin: the CBF-QP clears every obstacle
        # to a single shared obs_d_min, so a cluster reported larger than this
        # would be under-cleared. See obstacle_sensing_envelope.yaml.
        self.declare_parameter('max_obstacle_radius_m', 0.5)
        self.declare_parameter('max_obstacles', 20)
        self.declare_parameter('tracking_max_jump_m', 0.5)

        # Static sensor-in-body-frame mount transform -- MUST be measured for
        # your actual mount, defaults are "sensor at body origin, aligned
        # with body FLU axes" which is almost certainly wrong.
        self.declare_parameter('sensor_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('sensor_offset_rpy_deg', [0.0, 0.0, 0.0])

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
            PointCloud2, self.get_parameter('pointcloud_topic').value,
            self._on_pointcloud, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, 'state_estimator/local_position/odom', self._on_odom, odom_qos)

        offset_xyz = self.get_parameter('sensor_offset_xyz').value
        offset_rpy = self.get_parameter('sensor_offset_rpy_deg').value
        self._t_offset = np.array(offset_xyz, dtype=float)
        self._R_offset = _rpy_deg_to_rotation_matrix(*offset_rpy)

        self._odom_valid = False
        self._pos_world = np.zeros(3)
        self._R_body_world = np.eye(3)

        self._tracker = GreedyCentroidTracker()

        self.get_logger().info(
            f'lidar_obstacle_publisher_node started: subscribing to '
            f'{self.get_parameter("pointcloud_topic").value}, publishing /arc/obstacles '
            f'once per input cloud. sensor_offset_xyz={offset_xyz} '
            f'sensor_offset_rpy_deg={offset_rpy} -- verify these against the real mount '
            f'before trusting obstacle positions.')

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pos_world = np.array([p.x, p.y, p.z])
        self._R_body_world = _quat_to_rotation_matrix(q.w, q.x, q.y, q.z)
        self._odom_valid = True

    def _on_pointcloud(self, msg: PointCloud2):
        # One publish per input cloud -- no separate timer. A stalled LiDAR /
        # driver therefore shows up directly as /arc/obstacles going silent
        # (visible on `ros2 topic hz` and the experiment GUI), instead of a
        # timer re-emitting the last markers with fresh timestamps forever.
        stamp = msg.header.stamp

        if not self._odom_valid:
            self.get_logger().warning(
                'No odometry yet -- cannot place LiDAR points in the world frame, dropping cloud',
                throttle_duration_sec=2.0)
            return

        points_sensor = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if points_sensor.shape[0] == 0:
            self._publish([], stamp)
            return

        ranges = np.linalg.norm(points_sensor, axis=1)
        min_range = self.get_parameter('min_range_m').value
        max_range = self.get_parameter('max_range_m').value
        keep = (ranges >= min_range) & (ranges <= max_range)
        points_sensor = points_sensor[keep]
        if points_sensor.shape[0] == 0:
            self._publish([], stamp)
            return

        points_body = (self._R_offset @ points_sensor.T).T + self._t_offset
        points_world = (self._R_body_world @ points_body.T).T + self._pos_world

        z_min = self.get_parameter('z_min_m').value
        z_max = self.get_parameter('z_max_m').value
        z_keep = (points_world[:, 2] >= z_min) & (points_world[:, 2] <= z_max)
        points_world = points_world[z_keep]

        clusters = _cluster_points(
            points_world,
            voxel_size=self.get_parameter('voxel_size_m').value,
            min_cluster_points=self.get_parameter('min_cluster_points').value,
            max_cluster_radius=self.get_parameter('max_obstacle_radius_m').value,
        )

        max_obstacles = self.get_parameter('max_obstacles').value
        if len(clusters) > max_obstacles:
            clusters.sort(key=lambda c: np.linalg.norm(c[0] - self._pos_world))
            clusters = clusters[:max_obstacles]

        tracks = self._tracker.update(
            clusters, self.get_parameter('tracking_max_jump_m').value)
        self._publish(self._build_markers(tracks), stamp)

    def _build_markers(self, tracks):
        min_diameter = self.get_parameter('min_obstacle_diameter_m').value
        markers = []
        for obstacle_id, (centroid, radius) in tracks.items():
            diameter = max(2.0 * radius, min_diameter)
            m = Marker()
            m.ns = 'lidar_obstacles'
            m.id = obstacle_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(centroid[0])
            m.pose.position.y = float(centroid[1])
            m.pose.position.z = float(centroid[2])
            m.pose.orientation.w = 1.0
            m.scale.x = diameter
            m.scale.y = diameter
            m.scale.z = diameter
            m.color.r = 0.9
            m.color.g = 0.1
            m.color.b = 0.1
            m.color.a = 0.6
            markers.append(m)
        return markers

    def _publish(self, markers, stamp):
        frame_id = self.get_parameter('frame_id').value
        for m in markers:
            m.header.frame_id = frame_id
            m.header.stamp = stamp
        self._pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstaclePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
