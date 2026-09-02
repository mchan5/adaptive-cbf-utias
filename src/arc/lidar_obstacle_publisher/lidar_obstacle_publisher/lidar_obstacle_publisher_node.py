#!/usr/bin/env python3
"""LiDAR-informed obstacle detection: PointCloud2 -> /arc/obstacles."""

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
        # max_range_m / z_min_m / z_max_m / max_obstacle_radius_m below are the shared "sensing
        # envelope" -- every launch path overrides these from …
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('z_min_m', 0.1)   # world-frame obstacle band -- excludes
        self.declare_parameter('z_max_m', 2.5)   # floor returns and ceiling/overhead clutter

        self.declare_parameter('voxel_size_m', 0.15)
        self.declare_parameter('min_cluster_points', 8)
        self.declare_parameter('min_obstacle_diameter_m', 0.2)
        # 0.5 == obs_d_min - obs_safety_margin: the CBF-QP clears every obstacle to a single shared
        # obs_d_min, so a cluster reported larger than this would be under-cleared.
        self.declare_parameter('max_obstacle_radius_m', 0.5)
        self.declare_parameter('max_obstacles', 20)
        self.declare_parameter('tracking_max_jump_m', 0.5)

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
        # One publish per input cloud -- no separate timer.
        stamp = msg.header.stamp

        if not self._odom_valid:
            self.get_logger().warning(
                'No odometry yet -- cannot place LiDAR points in the world frame, dropping cloud',
                throttle_duration_sec=2.0)
            return

        # read_points (not read_points_numpy): Humble's read_points_numpy asserts a single
        # datatype across ALL fields in the cloud, which a Livox PointCloud2 (float32 xyz +
        # uint8 tag/line + float64 timestamp) violates. read_points builds a compound dtype
        # from only the requested fields and is safe.
        structured = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points_sensor = np.column_stack(
            (structured['x'], structured['y'], structured['z'])).astype(np.float64, copy=False)
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
