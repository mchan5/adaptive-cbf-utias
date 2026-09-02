#!/usr/bin/env python3
"""LiDAR-informed obstacle detection: PointCloud2 -> /arc/obstacles."""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

from lidar_obstacle_publisher.filters import roi_mask, self_filter_mask
from lidar_obstacle_publisher.hysteresis_tracker import HysteresisTracker
from lidar_obstacle_publisher.marker_geometry import vertical_extent
from lidar_obstacle_publisher.occupancy_grid import OccupancyGridClusterer


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

        self.declare_parameter('min_obstacle_diameter_m', 0.2)
        # Marker shape on /arc/obstacles. The consumer only reads pose + scale.x
        # (radius) either way; this is for RViz and matches the CBF's obstacle
        # model -- "cylinder" for the horizontal-distance (cylinder) barrier,
        # "sphere" for the spherical barrier on master.
        self.declare_parameter('marker_shape', 'sphere')
        # 0.5 == obs_d_min - obs_safety_margin: the CBF-QP clears every obstacle to a single shared
        # obs_d_min, so a cluster reported larger than this would be under-cleared.
        self.declare_parameter('max_obstacle_radius_m', 0.5)
        self.declare_parameter('max_obstacles', 20)
        self.declare_parameter('marker_lifetime_s', 0.3)

        # --- hysteresis tracker --------------------------------------------------
        # max_jump: greedy association gate (m). min_hits: consecutive frames a
        # detection must persist before it is published at all (birth debounce --
        # keeps flicker out of the QP / certification). max_misses: frames a
        # confirmed track keeps being published at its last pose after it stops
        # matching (death debounce -- survives a brief dropout). ema_alpha:
        # smoothing on centre/radius, 1.0 = snap to latest.
        self.declare_parameter('tracking_max_jump_m', 0.5)
        self.declare_parameter('track_min_hits', 3)
        self.declare_parameter('track_max_misses', 5)
        self.declare_parameter('track_ema_alpha', 0.5)

        self.declare_parameter('sensor_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('sensor_offset_rpy_deg', [0.0, 0.0, 0.0])

        # --- spatial pre-filters (both default-off) --------------------------------
        # Body-frame box half-extents around the airframe origin: points inside it
        # are the drone itself (props/legs/payload) in the sensor near field.
        # [0, 0, 0] disables. Read live, so `ros2 param set` retunes without a restart.
        self.declare_parameter('self_filter_half_extents_xyz', [0.0, 0.0, 0.0])
        # World-frame arena region of interest. When roi_enabled, points with X or Y
        # outside [roi_xy_min, roi_xy_max] are dropped before clustering so the room
        # walls never become obstacles. The flight geofence must sit strictly inside.
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_xy_min', [-5.0, -5.0])
        self.declare_parameter('roi_xy_max', [5.0, 5.0])

        # --- temporal 2D occupancy grid (clustering) ------------------------------
        # Grid geometry is FIXED at startup (it sizes the backing arrays); a change
        # needs a node restart. Big enough to cover max_range_m in every direction.
        self.declare_parameter('grid_origin_xy', [-8.0, -8.0])   # world XY of the grid's min corner
        self.declare_parameter('grid_size_xy', [16.0, 16.0])
        self.declare_parameter('grid_resolution_m', 0.1)
        # These retune live. half-life: a cell not re-hit decays to 50% in this many
        # seconds (removed obstacle fades in ~3x this). hit_increment: fraction added
        # per cell per frame it is seen (1 / this = frames to saturate). threshold:
        # occupancy above which a cell counts as an obstacle. min_cluster_cells:
        # smallest component kept.
        self.declare_parameter('occupancy_half_life_s', 0.4)
        self.declare_parameter('occupancy_hit_increment', 0.35)
        self.declare_parameter('occupancy_threshold', 0.5)
        self.declare_parameter('occupancy_reset_epsilon', 0.05)
        self.declare_parameter('min_cluster_cells', 4)

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

        self._tracker = HysteresisTracker()
        self._grid = OccupancyGridClusterer(
            self.get_parameter('grid_origin_xy').value,
            self.get_parameter('grid_size_xy').value,
            self.get_parameter('grid_resolution_m').value)
        self._last_tick = None   # rclpy.time.Time of the previous cloud, for the decay dt

        sf = self.get_parameter('self_filter_half_extents_xyz').value
        roi_on = self.get_parameter('roi_enabled').value
        roi_str = (
            f' roi=[{self.get_parameter("roi_xy_min").value},'
            f'{self.get_parameter("roi_xy_max").value}]' if roi_on else '')
        self.get_logger().info(
            f'lidar_obstacle_publisher_node started: subscribing to '
            f'{self.get_parameter("pointcloud_topic").value}, publishing /arc/obstacles '
            f'once per input cloud. sensor_offset_xyz={offset_xyz} '
            f'sensor_offset_rpy_deg={offset_rpy} -- verify these against the real mount '
            f'before trusting obstacle positions. '
            f'self_filter_half_extents_xyz={sf} roi_enabled={roi_on}{roi_str} '
            f'occupancy grid {self._grid.shape} @ {self.get_parameter("grid_resolution_m").value} m '
            f'origin {self.get_parameter("grid_origin_xy").value}')

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

        n_raw = points_sensor.shape[0]

        ranges = np.linalg.norm(points_sensor, axis=1)
        min_range = self.get_parameter('min_range_m').value
        max_range = self.get_parameter('max_range_m').value
        keep = (ranges >= min_range) & (ranges <= max_range)
        points_sensor = points_sensor[keep]
        n_range = points_sensor.shape[0]
        if n_range == 0:
            self._publish([], stamp)
            return

        points_body = (self._R_offset @ points_sensor.T).T + self._t_offset

        half_extents = self.get_parameter('self_filter_half_extents_xyz').value
        if any(h > 0.0 for h in half_extents):
            points_body = points_body[self_filter_mask(points_body, half_extents)]
            if points_body.shape[0] == 0:
                self._publish([], stamp)
                return
        n_self = points_body.shape[0]

        points_world = (self._R_body_world @ points_body.T).T + self._pos_world

        z_min = self.get_parameter('z_min_m').value
        z_max = self.get_parameter('z_max_m').value
        keep_world = (points_world[:, 2] >= z_min) & (points_world[:, 2] <= z_max)
        if self.get_parameter('roi_enabled').value:
            keep_world &= roi_mask(
                points_world,
                self.get_parameter('roi_xy_min').value,
                self.get_parameter('roi_xy_max').value)
        points_world = points_world[keep_world]

        # Grid ages in SENSOR time (header stamps), so decay is faithful whether
        # the node runs live or replays a bag. dt <= 0 (first cloud, or an
        # unstamped one) just skips the decay step.
        cloud_time = Time.from_msg(msg.header.stamp)
        dt = 0.0 if self._last_tick is None else max(
            0.0, min(1.0, (cloud_time - self._last_tick).nanoseconds * 1e-9))
        self._last_tick = cloud_time

        self._grid.decay(dt, self.get_parameter('occupancy_half_life_s').value)
        self._grid.add_cloud(
            points_world,
            hit_increment=self.get_parameter('occupancy_hit_increment').value,
            reset_epsilon=self.get_parameter('occupancy_reset_epsilon').value)

        threshold = self.get_parameter('occupancy_threshold').value
        grid_clusters = self._grid.clusters(
            threshold=threshold,
            min_cells=self.get_parameter('min_cluster_cells').value,
            max_radius=self.get_parameter('max_obstacle_radius_m').value)

        self.get_logger().info(
            f'cloud stages: raw={n_raw} -> range={n_range} -> self_filter={n_self} '
            f'-> band+roi={points_world.shape[0]} -> '
            f'occ_cells={self._grid.occupied_cell_count(threshold)} -> '
            f'clusters={len(grid_clusters)}',
            throttle_duration_sec=2.0)

        # Cap the detections fed to the tracker at the nearest max_obstacles so a
        # clustering blowup can't spawn unbounded tracks / monotonic ids.
        max_obstacles = self.get_parameter('max_obstacles').value
        pos_xy = self._pos_world[:2]
        if len(grid_clusters) > max_obstacles:
            grid_clusters.sort(
                key=lambda c: float(np.linalg.norm(c['centroid_xy'] - pos_xy)))
            grid_clusters = grid_clusters[:max_obstacles]

        tracks = self._tracker.update(
            grid_clusters,
            max_jump_m=self.get_parameter('tracking_max_jump_m').value,
            min_hits=self.get_parameter('track_min_hits').value,
            max_misses=self.get_parameter('track_max_misses').value,
            ema_alpha=self.get_parameter('track_ema_alpha').value)

        if len(tracks) > max_obstacles:
            nearest = sorted(
                tracks.items(),
                key=lambda kv: float(np.linalg.norm(kv[1]['center_xy'] - pos_xy)))
            tracks = dict(nearest[:max_obstacles])

        self.get_logger().info(
            f'tracks: {len(tracks)} published '
            f'({sum(1 for t in tracks.values() if t["coasting"])} coasting)',
            throttle_duration_sec=2.0)
        self._publish(self._build_markers(tracks), stamp)

    def _build_markers(self, tracks):
        min_diameter = self.get_parameter('min_obstacle_diameter_m').value
        lifetime = Duration(
            seconds=max(0.0, self.get_parameter('marker_lifetime_s').value)).to_msg()
        as_cylinder = self.get_parameter('marker_shape').value == 'cylinder'
        band_lo = self.get_parameter('z_min_m').value
        band_hi = self.get_parameter('z_max_m').value
        markers = []
        for obstacle_id, tr in tracks.items():
            diameter = max(2.0 * tr['radius'], min_diameter)
            z_mid, height = vertical_extent(
                tr['z_min'], tr['z_max'], band_lo, band_hi, min_diameter)
            m = Marker()
            m.ns = 'lidar_obstacles'
            m.id = int(obstacle_id)
            m.type = Marker.CYLINDER if as_cylinder else Marker.SPHERE
            m.action = Marker.ADD
            m.lifetime = lifetime
            m.pose.position.x = float(tr['center_xy'][0])
            m.pose.position.y = float(tr['center_xy'][1])
            m.pose.position.z = float(z_mid)
            m.pose.orientation.w = 1.0
            m.scale.x = diameter
            m.scale.y = diameter
            m.scale.z = float(height) if as_cylinder else diameter
            m.color.r = 0.9
            m.color.g = 0.1
            m.color.b = 0.1
            m.color.a = 0.35 if tr['coasting'] else 0.6   # coasting tracks render dimmer
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
