import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PointStamped
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand, VehicleLocalPosition, VehicleStatus)
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

NAN = float('nan')

# PENN+GAT model tree lives in the repo at research/penn_gat/nn_model. Resolve
# it from $ARC_ROOT (exported by scripts/setup.sh) so no absolute path is baked
# into the source; the ~/arc_2026 fallback only applies when the env var is
# unset. If the tree is absent the node logs a warning and falls back to the
# fixed cbf_alpha_obs gain.
_ARC_ROOT = os.environ.get('ARC_ROOT', os.path.expanduser('~/arc_2026'))
_PENN_MODEL_DIR = os.path.join(_ARC_ROOT, 'research', 'penn_gat', 'nn_model')

def ned_to_enu(v): return np.array([v[1], v[0], -v[2]])
def enu_to_ned(v): return np.array([v[1], v[0], -v[2]])

class ObstacleCBFNode(Node):

    ARMING_STATE_ARMED = 2

    def __init__(self):
        super().__init__('obstacle_cbf_node')

        self.declare_parameter('boundary_x_min', -3.0)
        self.declare_parameter('boundary_x_max',  3.0)
        self.declare_parameter('boundary_y_min', -3.0)
        self.declare_parameter('boundary_y_max',  3.0)
        self.declare_parameter('boundary_z_min',  0.5)
        self.declare_parameter('boundary_z_max',  2.5)

        self.declare_parameter('waypoint_x', 0.0)
        self.declare_parameter('waypoint_y', 0.0)
        self.declare_parameter('waypoint_z', 1.5)

        self.declare_parameter('cbf_alpha',        0.5)
        self.declare_parameter('cbf_alpha_obs',    1.0)
        self.declare_parameter('obs_safety_margin', 0.5)
        self.declare_parameter('k_p',              0.5)
        self.declare_parameter('v_max',            0.5)
        self.declare_parameter('yaw_deg',          0.0)

        # PENN+GAT adaptive gamma parameters
        self.declare_parameter('penn_enabled',     True)
        self.declare_parameter('penn_model_dir',   _PENN_MODEL_DIR)
        self.declare_parameter('penn_model_path',  'checkpoint/Quadrotor3D_gat.pth')
        self.declare_parameter('penn_gamma_min',   0.5)
        self.declare_parameter('penn_gamma_max',   10.0)
        self.declare_parameter('penn_gamma_steps', 20)
        self.declare_parameter('penn_update_ticks', 20)       # ticks between re-selection (1 s at 20 Hz)
        self.declare_parameter('epistemic_threshold', 0.046695)
        self.declare_parameter('cvar_boundary',       0.1570)
        self.declare_parameter('deadlock_threshold',   5.0)   # seconds

        pub_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        sub_qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos)
        self._command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', pub_qos)
        self._drone_viz_pub = self.create_publisher(Marker, '/arc/drone_viz', 10)

        # PX4's own EKF local position (GPS-fused outdoors). Nothing in this repo
        # publishes '/uav_0/state/pose' -- that subscription was a dead wire.
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.position_callback, sub_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v4', self.vehicle_status_callback, sub_qos)

        self.create_subscription(PointStamped, '/arc/boundary_min', self.boundary_min_callback, sub_qos)
        self.create_subscription(PointStamped, '/arc/boundary_max', self.boundary_max_callback, sub_qos)
        self.create_subscription(PointStamped, '/arc/waypoint', self.waypoint_callback, sub_qos)
        self.create_subscription(MarkerArray, '/arc/obstacles', self.obstacles_callback, sub_qos)

        self._pos_ned = np.zeros(3)
        self._pos_valid = False
        self._arming_state = -1
        self._tick_count = 0
        self.origin_ned = None
        self._last_pos_time = None
        self._at_waypoint = False

        self._arc_min = None
        self._arc_max = None
        self._arc_wp = None

        wp0 = np.array([self.get_parameter('waypoint_x').value,
                        self.get_parameter('waypoint_y').value])
        if np.linalg.norm(wp0) > 0.1:
            self._yaw = math.atan2(wp0[0], wp0[1])
        else:
            self._yaw = math.radians(self.get_parameter('yaw_deg').value)

        self._obstacles: list[tuple[np.ndarray, float]] = []

        # PENN state
        self._penn_model = None
        self._gat_module = None
        self._alpha_obs_selected = self.get_parameter('cbf_alpha_obs').value
        self._penn_tick_counter = 0
        self._load_penn_model()

        self.create_timer(0.05, self._tick)
        self.get_logger().info('Obstacle CBF node started')

    # ── PENN model loading ────────────────────────────────────────────────────

    def _load_penn_model(self):
        model_dir = self.get_parameter('penn_model_dir').value
        model_rel = self.get_parameter('penn_model_path').value
        model_abs = os.path.join(model_dir, model_rel)

        if not os.path.exists(model_abs):
            self.get_logger().warn(f'PENN model not found at {model_abs} — using fixed cbf_alpha_obs')
            return

        try:
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            import torch
            from gat_3d import GATModule3D
            from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            gat_module = GATModule3D(robot_radius=0.25, device=device).to(device)
            penn_gat = ProbabilisticEnsembleGAT(
                gat_module.gat, n_output=2, n_hidden=40, n_ensemble=3,
                gamma_dim=1, device=device, lr=0.0003,
            ).to(device)
            penn_gat.load_model(model_abs)
            penn_gat.eval()

            self._penn_model = penn_gat
            self._gat_module = gat_module
            self._torch = torch
            self.get_logger().info(f'PENN+GAT model loaded from {model_abs} (device={device})')
        except Exception as e:
            self.get_logger().error(f'PENN model load failed: {e} — using fixed cbf_alpha_obs')

    # ── PENN gamma selection ──────────────────────────────────────────────────

    def _penn_select_alpha(self, pos_enu):
        """Return the highest gamma candidate whose predicted risk and deadlock are acceptable."""
        if not self.get_parameter('penn_enabled').value:
            return self.get_parameter('cbf_alpha_obs').value
        if self._penn_model is None or len(self._obstacles) == 0:
            return self.get_parameter('cbf_alpha_obs').value

        # Skip inference during position hold — alpha_obs has no effect there
        if self._at_waypoint:
            return self._alpha_obs_selected

        if self._arc_wp is not None:
            goal = self._arc_wp.tolist()
        else:
            goal = [self.get_parameter('waypoint_x').value,
                    self.get_parameter('waypoint_y').value,
                    self.get_parameter('waypoint_z').value]

        robot_state = [float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2]),
                       0.0, 0.0, 0.0]
        obs_list = [[float(c[0]), float(c[1]), float(c[2]), float(r),
                     0.0, 0.0, 0.0]
                    for c, r in self._obstacles]

        try:
            graph = self._gat_module.create_graph(
                robot=robot_state, obstacles=obs_list, goal=goal)

            g_min  = self.get_parameter('penn_gamma_min').value
            g_max  = self.get_parameter('penn_gamma_max').value
            steps  = int(self.get_parameter('penn_gamma_steps').value)
            ep_thr = self.get_parameter('epistemic_threshold').value
            cvar   = self.get_parameter('cvar_boundary').value
            dl_thr = self.get_parameter('deadlock_threshold').value

            candidates = self._torch.linspace(g_min, g_max, steps).unsqueeze(1)
            safety_preds, deadlock_preds, divs = \
                self._penn_model.predict_single_scene(graph, candidates)

            # Iterate from largest gamma down, return the first feasible one
            first_reject_reason = None
            for i in range(steps - 1, -1, -1):
                gamma_i = float(candidates[i].item())
                div_i = divs[i]
                if ep_thr > 0 and div_i > ep_thr:
                    if first_reject_reason is None:
                        first_reject_reason = f'epistemic div={div_i:.4f} > {ep_thr:.4f}'
                    continue
                mean_risk     = float(np.mean([m for m, _ in safety_preds[i]]))
                mean_deadlock = float(np.mean([m for m, _ in deadlock_preds[i]]))
                if mean_risk < cvar and mean_deadlock < dl_thr:
                    self.get_logger().info(
                        f'PENN select: gamma={gamma_i:.3f}  '
                        f'risk={mean_risk:.4f} (thr={cvar:.4f})  '
                        f'deadlock={mean_deadlock:.4f} (thr={dl_thr:.1f})  '
                        f'div={div_i:.4f}')
                    return gamma_i
                if first_reject_reason is None:
                    first_reject_reason = (f'risk={mean_risk:.4f}>={cvar:.4f} '
                                           f'or deadlock={mean_deadlock:.4f}>={dl_thr:.1f}')

            # No feasible gamma found — fall back to most conservative
            self.get_logger().warn(
                f'PENN: no feasible gamma (reject reason: {first_reject_reason}) '
                f'— using g_min={g_min:.3f}')
            return float(g_min)

        except Exception as e:
            self.get_logger().warn(f'PENN inference failed: {e}', throttle_duration_sec=5.0)
            return self.get_parameter('cbf_alpha_obs').value

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def position_callback(self, msg: VehicleLocalPosition):
        self._pos_ned = np.array([msg.x, msg.y, msg.z])
        self._pos_valid = bool(msg.xy_valid and msg.z_valid)
        if self._pos_valid:
            self._last_pos_time = self.get_clock().now()

    def vehicle_status_callback(self, msg: VehicleStatus):
        self._arming_state = msg.arming_state

    def boundary_min_callback(self, msg: PointStamped):
        self._arc_min = np.array([msg.point.x, msg.point.y, msg.point.z])

    def boundary_max_callback(self, msg: PointStamped):
        self._arc_max = np.array([msg.point.x, msg.point.y, msg.point.z])

    def waypoint_callback(self, msg: PointStamped):
        self._arc_wp = np.array([msg.point.x, msg.point.y, msg.point.z])

    def obstacles_callback(self, msg: MarkerArray):
        margin = self.get_parameter('obs_safety_margin').value
        self._obstacles = [
            (np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]),
             m.scale.x / 2.0 + margin)
            for m in msg.markers
        ]

    # ── Main control tick ─────────────────────────────────────────────────────

    def _tick(self):
        self._publish_heartbeat()
        self._tick_count += 1

        origin = self.origin_ned if self.origin_ned is not None else np.zeros(3)
        self._publish_drone_viz(ned_to_enu(self._pos_ned - origin))

        if self._tick_count < 50:
            return

        if self._tick_count == 50:
            self._engage_offboard()
            self._arm()
            return

        if self._arming_state != self.ARMING_STATE_ARMED and self._tick_count % 100 == 0:
            self._engage_offboard()
            self._arm()
            return

        if not self._pos_valid:
            self._publish_velocity(np.zeros(3))
            return

        if self._last_pos_time is not None:
            age = (self.get_clock().now() - self._last_pos_time).nanoseconds * 1e-9
            if age > 2.0:
                self.get_logger().warn(f'Position feed stale ({age:.1f}s) — zeroing velocity.', throttle_duration_sec=1.0)
                self._publish_velocity(np.zeros(3))
                return

        if self._arming_state == self.ARMING_STATE_ARMED and self.origin_ned is None:
            self.origin_ned = self._pos_ned.copy()
            self.get_logger().info(f'Arming detected — setting origin to {self.origin_ned}')

        if self.origin_ned is None:
            self._publish_velocity(np.zeros(3))
            return

        pos_enu = ned_to_enu(self._pos_ned - self.origin_ned)

        if self._arc_wp is not None:
            wp = self._arc_wp.copy()
        else:
            wp = np.array([self.get_parameter('waypoint_x').value,
                           self.get_parameter('waypoint_y').value,
                           self.get_parameter('waypoint_z').value])

        if self._arc_min is not None and self._arc_max is not None:
            bmin, bmax = self._arc_min, self._arc_max
            boundary = (bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2])
        else:
            boundary = self._boundary_params()

        x_min, x_max, y_min, y_max, z_min, z_max = boundary

        wp = np.array([np.clip(wp[0], x_min, x_max),
                       np.clip(wp[1], y_min, y_max),
                       np.clip(wp[2], z_min, z_max)])

        # Takeoff phase: climb straight up to z_min before engaging CBF/navigation.
        # Avoids simultaneous multi-obstacle CBF corrections destabilising attitude on liftoff.
        TAKEOFF_CLEARANCE = 0.1
        if pos_enu[2] < z_min + TAKEOFF_CLEARANCE:
            self._publish_velocity(enu_to_ned(np.array([0.0, 0.0, 0.5])))
            if self._tick_count % 20 == 0:
                self.get_logger().info(
                    f'TAKEOFF climbing z={pos_enu[2]:.2f}m → target={z_min + TAKEOFF_CLEARANCE:.2f}m')
            return

        alpha     = self.get_parameter('cbf_alpha').value
        k_p       = self.get_parameter('k_p').value
        v_max     = self.get_parameter('v_max').value

        # Update PENN-selected alpha periodically
        self._penn_tick_counter += 1
        if self._penn_tick_counter >= self.get_parameter('penn_update_ticks').value:
            self._penn_tick_counter = 0
            self._alpha_obs_selected = self._penn_select_alpha(pos_enu)

        alpha_obs = self._alpha_obs_selected

        err = wp - pos_enu
        dist = float(np.linalg.norm(err))

        if dist > 0.1:
            self._yaw = math.atan2(err[0], err[1])

        HOVER_THRESHOLD = 0.3
        self._at_waypoint = dist < HOVER_THRESHOLD

        if self._at_waypoint:
            wp_ned = enu_to_ned(wp) + self.origin_ned
            self._publish_position(wp_ned)
        else:
            v_nominal = k_p * err
            speed = np.linalg.norm(v_nominal)
            if speed > v_max:
                v_nominal = v_nominal / speed * v_max

            v_safe = self._cbf_boundary(pos_enu, v_nominal, boundary, alpha)

            for center, radius in self._obstacles:
                v_safe = self._cbf_obstacle(pos_enu, v_safe, center, radius, alpha_obs)

            self._publish_velocity(enu_to_ned(v_safe))

        if self._tick_count % 20 == 0:
            obs_h = [float(np.dot(pos_enu - center, pos_enu - center)) - radius**2
                     for center, radius in self._obstacles]
            self.get_logger().info(
                f'pos={pos_enu.round(2)}  err={dist:.2f}m  holding={self._at_waypoint}  '
                f'obstacles={len(self._obstacles)}  '
                f'min_obs_h={min(obs_h, default=float("inf")):.2f}  '
                f'alpha_obs={alpha_obs:.3f}{"(PENN)" if self._penn_model else "(fixed)"}')

    # ── CBF helpers ───────────────────────────────────────────────────────────

    def _cbf_boundary(self, pos, v_nominal, boundary, alpha):
        x, y, z = pos
        x_min, x_max, y_min, y_max, z_min, z_max = boundary

        vx = float(np.clip(v_nominal[0], -alpha * (x - x_min), alpha * (x_max - x)))
        vy = float(np.clip(v_nominal[1], -alpha * (y - y_min), alpha * (y_max - y)))
        vz = float(np.clip(v_nominal[2], -alpha * (z - z_min), alpha * (z_max - z)))
        return np.array([vx, vy, vz])

    def _cbf_obstacle(self, pos, v_nominal, center, radius, alpha):
        n = pos - center
        n_sq = float(np.dot(n, n))

        if n_sq < 1e-6:
            return v_nominal

        h = n_sq - radius**2
        lf_h = 2.0 * float(np.dot(n, v_nominal))
        slack = lf_h + alpha * h

        if slack >= 0:
            return v_nominal

        correction = (- slack / (2.0 * n_sq)) * n

        return v_nominal + correction

    # ── Param / publish helpers ───────────────────────────────────────────────

    def _boundary_params(self):
        gp = lambda n: self.get_parameter(n).value
        return (gp('boundary_x_min'), gp('boundary_x_max'),
                gp('boundary_y_min'), gp('boundary_y_max'),
                gp('boundary_z_min'), gp('boundary_z_max'))

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
        msg.timestamp = self._ts()
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = self._yaw
        self._setpoint_pub.publish(msg)

    def _publish_velocity(self, vel_ned):
        msg = TrajectorySetpoint()
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = self._yaw
        msg.timestamp = self._ts()
        self._setpoint_pub.publish(msg)

    def _publish_drone_viz(self, pos_enu):
        msg = Marker()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = 'drone'
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.pose.position.x = float(pos_enu[0])
        msg.pose.position.y = float(pos_enu[1])
        msg.pose.position.z = float(pos_enu[2])
        msg.pose.orientation.w = 1.0
        msg.scale.x = 0.3
        msg.scale.y = 0.3
        msg.scale.z = 0.15
        msg.color.r = 0.0
        msg.color.g = 0.8
        msg.color.b = 1.0
        msg.color.a = 1.0
        self._drone_viz_pub.publish(msg)

    def _engage_offboard(self):
        self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def _arm(self):
        self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=21196.0)

    def _vehicle_command(self, command, **kw):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(kw.get('param1', 0.0))
        msg.param2           = float(kw.get('param2', 0.0))
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self._ts()
        self._command_pub.publish(msg)

    def _ts(self):
        return int(self.get_clock().now().nanoseconds * 1e-3)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
