"""ROS 2 side of the experiment GUI. A single rclpy node, spun in its own QThread, that forwards
the telemetry the dashboard needs as Qt signals. Read-only in Phase 1 -- it only subscribes."""

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray
from px4_msgs.msg import VehicleStatus, BatteryStatus
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from single_vehicle_cbf_rate_arc.msg import CbfDiagnostics

# The CBF node's parameters this GUI manages (Phase 2). All are live-tunable
# in single_vehicle_cbf_rate_client.cpp's tick().
MANAGED_PARAMS = ('penn_enabled', 'cbf_gamma_obs', 'penn_gamma_min', 'penn_gamma_max')


def _wrap(value):
    pv = ParameterValue()
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = value
    elif isinstance(value, int):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = value
    else:
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
    return pv


def _unwrap(pv):
    return {
        ParameterType.PARAMETER_BOOL: pv.bool_value,
        ParameterType.PARAMETER_INTEGER: pv.integer_value,
        ParameterType.PARAMETER_DOUBLE: pv.double_value,
        ParameterType.PARAMETER_STRING: pv.string_value,
    }.get(pv.type)


def _best_effort(depth=5):
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _px4_out(depth=10):
    # Matches single_vehicle_cbf_rate_client.cpp's qos_px4_out.
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _obstacles_qos():
    # Matches synthetic_obstacle_publisher / lidar_obstacle_publisher.
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class TelemetryNode(Node, QObject):
    """rclpy Node that re-emits messages as Qt signals on the GUI thread."""

    diag = pyqtSignal(object)          # single_vehicle_cbf_rate_arc/CbfDiagnostics
    odom = pyqtSignal(object)          # nav_msgs/Odometry
    obstacles = pyqtSignal(object)     # visualization_msgs/MarkerArray
    status = pyqtSignal(object)        # px4_msgs/VehicleStatus
    battery = pyqtSignal(object)       # px4_msgs/BatteryStatus

    params_read = pyqtSignal(dict)     # {name: current_value} from the CBF node
    params_result = pyqtSignal(bool, str)  # (ok, human-readable outcome)

    def __init__(self):
        Node.__init__(self, 'arc_experiment_gui')
        QObject.__init__(self)
        self.uav_prefix = self.declare_parameter('uav_prefix', 'uav_0').value
        p = f'/{self.uav_prefix}'

        self.create_subscription(
            CbfDiagnostics, f'{p}/cbf/diagnostics', self._on_diag, _best_effort(5))
        self.create_subscription(
            Odometry, f'{p}/state_estimator/local_position/odom', self._on_odom, _best_effort(10))
        self.create_subscription(
            MarkerArray, '/arc/obstacles', self._on_obstacles, _obstacles_qos())
        # Unversioned -- PX4 v1.15.4's SITL checkout has zero publishers on both vehicle_status_v1
        # and _v4 (see px4_config/README.md's "vehicle_status_v1 vs v4" section); this is why …
        self.create_subscription(
            VehicleStatus, f'{p}/fmu/out/vehicle_status', self._on_status, _px4_out(10))
        self.create_subscription(
            BatteryStatus, f'{p}/fmu/out/battery_status', self._on_battery, _best_effort(5))

        # ---- parameter service clients (Phase 2 -- the only thing this GUI
        # writes; still not in the control path) ----
        self.cbf_node_name = f'{p}/autopilot_sv_cbf_rate_node'
        self._get_cli = self.create_client(
            GetParameters, f'{self.cbf_node_name}/get_parameters')
        self._set_cli = self.create_client(
            SetParameters, f'{self.cbf_node_name}/set_parameters')

        # Scene switching: push the frozen-scene obstacle list to the synthetic obstacle publisher
        # as its 'obstacles' param.
        self.obs_node_name = self.declare_parameter(
            'obstacle_node', f'{p}/synthetic_obstacle_publisher').value
        self._obs_set_cli = self.create_client(
            SetParameters, f'{self.obs_node_name}/set_parameters')

        # Operator command publishers -- rare, deliberate, not a stream.
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self._goal_pub = self.create_publisher(PointStamped, f'{p}/goal_waypoint', goal_qos)
        self._arm_pub = self.create_publisher(Bool, f'{p}/operator/arm_confirm', 1)

        # Graph / service-discovery state, refreshed on a timer that runs INSIDE the executor
        # thread.
        self._graph_cache = {
            'node_names': set(),
            'params_ready': False,
            'scene_ready': False,
            'obs_pub_count': -1,
        }
        self.create_timer(1.0, self._refresh_graph_cache)

        self.get_logger().info(
            f'arc_experiment_gui telemetry node up, namespace /{self.uav_prefix}')

    def _refresh_graph_cache(self):
        try:
            names = {n for n, _ns in self.get_node_names_and_namespaces()}
        except Exception:  # noqa: BLE001
            names = self._graph_cache['node_names']
        try:
            obs_n = self.count_publishers('/arc/obstacles')
        except Exception:  # noqa: BLE001
            obs_n = -1
        self._graph_cache = {
            'node_names': names,
            'params_ready': (self._get_cli.service_is_ready()
                             and self._set_cli.service_is_ready()),
            'scene_ready': self._obs_set_cli.service_is_ready(),
            'obs_pub_count': obs_n,
        }

    def publish_goal(self, x, y, z, frame_id='map'):
        m = PointStamped()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x, m.point.y, m.point.z = float(x), float(y), float(z)
        self._goal_pub.publish(m)

    def publish_arm_confirm(self):
        self._arm_pub.publish(Bool(data=True))

    def obstacle_publisher_count(self):
        return self._graph_cache['obs_pub_count']

    # ---- parameter access (call from the GUI thread; results come back as
    # Qt signals marshalled by the spinning executor) ----
    def params_service_ready(self):
        return self._graph_cache['params_ready']

    def refresh_params(self):
        if not self._get_cli.service_is_ready():
            self.params_result.emit(False, 'CBF node param service not available')
            return
        req = GetParameters.Request(names=list(MANAGED_PARAMS))
        self._get_cli.call_async(req).add_done_callback(self._on_get_done)

    def _on_get_done(self, fut):
        try:
            res = fut.result()
        except Exception as exc:  # noqa: BLE001
            self.params_result.emit(False, f'param read failed: {exc}')
            return
        self.params_read.emit(
            {n: _unwrap(v) for n, v in zip(MANAGED_PARAMS, res.values)})

    def set_params(self, values):
        """values: {name: python_value}. Applies atomically, then re-reads."""
        if not self._set_cli.service_is_ready():
            self.params_result.emit(False, 'CBF node param service not available')
            return
        params = [Parameter(name=n, value=_wrap(v)) for n, v in values.items()]
        req = SetParameters.Request(parameters=params)
        names = ', '.join(values)
        self._set_cli.call_async(req).add_done_callback(
            lambda f, nm=names: self._on_set_done(f, nm))

    def _on_set_done(self, fut, names):
        try:
            res = fut.result()
            ok = all(r.successful for r in res.results)
            why = '' if ok else next((r.reason for r in res.results if not r.successful), '')
            self.params_result.emit(ok, f'set [{names}] {"ok" if ok else "REJECTED " + why}')
        except Exception as exc:  # noqa: BLE001
            self.params_result.emit(False, f'set [{names}] failed: {exc}')
        self.refresh_params()

    def scene_service_ready(self):
        return self._graph_cache['scene_ready']

    def set_scene_obstacles(self, flat):
        """flat: [x, y, z, diameter, ...]. Pushes it to the synthetic obstacle
        publisher's 'obstacles' parameter (live, no relaunch)."""
        if not self._obs_set_cli.service_is_ready():
            self.params_result.emit(
                False, 'obstacle publisher param service not available')
            return
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
        pv.double_array_value = [float(v) for v in flat]
        req = SetParameters.Request(
            parameters=[Parameter(name='obstacles', value=pv)])
        self._obs_set_cli.call_async(req).add_done_callback(
            lambda f: self._on_scene_set_done(f, len(flat) // 4))

    def _on_scene_set_done(self, fut, n_obs):
        try:
            res = fut.result()
            ok = all(r.successful for r in res.results)
            why = '' if ok else next(
                (r.reason for r in res.results if not r.successful), '')
            self.params_result.emit(
                ok, f'scene obstacles ({n_obs}) '
                    f'{"pushed" if ok else "REJECTED " + why}')
        except Exception as exc:  # noqa: BLE001
            self.params_result.emit(False, f'scene push failed: {exc}')

    def graph_node_names(self):
        return self._graph_cache['node_names']

    def _on_diag(self, msg):
        self.diag.emit(msg)

    def _on_odom(self, msg):
        self.odom.emit(msg)

    def _on_obstacles(self, msg):
        self.obstacles.emit(msg)

    def _on_status(self, msg):
        self.status.emit(msg)

    def _on_battery(self, msg):
        self.battery.emit(msg)


class RosSpinThread(QThread):
    """Owns the executor spin loop so the Qt event loop is never blocked."""

    def __init__(self, node):
        super().__init__()
        self._node = node
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)

    def run(self):
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the GUI
            self._node.get_logger().error(f'executor stopped: {exc}')

    def stop(self):
        self._executor.shutdown()
        self.wait(2000)
