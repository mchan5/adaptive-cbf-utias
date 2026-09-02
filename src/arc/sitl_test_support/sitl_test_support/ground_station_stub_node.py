#!/usr/bin/env python3
"""SITL-only stand-in for the missing ground-station GUI.

Publishes a fixed hold-position reference and enables the CBF blend, then
arms the vehicle and switches it to offboard mode once single_vehicle_cbf's
own OffboardControlMode heartbeat has had time to reach PX4. Scoped for
bench/SITL testing only -- not a replacement for the real ground station.

Goal sequencing (hover_first, default on): on the arm rising edge the stub
first commands a hover directly above the arming point at hover_z, holds
there until the vehicle has settled within hover_pos_tol for
hover_settle_sec (or hover_timeout_sec elapses), and only then advances
goal_waypoint to the mission waypoint. This gives every SITL run a clean
takeoff-and-stabilise phase before it drives into the obstacle course,
instead of climbing and translating toward the goal from the ground.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleCommand, VehicleStatus
from fsc_autopilot_ros2_msgs.msg import PositionControllerReference


class GroundStationStubNode(Node):
    def __init__(self):
        super().__init__('ground_station_stub_node')

        self.declare_parameter('waypoint_x', 0.0)
        self.declare_parameter('waypoint_y', 0.0)
        self.declare_parameter('waypoint_z', 1.5)
        self.declare_parameter('arm_delay_sec', 5.0)
        self.declare_parameter('arm_retry_period_sec', 1.0)
        # publish_goal: when false, this node still auto-arms and publishes the
        # PositionControllerReference/CBF_flag heartbeat, but does NOT publish
        # goal_waypoint -- an external driver (the operator GUI) owns the goal.
        # Without this, the stub's 10 Hz goal_waypoint publish overrides any
        # slower external goal within one stub period. Default true keeps the
        # standalone unattended-SITL behaviour the batch scripts rely on.
        self.declare_parameter('publish_goal', True)
        # hover_first: takeoff-and-stabilise phase before the mission goal.
        # See module docstring. Disable (hover_first:=false) to restore the
        # old "publish the mission waypoint from t=0" behaviour.
        self.declare_parameter('hover_first', True)
        self.declare_parameter('hover_z', 1.5)
        self.declare_parameter('hover_pos_tol', 0.35)
        self.declare_parameter('hover_settle_sec', 3.0)
        self.declare_parameter('hover_timeout_sec', 20.0)

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        flag_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        # Matches px4_odom_bridge_node's publisher QoS (RELIABLE / VOLATILE).
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self._ref_pub = self.create_publisher(
            PositionControllerReference, 'fsc_autopilot_ros2/position_controller/reference', flag_qos)
        # goal_waypoint (PointStamped) -- since d29ee86 the CBF node holds
        # its position on arm and only leaves it on a goal_waypoint message
        # (single_vehicle_cbf_rate_client.cpp goalCallback). The stub
        # predates that; without this publisher the vehicle arms and never
        # flies the mission. Published every tick so it overrides the
        # arm-time hold-in-place within one stub period.
        self._goal_pub = self.create_publisher(PointStamped, 'goal_waypoint', flag_qos)
        self._cbf_flag_pub = self.create_publisher(Bool, 'ground_station/CBF_flag', flag_qos)
        self._enu_flag_pub = self.create_publisher(Bool, 'ground_station/ENU_flag', flag_qos)
        self._vehicle_command_pub = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', cmd_qos)
        # The versioned suffix PX4's uxrce_dds_client publishes under has
        # drifted across firmware builds (v1.17.0 published vehicle_status_v1,
        # not _v4). As of 2026-09-01, PX4-Autopilot v1.15.4's dds_topics.yaml
        # defines only the unversioned "vehicle_status" -- confirmed via
        # `ros2 topic info -v` (both _v1 and _v4 have zero publishers on this
        # build). Use the unversioned name; see px4_config/README.md's
        # "vehicle_status_v1 vs v4" section.
        self.create_subscription(VehicleStatus, 'fmu/out/vehicle_status', self._on_vehicle_status, cmd_qos)
        # Odometry (ENU, from px4_odom_bridge_node) -- only used to place the
        # hover point above the arming spot and to detect that the vehicle
        # has settled there before the mission goal is released.
        self.create_subscription(
            Odometry, 'state_estimator/local_position/odom', self._on_odom, odom_qos)

        self._armed = False
        self._last_arm_attempt = None
        self._start_time = self.get_clock().now()

        self._pos = None  # (x, y, z) ENU, latest odom
        # Goal phase: 'pre_arm' -> 'hover' -> 'mission'. 'hover' is skipped
        # entirely when hover_first is false.
        self._phase = 'pre_arm'
        self._hover_xy = None       # captured from odom on the arm rising edge
        self._hover_start = None    # clock time the hover phase began
        self._settle_start = None   # clock time the vehicle first entered hover_pos_tol

        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'ground_station_stub_node started: waypoint=({self.get_parameter("waypoint_x").value}, '
            f'{self.get_parameter("waypoint_y").value}, {self.get_parameter("waypoint_z").value}), '
            f'publish_goal={self.get_parameter("publish_goal").value}, '
            f'hover_first={self.get_parameter("hover_first").value} '
            f'(hover_z={self.get_parameter("hover_z").value}), '
            f'first arm attempt in {self.get_parameter("arm_delay_sec").value:.1f}s, '
            f'retrying every {self.get_parameter("arm_retry_period_sec").value:.1f}s until armed')

    def _on_vehicle_status(self, msg: VehicleStatus):
        now_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if now_armed and not self._armed:
            # Arm rising edge -- start the hover phase (or go straight to the
            # mission goal if hover_first is off).
            if self.get_parameter('hover_first').value:
                self._phase = 'hover'
                self._hover_start = self.get_clock().now()
                self._settle_start = None
                self._hover_xy = (self._pos[0], self._pos[1]) if self._pos is not None else None
                if self._hover_xy is not None:
                    self.get_logger().info(
                        f'armed: hovering at [{self._hover_xy[0]:.3f}, {self._hover_xy[1]:.3f}, '
                        f'{self.get_parameter("hover_z").value:.3f}] before releasing the mission goal')
                else:
                    self.get_logger().warning(
                        'armed before odometry valid -- will capture the hover point '
                        'from the first odom message')
            else:
                self._phase = 'mission'
        self._armed = now_armed

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pos = (p.x, p.y, p.z)

    def _mission_goal(self):
        return (self.get_parameter('waypoint_x').value,
                self.get_parameter('waypoint_y').value,
                self.get_parameter('waypoint_z').value)

    def _update_phase(self):
        """Advance the hover phase toward 'mission' once the vehicle has
        settled at the hover point (or the timeout fires)."""
        if self._phase != 'hover':
            return
        hover_z = self.get_parameter('hover_z').value

        if self._hover_xy is None:
            if self._pos is None:
                return
            self._hover_xy = (self._pos[0], self._pos[1])
            self._hover_start = self.get_clock().now()
            self.get_logger().info(
                f'hover point captured at [{self._hover_xy[0]:.3f}, {self._hover_xy[1]:.3f}, '
                f'{hover_z:.3f}]')

        now = self.get_clock().now()
        elapsed = (now - self._hover_start).nanoseconds * 1e-9

        settled = False
        if self._pos is not None:
            target = (self._hover_xy[0], self._hover_xy[1], hover_z)
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self._pos, target)))
            if dist <= self.get_parameter('hover_pos_tol').value:
                if self._settle_start is None:
                    self._settle_start = now
                settled = (now - self._settle_start).nanoseconds * 1e-9 >= \
                    self.get_parameter('hover_settle_sec').value
            else:
                self._settle_start = None

        if settled:
            self.get_logger().info(
                f'hover settled after {elapsed:.1f}s -- releasing mission goal '
                f'{self._mission_goal()}')
            self._phase = 'mission'
        elif elapsed >= self.get_parameter('hover_timeout_sec').value:
            self.get_logger().warning(
                f'hover did not settle within {elapsed:.1f}s -- releasing mission goal anyway')
            self._phase = 'mission'

    def _active_goal(self):
        """The (x, y, z) the stub is currently commanding."""
        if self._phase == 'hover':
            hover_z = self.get_parameter('hover_z').value
            if self._hover_xy is not None:
                return (self._hover_xy[0], self._hover_xy[1], hover_z)
            # Armed but no odom yet: command a straight climb over the SITL
            # spawn origin rather than leaking the mission goal for a tick.
            return (0.0, 0.0, hover_z)
        return self._mission_goal()

    def _tick(self):
        self._update_phase()
        gx, gy, gz = self._active_goal()

        ref = PositionControllerReference()
        ref.header.stamp = self.get_clock().now().to_msg()
        ref.position.x = gx
        ref.position.y = gy
        ref.position.z = gz
        ref.yaw = 0.0
        ref.yaw_unit = PositionControllerReference.RADIANS
        self._ref_pub.publish(ref)

        if self.get_parameter('publish_goal').value:
            goal = PointStamped()
            goal.header.stamp = ref.header.stamp
            goal.point.x = gx
            goal.point.y = gy
            goal.point.z = gz
            self._goal_pub.publish(goal)

        self._cbf_flag_pub.publish(Bool(data=True))
        self._enu_flag_pub.publish(Bool(data=False))

        if self._armed:
            return

        now = self.get_clock().now()
        elapsed = (now - self._start_time).nanoseconds * 1e-9
        if elapsed < self.get_parameter('arm_delay_sec').value:
            return

        retry_period = self.get_parameter('arm_retry_period_sec').value
        if self._last_arm_attempt is not None:
            since_last = (now - self._last_arm_attempt).nanoseconds * 1e-9
            if since_last < retry_period:
                return

        self._last_arm_attempt = now
        self._engage_offboard_mode()
        self._arm()

    def _arm(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def _engage_offboard_mode(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Switching to offboard mode')

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GroundStationStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
