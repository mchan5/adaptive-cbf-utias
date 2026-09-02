#!/usr/bin/env python3
"""Real-hardware analog of sitl_test_support's ground_station_stub_node."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool
from px4_msgs.msg import VehicleCommand, VehicleStatus


class OperatorArmNode(Node):
    def __init__(self):
        super().__init__('operator_arm_node')

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._vehicle_command_pub = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', cmd_qos)
        # Matches single_vehicle_cbf_rate_client.cpp's vehicle_status_sub_ topic -- verify against
        # the real Pixhawk's firmware build before flying; PX4 v1.15.4's SITL checkout publishes …
        self.create_subscription(VehicleStatus, 'fmu/out/vehicle_status', self._on_vehicle_status, cmd_qos)
        self.create_subscription(Bool, 'operator/arm_confirm', self._on_arm_confirm, 10)

        self._armed = False
        self._arm_in_progress = False

        self.get_logger().info(
            'operator_arm_node started -- waiting for a rising edge on operator/arm_confirm '
            'before switching to offboard and arming. No auto-arm.')

    def _on_vehicle_status(self, msg: VehicleStatus):
        now_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if now_armed != self._armed:
            self.get_logger().warn(f'arming_state changed: {"ARMED" if self._armed else "DISARMED"} '
                                    f'-> {"ARMED" if now_armed else "DISARMED"}')
        self._armed = now_armed
        if self._armed:
            self._arm_in_progress = False

    def _on_arm_confirm(self, msg: Bool):
        if not msg.data:
            return
        if self._armed:
            self.get_logger().info('arm_confirm received but already armed -- ignoring')
            return
        if self._arm_in_progress:
            self.get_logger().info('arm_confirm received but an arm attempt is already in flight -- ignoring')
            return

        self._arm_in_progress = True
        self.get_logger().warn('Operator-confirmed arm sequence starting: offboard mode, then ARM')
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
    node = OperatorArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
