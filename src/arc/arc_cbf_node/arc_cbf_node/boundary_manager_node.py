#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker, MarkerArray
)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

FRAME = 'map'
SQ2 = 0.7071067811865476


def _axis_ctrl(name, qx, qy, qz, qw):
    ctrl = InteractiveMarkerControl()
    ctrl.name = name
    ctrl.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    return ctrl


def _sphere_vis_ctrl(color, scale=0.3):
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.scale = Vector3(x=scale, y=scale, z=scale)
    marker.color = color
    ctrl = InteractiveMarkerControl()
    ctrl.name = 'visual'
    ctrl.always_visible = True
    ctrl.interaction_mode = InteractiveMarkerControl.MOVE_3D
    ctrl.markers.append(marker)
    return ctrl


class BoundaryManagerNode(Node):
    def __init__(self):
        super().__init__('boundary_manager_node')

        defaults = [
            ('init_min_x', -3.0), ('init_min_y', -3.0), ('init_min_z', 0.5),
            ('init_max_x',  3.0), ('init_max_y',  3.0), ('init_max_z', 2.5),
            ('init_wp_x',   0.0), ('init_wp_y',   0.0), ('init_wp_z',  1.5),
        ]
        for name, val in defaults:
            self.declare_parameter(name, val)

        def gp(n):
            return self.get_parameter(n).value

        self._min = [gp('init_min_x'), gp('init_min_y'), gp('init_min_z')]
        self._max = [gp('init_max_x'), gp('init_max_y'), gp('init_max_z')]
        self._wp  = [gp('init_wp_x'),  gp('init_wp_y'),  gp('init_wp_z')]

        self._server = InteractiveMarkerServer(self, 'boundary_manager')

        self._min_pub     = self.create_publisher(PointStamped, '/arc/boundary_min',     10)
        self._max_pub     = self.create_publisher(PointStamped, '/arc/boundary_max',     10)
        self._wp_pub      = self.create_publisher(PointStamped, '/arc/waypoint',         10)
        self._box_pub     = self.create_publisher(Marker,       '/arc/boundary_box_viz', 10)
        self._spheres_pub = self.create_publisher(MarkerArray,  '/arc/spheres_viz',      10)

        self._make_marker('boundary_min', self._min, ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.9))
        self._make_marker('boundary_max', self._max, ColorRGBA(r=0.3, g=0.3, b=1.0, a=0.9))
        self._make_marker('waypoint',     self._wp,  ColorRGBA(r=0.3, g=1.0, b=0.3, a=0.9))
        self._server.applyChanges()

        self.create_timer(0.1, self._publish)
        self.get_logger().info(
            f'Boundary manager started — min={self._min}  max={self._max}  wp={self._wp}'
        )

    def _make_marker(self, name, pos, color):
        im = InteractiveMarker()
        im.header.frame_id = FRAME
        im.name = name
        im.scale = 0.6
        im.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        im.pose.orientation = Quaternion(w=1.0)

        im.controls.append(_sphere_vis_ctrl(color))

        # Quaternions that align the control's local x-axis with each world axis
        # so MOVE_AXIS drags along that world axis.
        im.controls.append(_axis_ctrl('move_x', SQ2, 0.0, 0.0, SQ2))
        im.controls.append(_axis_ctrl('move_y', 0.0, 0.0, SQ2, SQ2))
        im.controls.append(_axis_ctrl('move_z', 0.0, SQ2, 0.0, SQ2))

        self._server.insert(im)
        self._server.setCallback(name, self._on_feedback)

    def _on_feedback(self, feedback):
        p = feedback.pose.position
        xyz = [p.x, p.y, p.z]
        if feedback.marker_name == 'boundary_min':
            self._min = xyz
        elif feedback.marker_name == 'boundary_max':
            self._max = xyz
        elif feedback.marker_name == 'waypoint':
            self._wp = xyz
        self._server.applyChanges()

    def _pt_msg(self, xyz):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME
        msg.point = Point(x=xyz[0], y=xyz[1], z=xyz[2])
        return msg

    def _box_marker(self):
        mn, mx = self._min, self._max
        x0, y0, z0 = mn
        x1, y1, z1 = mx

        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]

        box = Marker()
        box.header.stamp = self.get_clock().now().to_msg()
        box.header.frame_id = FRAME
        box.ns = 'boundary_box'
        box.id = 0
        box.type = Marker.LINE_LIST
        box.action = Marker.ADD
        box.pose.orientation = Quaternion(w=1.0)
        box.scale.x = 0.05
        box.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)

        for a, b in edges:
            ca, cb = corners[a], corners[b]
            box.points.append(Point(x=ca[0], y=ca[1], z=ca[2]))
            box.points.append(Point(x=cb[0], y=cb[1], z=cb[2]))

        return box

    def _sphere_marker(self, mid, xyz, color, scale=0.3):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = FRAME
        m.ns = 'spheres'
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
        m.pose.orientation = Quaternion(w=1.0)
        m.scale = Vector3(x=scale, y=scale, z=scale)
        m.color = color
        return m

    def _publish(self):
        self._min_pub.publish(self._pt_msg(self._min))
        self._max_pub.publish(self._pt_msg(self._max))
        self._wp_pub.publish(self._pt_msg(self._wp))
        self._box_pub.publish(self._box_marker())

        arr = MarkerArray()
        arr.markers.append(self._sphere_marker(0, self._min, ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0)))
        arr.markers.append(self._sphere_marker(1, self._max, ColorRGBA(r=0.3, g=0.3, b=1.0, a=1.0)))
        arr.markers.append(self._sphere_marker(2, self._wp,  ColorRGBA(r=0.3, g=1.0, b=0.3, a=1.0)))
        self._spheres_pub.publish(arr)

        self._server.applyChanges()


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
