#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

FRAME = 'map'


class ObstaclePublisher(Node):

    def __init__(self):
        super().__init__('obstacle_publisher')

        # flat list [x, y, z, r, x, y, z, r, ...]
        self.declare_parameter('obstacles', [0.0, 0.0, 1.5, 0.5])

        self._data_pub = self.create_publisher(MarkerArray, '/arc/obstacles',    10)
        self._viz_pub  = self.create_publisher(MarkerArray, '/arc/obstacle_viz', 10)

        self.create_timer(0.1, self._publish)
        self.get_logger().info('Obstacle publisher started.')

    def _parse(self):
        raw = self.get_parameter('obstacles').value
        if len(raw) % 4 != 0:
            self.get_logger().warn(
                f'obstacles parameter has {len(raw)} values — must be a multiple of 4 (x,y,z,r). '
                'Ignoring last incomplete entry.', throttle_duration_sec=5.0)
        return [(raw[i], raw[i+1], raw[i+2], raw[i+3])
                for i in range(0, len(raw) - 3, 4)]

    def _make_marker(self, idx, x, y, z, r, alpha=0.6):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = FRAME
        m.ns              = 'obstacles'
        m.id              = idx
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position   = Point(x=float(x), y=float(y), z=float(z))
        m.pose.orientation = Quaternion(w=1.0)
        # scale = diameter for a sphere marker
        m.scale           = Vector3(x=float(r * 2), y=float(r * 2), z=float(r * 2))
        m.color           = ColorRGBA(r=1.0, g=0.5, b=0.0, a=alpha)  # orange
        return m

    def _publish(self):
        obstacles = self._parse()

        data_arr = MarkerArray()
        viz_arr  = MarkerArray()

        for idx, (x, y, z, r) in enumerate(obstacles):
            # data marker: obstacle_cbf_node reads position + scale.x (diameter)
            data_arr.markers.append(self._make_marker(idx, x, y, z, r, alpha=1.0))
            # viz marker: same geometry, slightly transparent for better visibility
            viz_arr.markers.append(self._make_marker(idx, x, y, z, r, alpha=0.5))

        # if obstacle list shrank, delete stale markers in RViz2
        if not obstacles:
            delete = Marker()
            delete.action = Marker.DELETEALL
            viz_arr.markers.append(delete)

        self._data_pub.publish(data_arr)
        self._viz_pub.publish(viz_arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
