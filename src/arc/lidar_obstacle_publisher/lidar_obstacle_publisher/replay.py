"""Replay a recorded bag through LidarObstaclePublisherNode offline.

Shared by test/test_bag_replay.py (pass/fail) and tools/eval_detection_accuracy.py
(metrics report). Needs rclpy + rosbag2_py, so it only runs in a sourced ROS
environment; the caller owns rclpy.init()/shutdown().
"""

import os


def read_bag(bag_path, storage_id):
    """Yield ``(topic, deserialized_msg)`` in recorded order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id or ''),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic in types:
            yield topic, deserialize_message(data, get_message(types[topic]))


def resolve_bag_path(fixture_path, bag_path):
    """A bag_path in the fixture is relative to the fixture's own directory."""
    return os.path.join(os.path.dirname(fixture_path), bag_path)


class _CapturePub:
    def __init__(self):
        self.frames = []          # list[list[Marker]], one entry per publish

    def publish(self, marker_array):
        self.frames.append(list(marker_array.markers))


def replay(cfg, bag_path):
    """Construct the node with ``cfg['params']`` as overrides, feed it the bag's
    odom + point-cloud messages in order, and return
    ``(frames, n_clouds)`` where ``frames`` is the list of published marker
    lists. Caller must have called ``rclpy.init()``."""
    from rclpy.parameter import Parameter
    from lidar_obstacle_publisher.lidar_obstacle_publisher_node import LidarObstaclePublisherNode

    overrides = [Parameter(k, value=v) for k, v in (cfg.get('params') or {}).items()]
    node = LidarObstaclePublisherNode(parameter_overrides=overrides)
    pub = _CapturePub()
    node._pub = pub

    pc_topic = cfg['pointcloud_topic']
    odom_topic = cfg['odom_topic']
    n_clouds = 0
    try:
        for topic, msg in read_bag(bag_path, cfg.get('storage_id')):
            if topic == odom_topic:
                node._on_odom(msg)
            elif topic == pc_topic:
                node._on_pointcloud(msg)
                n_clouds += 1
    finally:
        node.destroy_node()
    return pub.frames, n_clouds
