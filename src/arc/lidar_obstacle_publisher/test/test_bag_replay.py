"""Replay a recorded bench bag through LidarObstaclePublisherNode and check
the detections against hand-measured ground truth.

This is the integration test SITL can't be -- it exercises the real
sensor->world transform, voxel clustering, and ID tracker against actual
Livox point clouds. It is inert until someone records data:

  * fixtures/bench_replay.yaml ships with bag_path: null  -> the test SKIPS
  * after a bench session (see fixtures/README.md) bag_path points at a
    rosbag2 directory and the obstacle centres are filled in -> it RUNS

Runs as plain pytest (no ROS graph / executor): bag messages are fed straight
into the node's callbacks in recorded order, and the MarkerArray the node
would publish is captured via a stub publisher.
"""

import os

import pytest
import yaml

_FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'bench_replay.yaml')


def _load_fixture():
    with open(_FIXTURE) as f:
        return yaml.safe_load(f)


def _require_data(cfg):
    bag = cfg.get('bag_path')
    if not bag:
        pytest.skip('fixtures/bench_replay.yaml has no bag_path yet -- record a bench bag first')
    bag_path = os.path.join(os.path.dirname(_FIXTURE), bag)
    if not os.path.exists(bag_path):
        pytest.skip(f'bag not found: {bag_path}')
    return bag_path


def _read_bag(bag_path, storage_id):
    """Yield (topic, deserialized_msg) in recorded order."""
    rosbag2_py = pytest.importorskip('rosbag2_py')
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


class _StubPub:
    def __init__(self):
        self.frames = []   # list[list[Marker]]

    def publish(self, marker_array):
        self.frames.append(list(marker_array.markers))


@pytest.fixture
def rclpy_ctx():
    rclpy = pytest.importorskip('rclpy')
    rclpy.init()
    yield
    rclpy.shutdown()


def test_bench_bag_replay(rclpy_ctx):
    cfg = _load_fixture()
    bag_path = _require_data(cfg)

    from rclpy.parameter import Parameter
    from lidar_obstacle_publisher.lidar_obstacle_publisher_node import LidarObstaclePublisherNode

    overrides = [Parameter(k, value=v) for k, v in (cfg.get('params') or {}).items()]
    node = LidarObstaclePublisherNode(parameter_overrides=overrides)
    stub = _StubPub()
    node._pub = stub
    try:
        pc_topic = cfg['pointcloud_topic']
        odom_topic = cfg['odom_topic']
        n_clouds = 0
        for topic, msg in _read_bag(bag_path, cfg.get('storage_id')):
            if topic == odom_topic:
                node._on_odom(msg)
            elif topic == pc_topic:
                node._on_pointcloud(msg)
                n_clouds += 1
    finally:
        node.destroy_node()

    assert n_clouds > 0, 'bag contained no point clouds on ' + cfg['pointcloud_topic']
    assert stub.frames, 'node never published -- no odom in the bag, or every cloud was dropped'

    exp = cfg['expect']
    tol = float(exp['centroid_tol_m'])
    max_r = float(exp['max_reported_radius_m'])

    # global invariant: nothing larger than the CBF-QP can clear
    for frame in stub.frames:
        for m in frame:
            assert m.scale.x / 2.0 <= max_r + 1e-6, (
                f'marker id={m.id} radius {m.scale.x / 2.0:.3f} > max_reported_radius_m {max_r}')

    for obs in cfg['obstacles']:
        cx, cy, cz = obs['center_xyz']
        hits = []   # (frame_idx, marker.id)
        for i, frame in enumerate(stub.frames):
            best = None
            for m in frame:
                d = ((m.pose.position.x - cx) ** 2 + (m.pose.position.y - cy) ** 2
                     + (m.pose.position.z - cz) ** 2) ** 0.5
                if d <= tol and (best is None or d < best[1]):
                    best = (m.id, d)
            if best is not None:
                hits.append((i, best[0]))

        assert len(hits) >= int(exp['min_detections']), (
            f"{obs['name']}: matched in {len(hits)} frames, need {exp['min_detections']} "
            f"(centre {obs['center_xyz']}, tol {tol} m)")

        if exp.get('require_stable_id'):
            ids = {mid for _i, mid in hits}
            assert len(ids) == 1, (
                f"{obs['name']}: marker.id churned across matched frames: {sorted(ids)}")
