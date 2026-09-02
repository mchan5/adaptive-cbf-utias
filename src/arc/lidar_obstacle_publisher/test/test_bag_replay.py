"""Replay a recorded bench bag through LidarObstaclePublisherNode and check the
detections against hand-measured ground truth. Skips until fixtures/bench_replay.yaml
has a real bag_path (see fixtures/README.md)."""

import os

import pytest
import yaml

from lidar_obstacle_publisher.replay import replay, resolve_bag_path

_FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'bench_replay.yaml')


def _load_fixture():
    with open(_FIXTURE) as f:
        return yaml.safe_load(f)


def _require_bag(cfg):
    bag = cfg.get('bag_path')
    if not bag:
        pytest.skip('fixtures/bench_replay.yaml has no bag_path yet -- record a bench bag first')
    bag_path = resolve_bag_path(_FIXTURE, bag)
    if not os.path.exists(bag_path):
        pytest.skip(f'bag not found: {bag_path}')
    return bag_path


@pytest.fixture
def rclpy_ctx():
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('rosbag2_py')
    rclpy.init()
    yield
    rclpy.shutdown()


def test_bench_bag_replay(rclpy_ctx):
    cfg = _load_fixture()
    bag_path = _require_bag(cfg)

    frames, n_clouds = replay(cfg, bag_path)

    assert n_clouds > 0, 'bag contained no point clouds on ' + cfg['pointcloud_topic']
    assert frames, 'node never published -- no odom in the bag, or every cloud was dropped'

    exp = cfg['expect']
    tol = float(exp['centroid_tol_m'])
    max_r = float(exp['max_reported_radius_m'])

    # global invariant: nothing larger than the CBF-QP can clear
    for frame in frames:
        for m in frame:
            assert m.scale.x / 2.0 <= max_r + 1e-6, (
                f'marker id={m.id} radius {m.scale.x / 2.0:.3f} > max_reported_radius_m {max_r}')

    for obs in cfg['obstacles']:
        cx, cy, cz = obs['center_xyz']
        hits = []   # (frame_idx, marker.id)
        for i, frame in enumerate(frames):
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
