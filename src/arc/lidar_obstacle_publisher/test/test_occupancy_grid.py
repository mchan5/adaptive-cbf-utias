"""Unit tests for OccupancyGridClusterer -- the temporal 2D occupancy grid that
replaced per-frame 3D voxel clustering."""

import numpy as np

from lidar_obstacle_publisher.occupancy_grid import OccupancyGridClusterer, _label_8


def _grid():
    # 4 m x 4 m centred on the origin, 0.1 m cells -> 40 x 40
    return OccupancyGridClusterer(origin_xy=[-2.0, -2.0], size_xy=[4.0, 4.0], resolution_m=0.1)


def _blob(cx, cy, z=1.0, n=40, spread=0.12):
    rng = np.random.default_rng(0)
    xy = rng.normal([cx, cy], spread, size=(n, 2))
    return np.column_stack([xy, np.full(n, z)])


def test_shape_rounds_up():
    assert _grid().shape == (40, 40)


def test_repeated_frames_saturate_a_cell_past_threshold():
    g = _grid()
    pts = _blob(0.0, 0.0)
    # one frame: cells seen once -> occupancy == hit_increment (0.35) < 0.5
    g.add_cloud(pts, hit_increment=0.35)
    assert g.clusters(threshold=0.5, min_cells=4) == []
    # two more frames without decay -> 3 * 0.35 clipped to 1.0, well above threshold
    g.add_cloud(pts, hit_increment=0.35)
    g.add_cloud(pts, hit_increment=0.35)
    cl = g.clusters(threshold=0.5, min_cells=4)
    assert len(cl) == 1
    assert np.linalg.norm(cl[0]['centroid_xy'] - [0.0, 0.0]) < 0.15


def test_decay_drops_a_stale_cell_below_threshold():
    g = _grid()
    pts = _blob(0.0, 0.0)
    for _ in range(3):
        g.add_cloud(pts, hit_increment=0.35)     # saturated (~1.0)
    assert len(g.clusters(threshold=0.5, min_cells=4)) == 1
    # no more hits; 3 half-lives of elapsed time -> ~1/8 of 1.0
    g.decay(dt=1.2, half_life_s=0.4)
    assert g.clusters(threshold=0.5, min_cells=4) == []


def test_two_separated_blobs_are_two_clusters_adjacent_ones_merge():
    g = _grid()
    for _ in range(3):
        g.add_cloud(_blob(-1.0, 0.0), hit_increment=0.5)
        g.add_cloud(_blob(1.0, 0.0), hit_increment=0.5)
    assert len(g.clusters(threshold=0.5, min_cells=4)) == 2

    g2 = _grid()
    for _ in range(3):
        g2.add_cloud(_blob(0.0, 0.0), hit_increment=0.5)
        g2.add_cloud(_blob(0.15, 0.0), hit_increment=0.5)   # overlapping
    assert len(g2.clusters(threshold=0.5, min_cells=4)) == 1


def test_min_cells_rejects_a_tiny_component():
    g = _grid()
    one_point = np.array([[0.0, 0.0, 1.0]])
    for _ in range(3):
        g.add_cloud(one_point, hit_increment=0.5)
    assert g.clusters(threshold=0.5, min_cells=4) == []
    assert len(g.clusters(threshold=0.5, min_cells=1)) == 1


def test_radius_is_clamped_to_max_radius():
    g = _grid()
    wide = np.column_stack([
        np.linspace(-1.5, 1.5, 200),
        np.zeros(200),
        np.ones(200),
    ])
    for _ in range(3):
        g.add_cloud(wide, hit_increment=0.5)
    cl = g.clusters(threshold=0.5, min_cells=4, max_radius=0.5)
    assert cl and all(c['radius'] <= 0.5 + 1e-9 for c in cl)


def test_out_of_bounds_points_are_ignored():
    g = _grid()
    pts = np.array([[100.0, 100.0, 1.0], [-50.0, 0.0, 1.0]])
    for _ in range(3):
        g.add_cloud(pts, hit_increment=0.5)
    assert g.occupied_cell_count(0.5) == 0
    assert g.clusters() == []


def test_vertical_span_is_captured_and_reset_after_decay_to_empty():
    g = _grid()
    tall = _blob(0.0, 0.0, z=0.4)
    tall = np.vstack([tall, _blob(0.0, 0.0, z=2.1)])
    for _ in range(3):
        g.add_cloud(tall, hit_increment=0.5)
    c = g.clusters(threshold=0.5, min_cells=4)[0]
    assert c['z_min'] < 0.5 and c['z_max'] > 2.0

    g.decay(dt=5.0, half_life_s=0.4)          # fully forgotten
    low = _blob(0.0, 0.0, z=1.0)
    for _ in range(3):
        g.add_cloud(low, hit_increment=0.5, reset_epsilon=0.05)
    c2 = g.clusters(threshold=0.5, min_cells=4)[0]
    assert 0.8 < c2['z_min'] and c2['z_max'] < 1.2   # old 0.4/2.1 span did not stick


def test_label_8_is_diagonally_connected():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True   # a diagonal chain
    labels = _label_8(mask)
    assert labels.max() == 1
    assert len({labels[1, 1], labels[2, 2], labels[3, 3]}) == 1
