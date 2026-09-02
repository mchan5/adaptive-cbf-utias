"""Unit tests for GreedyCentroidTracker -- the ID-stability property that
single_vehicle_cbf_rate_arc's entry-certification / QP routing depends on."""

import numpy as np

from lidar_obstacle_publisher.tracker import GreedyCentroidTracker

MAX_JUMP = 0.5


def _cluster(x, y, z, radius=0.3, n=20):
    return (np.array([x, y, z], dtype=float), radius, n)


def test_fresh_clusters_get_monotonic_ids():
    t = GreedyCentroidTracker()
    out = t.update([_cluster(0, 0, 1), _cluster(3, 0, 1)], MAX_JUMP)
    assert sorted(out) == [0, 1]


def test_small_move_keeps_id():
    t = GreedyCentroidTracker()
    t.update([_cluster(0.0, 0.0, 1.0)], MAX_JUMP)
    out = t.update([_cluster(0.3, 0.0, 1.0)], MAX_JUMP)  # 0.3 m < MAX_JUMP
    assert list(out) == [0]
    assert np.allclose(out[0][0], [0.3, 0.0, 1.0])


def test_large_jump_gets_new_id():
    t = GreedyCentroidTracker()
    t.update([_cluster(0.0, 0.0, 1.0)], MAX_JUMP)
    out = t.update([_cluster(1.0, 0.0, 1.0)], MAX_JUMP)  # 1.0 m > MAX_JUMP
    assert list(out) == [1]


def test_disappeared_obstacle_frees_nothing_and_new_one_gets_next_id():
    t = GreedyCentroidTracker()
    t.update([_cluster(0.0, 0.0, 1.0)], MAX_JUMP)     # id 0
    t.update([], MAX_JUMP)                             # id 0 dropped
    out = t.update([_cluster(0.0, 0.0, 1.0)], MAX_JUMP)
    assert list(out) == [1]  # ids are monotonic, never reused


def test_two_clusters_do_not_collapse_onto_one_previous_id():
    t = GreedyCentroidTracker()
    t.update([_cluster(0.0, 0.0, 1.0)], MAX_JUMP)
    # id 0 both new clusters are within MAX_JUMP of the single old track, but a previous id may only
    # be claimed once -- the second cluster must get a fresh id, not share id 0.
    out = t.update([_cluster(0.1, 0.0, 1.0), _cluster(0.4, 0.0, 1.0)], MAX_JUMP)
    assert sorted(out) == [0, 1]
    assert np.allclose(out[0][0], [0.1, 0.0, 1.0])  # first cluster claimed id 0
