"""Unit tests for the spatial pre-filters -- the body-frame self-filter and the
world-frame arena ROI -- that gate the cloud before clustering."""

import numpy as np

from lidar_obstacle_publisher.filters import roi_mask, self_filter_mask


def test_self_filter_drops_points_inside_the_box():
    pts = np.array([
        [0.0, 0.0, 0.0],     # dead centre -- the drone body
        [0.2, -0.1, 0.05],   # inside on every axis
        [0.6, 0.0, 0.0],     # outside in x
        [0.0, 0.0, 1.5],     # outside in z (well above the airframe)
    ])
    keep = self_filter_mask(pts, [0.35, 0.35, 0.20])
    assert list(keep) == [False, False, True, True]


def test_self_filter_all_zero_half_extents_keeps_everything():
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    assert self_filter_mask(pts, [0.0, 0.0, 0.0]).all()


def test_self_filter_face_is_exclusive():
    # a point exactly on a box face is NOT inside (strict <), so it survives
    pts = np.array([[0.35, 0.0, 0.0], [0.0, 0.0, 0.20]])
    assert self_filter_mask(pts, [0.35, 0.35, 0.20]).all()


def test_self_filter_needs_all_three_axes_inside():
    # inside x and y but outside z -> kept (it's above/below the airframe)
    pts = np.array([[0.1, 0.1, 0.9]])
    assert self_filter_mask(pts, [0.35, 0.35, 0.20]).all()


def test_self_filter_handles_empty_cloud():
    keep = self_filter_mask(np.empty((0, 3)), [0.35, 0.35, 0.20])
    assert keep.shape == (0,)


def test_roi_keeps_only_points_inside_the_xy_box():
    pts = np.array([
        [0.0, 0.0, 1.0],     # inside
        [1.5, 3.0, 0.2],     # inside (z is ignored by the ROI)
        [-2.0, 0.0, 1.0],    # outside in x
        [0.0, 9.0, 1.0],     # outside in y
    ])
    keep = roi_mask(pts, [-1.5, -0.5], [1.5, 6.5])
    assert list(keep) == [True, True, False, False]


def test_roi_boundary_is_inclusive():
    pts = np.array([[1.5, 6.5, 0.0], [-1.5, -0.5, 0.0]])
    assert roi_mask(pts, [-1.5, -0.5], [1.5, 6.5]).all()


def test_roi_handles_empty_cloud():
    keep = roi_mask(np.empty((0, 3)), [-1.5, -0.5], [1.5, 6.5])
    assert keep.shape == (0,)
