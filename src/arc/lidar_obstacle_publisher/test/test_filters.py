"""Unit tests for the spatial pre-filters -- the body-frame self-filter cylinder
and the world-frame arena ROI -- that gate the cloud before clustering."""

import numpy as np

from lidar_obstacle_publisher.filters import roi_mask, self_filter_mask

R, ZLO, ZHI = 0.40, -0.50, 0.50


def test_self_filter_drops_points_inside_the_cylinder():
    pts = np.array([
        [0.0, 0.0, 0.0],      # dead centre -- the drone body
        [0.25, -0.20, 0.10],  # inside: r=0.32 < 0.40, z in band
        [0.30, 0.30, 0.0],    # outside: r=0.42 > 0.40
        [0.1, 0.0, 0.9],      # outside: z above the band (nothing to fly over here, but be safe)
        [0.1, 0.0, -0.8],     # outside: z below the band
    ])
    keep = self_filter_mask(pts, R, ZLO, ZHI)
    assert list(keep) == [False, False, True, True, True]


def test_self_filter_radius_boundary_is_exclusive_z_is_inclusive():
    pts = np.array([
        [R, 0.0, 0.0],        # exactly on the radius -> kept (strict <)
        [0.0, 0.0, ZHI],      # exactly on the z ceiling -> dropped (<=)
        [0.0, 0.0, ZLO],      # exactly on the z floor -> dropped (>=)
    ])
    assert list(self_filter_mask(pts, R, ZLO, ZHI)) == [True, False, False]


def test_self_filter_handles_empty_cloud():
    keep = self_filter_mask(np.empty((0, 3)), R, ZLO, ZHI)
    assert keep.shape == (0,)


def test_self_filter_a_far_arm_point_outside_the_z_band_survives():
    # a point on a mast well above the airframe is not "self" for our mounts
    assert self_filter_mask(np.array([[0.1, 0.1, 1.2]]), R, ZLO, ZHI).all()


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
